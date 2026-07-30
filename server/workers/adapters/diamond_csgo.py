"""
DIAMOND(CS:GO) 어댑터 — eloialonso/diamond **`csgo` 브랜치** 실제 추론 코드에 맞춰 구현.

⚠ 이 파일은 `diamond_atari.py`와 별도 저장소 체크아웃을 요구합니다.

DIAMOND의 CS:GO 월드모델은 `main` 브랜치가 아니라 별도의 `csgo` 브랜치에만 존재합니다.
base denoiser + upsampler 2단계 구조, 51차원 연속 액션(원핫 키+마우스), 그리고 실제
녹화된 CS:GO 프레임(spawn dataset)으로 컨텍스트를 워밍업하는 방식이 Atari 쪽과 아키텍처
자체가 달라서(`models/diffusion`, `agent.py`, `envs/world_model_env.py` 전부 다른 코드)
같은 파이썬 프로세스/PYTHONPATH에 두 브랜치를 동시에 얹을 수 없습니다.

    git clone -b csgo https://github.com/eloialonso/diamond <경로>

로 **`models/diamond`와는 다른 별도 디렉터리**에 csgo 브랜치를 clone 하고,
`WM_DIAMOND_CSGO_REPO` 가 그 경로(저장소 루트)를 가리키게 하세요.
(`models/diamond` 자체는 `diamond_atari.py`가 쓰는 `main` 브랜치 그대로 둡니다.)

이 파일은 저장소의 `src/play.py`(`prepare_play_mode`)를 거의 그대로 감싼 것입니다 —
Atari 어댑터와 달리 CS:GO는 `Agent`(denoiser+upsampler 묶음)와 `WorldModelEnv`(2단계
샘플링 + 실제 프레임 기반 워밍업)를 원본 그대로 재사용하는 편이 안전합니다: 자체
재구현하면 upsampler 컨디셔닝·burn-in 로직을 놓치기 쉽습니다.

전제:
  * diamond 저장소의 `csgo` 브랜치 `src/` 가 PYTHONPATH 에 있어야 합니다.
  * 체크포인트 1개: `csgo.pt` (denoiser + upsampler 포함, rew_end_model/actor_critic 없음
    — `config/agent/csgo.yaml` 에서 `rew_end_model: null, actor_critic: null`).
  * spawn 데이터셋 — 실제 CS:GO 녹화에서 뽑은 초기 컨텍스트 프레임 모음. 체크포인트와
    함께 배포됩니다:
        huggingface-cli download eloialonso/diamond --include "csgo/*" --local-dir <저장소>/downloads
    spawn 디렉터리는 `<다운로드>/csgo/spawn` 아래에 있습니다. `WM_DIAMOND_CSGO_SPAWN_DIR`
    로 지정하세요. (`WorldModelEnv`가 여기서 진짜 프레임+액션을 읽어 컨텍스트를 채웁니다
    — Atari 어댑터처럼 정적 이미지로 흉내내지 않는 이유는, 실제 라이브 소스가 원 리포에
    이미 포함돼 있어 정확도를 낮출 이유가 없기 때문입니다.)

가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백합니다.

⚠ `WorldModelEnv`는 `horizon`(기본 1000 스텝)에 도달하면 spawn 데이터셋에서 새 컨텍스트를
뽑아 **내부적으로 자동 재시작**합니다 — 그 부분은 별도 처리가 필요 없습니다.

⚠ 실시간성 주의: base sampler와 upsampler 둘 다 매 프레임 forward pass가 필요해 Atari
보다 훨씬 무겁습니다. `config/world_model_env/fast.yaml`의 `diffusion_sampler_next_obs`/
`diffusion_sampler_upsampling` 의 `num_steps_denoising`을 줄이는 게 가장 큰 레버입니다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from ..common.actions import Action, csgo_vector
from ..common.base import WorldModel

log = logging.getLogger("diamond-csgo")

DIAMOND_CSGO_REPO = os.getenv("WM_DIAMOND_CSGO_REPO", "/opt/diamond-csgo")
CKPT = os.getenv("WM_DIAMOND_CSGO_CKPT", "/weights/diamond_csgo.pt")
SPAWN_DIR = os.getenv("WM_DIAMOND_CSGO_SPAWN_DIR", "")
DEVICE = os.getenv("WM_DEVICE", "cuda:0")


class DiamondCsgoWorldModel(WorldModel):
    fps = int(os.getenv("WM_FPS_CSGO", "10"))
    quality = int(os.getenv("WM_JPEG_QUALITY", "80"))

    def __init__(self) -> None:
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        self.torch = torch
        self.device = torch.device(DEVICE)

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        config_dir = os.path.join(DIAMOND_CSGO_REPO, "config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="trainer")

        assert cfg.env.train.id == "csgo", f"csgo 브랜치 config가 아닙니다: {cfg.env.train.id}"
        num_actions = cfg.env.num_actions   # 51 (config/env/csgo.yaml)

        from agent import Agent
        from envs import WorldModelEnv

        log.info("Agent(denoiser+upsampler) 로딩: %s", CKPT)
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self.device).eval()
        self.agent.load(Path(CKPT))

        sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
        if self.agent.upsampler is not None:
            sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)

        wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
        self.wm_env = WorldModelEnv(
            self.agent.denoiser, self.agent.upsampler, self.agent.rew_end_model,
            Path(SPAWN_DIR), 1, sl, wm_env_cfg, return_denoising_trajectory=False,
        )

        log.info("DIAMOND(CS:GO) 준비 완료. num_actions=%d seq_length=%d", num_actions, sl)

    # ----------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            obs, _ = self.wm_env.reset()
        frame = obs[0]
        return frame.float().cpu().numpy()

    def step(self, action: Action) -> np.ndarray:
        torch = self.torch
        act = torch.from_numpy(csgo_vector(action)).to(self.device).unsqueeze(0)   # (1, 51)
        with torch.no_grad():
            obs, _rew, _end, _trunc, _info = self.wm_env.step(act)
        frame = obs[0]
        return frame.float().cpu().numpy()          # CHW, [-1,1] (풀 해상도, 업샘플러 출력)

    def close(self) -> None:
        self.agent = None
        self.wm_env = None
        if getattr(self, "torch", None) is not None:
            self.torch.cuda.empty_cache()


def build(model_id: str) -> WorldModel:
    """워커 팩토리. 가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백."""
    if os.getenv("WM_DUMMY") == "1":
        from .dummy import make_dummy
        log.warning("WM_DUMMY=1 — 더미 모델로 기동합니다 (실제 추론 아님)")
        return make_dummy(model_id)

    missing = []
    if not os.path.exists(CKPT):
        missing.append(CKPT)
    if not SPAWN_DIR or not os.path.isdir(SPAWN_DIR):
        missing.append(SPAWN_DIR or "(WM_DIAMOND_CSGO_SPAWN_DIR 미설정)")
    if missing:
        raise FileNotFoundError(
            "체크포인트/spawn 데이터를 찾을 수 없습니다: " + ", ".join(missing) + "\n"
            "  huggingface-cli download eloialonso/diamond --include \"csgo/*\" "
            f"--local-dir {DIAMOND_CSGO_REPO}/downloads\n"
            "  csgo/model/csgo.pt → WM_DIAMOND_CSGO_CKPT\n"
            "  csgo/spawn/        → WM_DIAMOND_CSGO_SPAWN_DIR\n"
            "  파이프라인만 확인하려면 WM_DUMMY=1 로 실행하세요."
        )
    return DiamondCsgoWorldModel()
