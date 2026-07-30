"""
DIAMOND(Atari) 어댑터 — eloialonso/diamond `main` 브랜치 실제 추론 코드에 맞춰 구현.

⚠ DIAMOND는 GitHub 저장소가 브랜치별로 완전히 다른 모델입니다:
    * `main`  브랜치 — Atari. 단일 denoiser, 이산 액션(임베딩), 로컬에 이미 clone돼 있음.
    * `csgo`  브랜치 — CS:GO. base+upsampler 2단계 denoiser, 51차원 연속 액션, 실제 녹화
      프레임(spawn dataset)으로 워밍업. `main` 브랜치에는 이 코드가 아예 없습니다.
  두 브랜치의 `models/diffusion` 모듈이 이름은 같지만 내부 구조가 달라서 한 파이썬
  프로세스에 동시에 얹을 수 없습니다 — 그래서 CS:GO는 `diamond_csgo.py`로 완전히 분리돼
  있고, 그쪽은 별도 저장소 체크아웃(csgo 브랜치)을 요구합니다.

이 파일은 저장소의 `src/play.py`(`prepare_play_mode`)를 **상호작용용 상태 루프**로 바꾼
것입니다. 원본은 실제 Atari(gym/ALE) 환경에서 몇 스텝을 실제로 플레이해 그 프레임들로
world model의 초기 컨텍스트(`num_steps_conditioning`장)를 채웁니다. 우리는 게이트웨이가
ALE 의존성을 강제하지 않도록, 정적 이미지 한 장(WM_DIAMOND_ATARI_PROMPT, 없으면 회색
화면)을 컨텍스트 프레임 수만큼 복제해 시작합니다 — Oasis 어댑터가 시작 프레임 한 장으로
잠재 컨텍스트를 초기화하는 것과 같은 절충입니다. 실제 게임 프레임으로 시작하고 싶다면
`ale_py`/`gymnasium`을 추가로 설치하고 `reset()`을 원본처럼 바꿔도 됩니다.

전제:
  * diamond 저장소(main 브랜치)의 `src/` 가 PYTHONPATH 에 있어야 합니다 — 저장소 코드가
    `from models.diffusion import ...` 처럼 `src/` 를 루트로 임포트합니다. `run_local.sh`가
    WM_DIAMOND_ATARI_REPO(저장소 루트)에 `/src`를 붙여 PYTHONPATH로 넘겨줍니다.
  * 체크포인트 1개(denoiser만 사용 — rew_end_model/actor_critic은 로드하지 않습니다.
    우리는 보상/종료를 쓰지 않는 순수 화면 생성기이기 때문입니다).
  * `config/agent/default.yaml` + `config/env/atari.yaml` (저장소에 이미 들어있음, 그대로 사용).

가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백합니다.

⚠ 체크포인트가 학습된 게임의 액션 개수(ALE `env.action_space.n`)와
  `WM_DIAMOND_NUM_ACTIONS`가 반드시 일치해야 합니다. 다르면 액션 임베딩이 엉뚱한 값을
  참조해 화면이 무너집니다 (`workers/common/actions.py`의 `to_atari()`는 ALE 전체
  18액션 표를 기준으로 합니다 — 게임별 축소 액션셋으로 학습된 체크포인트라면 이 표와
  체크포인트가 실제로 대응하는지 확인하세요).

⚠ 실시간성 주의: README의 Oasis 절과 동일 — `WM_DENOISE_STEPS`로 denoising step 수를
  줄이는 게 가장 큰 레버입니다 (기본 3, 논문 기본값과 동일).
"""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from ..common.actions import Action, to_atari
from ..common.base import WorldModel

log = logging.getLogger("diamond-atari")

DIAMOND_REPO = os.getenv("WM_DIAMOND_ATARI_REPO", "/opt/diamond")
CKPT = os.getenv("WM_DIAMOND_ATARI_CKPT", "/weights/diamond_atari.pt")
DEVICE = os.getenv("WM_DEVICE", "cuda:0")
PROMPT_PATH = os.getenv("WM_DIAMOND_ATARI_PROMPT", "")

# DIAMOND denoiser step 수. 논문/저장소 기본은 3.
DENOISE_STEPS = int(os.getenv("WM_DENOISE_STEPS", "3"))
# 체크포인트가 학습된 게임의 액션 개수. ALE 표준 최대 18.
NUM_ACTIONS = int(os.getenv("WM_DIAMOND_NUM_ACTIONS", "18"))


def _load_prompt_image(path: str, size: int) -> np.ndarray:
    """시작 프레임 1장 → CHW float32 [-1, 1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0      # HWC [0,1]
    arr = arr.transpose(2, 0, 1) * 2.0 - 1.0              # CHW [-1,1]
    return arr


class DiamondAtariWorldModel(WorldModel):
    fps = int(os.getenv("WM_FPS_ATARI", "30"))
    quality = int(os.getenv("WM_JPEG_QUALITY", "80"))

    def __init__(self) -> None:
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        self.torch = torch
        self.device = torch.device(DEVICE)

        # world_model_env.diffusion_sampler.s_tmax = ${eval:'float("inf")'} 를 풀려면
        # 원본 리포의 main.py/play.py가 등록하는 커스텀 resolver가 필요하다.
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        config_dir = os.path.join(DIAMOND_REPO, "config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="trainer")

        self.img_size = int(cfg.env.train.size)

        from agent import AgentConfig  # noqa: F401  (인스턴스화가 실제로 이 타입을 만든다)
        from models.diffusion import Denoiser, DiffusionSampler
        from utils import extract_state_dict

        agent_cfg = instantiate(cfg.agent, num_actions=NUM_ACTIONS)
        self.context = agent_cfg.denoiser.inner_model.num_steps_conditioning

        log.info("Denoiser 로딩: %s (img_size=%d context=%d num_actions=%d)",
                 CKPT, self.img_size, self.context, NUM_ACTIONS)
        self.denoiser = Denoiser(agent_cfg.denoiser).to(self.device).eval()
        sd = torch.load(CKPT, map_location=self.device)
        self.denoiser.load_state_dict(extract_state_dict(sd, "denoiser"))

        sampler_cfg = instantiate(cfg.world_model_env.diffusion_sampler, num_steps_denoising=DENOISE_STEPS)
        self.sampler = DiffusionSampler(self.denoiser, sampler_cfg)

        log.info("DIAMOND(Atari) 준비 완료. denoise_steps=%d", DENOISE_STEPS)

    # ----------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        torch = self.torch
        if PROMPT_PATH and os.path.exists(PROMPT_PATH):
            frame = _load_prompt_image(PROMPT_PATH, self.img_size)
        else:
            frame = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)   # 회색(0) 화면
        obs0 = torch.from_numpy(frame).to(self.device)

        self.obs_buf: deque = deque([obs0.clone() for _ in range(self.context)], maxlen=self.context)
        self.act_buf: deque = deque([0] * self.context, maxlen=self.context)
        return obs0.float().cpu().numpy()

    def step(self, action: Action) -> np.ndarray:
        torch = self.torch
        self.act_buf.append(to_atari(action))

        obs_ctx = torch.stack(list(self.obs_buf)).unsqueeze(0)                                  # (1,T,C,H,W)
        act_ctx = torch.tensor([list(self.act_buf)], dtype=torch.long, device=self.device)       # (1,T)
        next_obs, _ = self.sampler.sample(obs_ctx, act_ctx)
        next_obs = next_obs[0]                                                                    # (C,H,W)

        self.obs_buf.append(next_obs)
        return next_obs.float().cpu().numpy()                                                      # CHW, [-1,1]

    def close(self) -> None:
        self.denoiser = None
        self.sampler = None
        self.obs_buf = None
        self.act_buf = None
        if getattr(self, "torch", None) is not None:
            self.torch.cuda.empty_cache()


def build(model_id: str) -> WorldModel:
    """워커 팩토리. 가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백."""
    if os.getenv("WM_DUMMY") == "1":
        from .dummy import make_dummy
        log.warning("WM_DUMMY=1 — 더미 모델로 기동합니다 (실제 추론 아님)")
        return make_dummy(model_id)

    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {CKPT}\n"
            "  huggingface-cli download eloialonso/diamond atari_100k/models/<게임>.pt\n"
            "  로 받아 WM_DIAMOND_ATARI_CKPT 에 경로를 지정하세요.\n"
            "  파이프라인만 확인하려면 WM_DUMMY=1 로 실행하세요."
        )
    return DiamondAtariWorldModel()
