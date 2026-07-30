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
`reset()`을 원본처럼 바꿔도 됩니다(ale_py/gymnasium은 아래 이유로 어차피 설치돼 있습니다).

전제:
  * diamond 저장소(main 브랜치)의 `src/` 가 PYTHONPATH 에 있어야 합니다 — 저장소 코드가
    `from models.diffusion import ...` 처럼 `src/` 를 루트로 임포트합니다. `run_local.sh`가
    WM_DIAMOND_ATARI_REPO(저장소 루트)에 `/src`를 붙여 PYTHONPATH로 넘겨줍니다.
  * `gymnasium`/`ale-py`가 설치돼 있어야 합니다 — 우리가 직접 쓰진 않지만, `agent.py`가
    무조건 `from envs import ...`를 하고 `envs/env.py`가 최상단에서 그 둘을 import합니다
    (ale-py 0.8+ 는 ROM을 함께 배포하므로 AutoROM 등 별도 라이선스 동의 절차는 필요 없음).
  * `config/agent/default.yaml` + `config/env/atari.yaml` (저장소에 이미 들어있음, 그대로 사용).

체크포인트 26종(atari_100k 벤치마크 게임 전부)을 한 워커가 model_id로 골라 서빙합니다 —
`workers/common/actions.ATARI_GAMES` 참고. `model_id` 형식은 `diamond-atari-<게임소문자>`
(예: `diamond-atari-breakout`, `diamond-atari-mspacman`). 접미사 없는 `diamond-atari`는
`WM_DIAMOND_ATARI_DEFAULT_GAME`(기본 Breakout)로 대체됩니다.

가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백합니다.

⚠ 게임마다 학습된 액션 개수가 다릅니다(예: Breakout 4개, Alien 18개 — ALE의 축소
  액션셋). `workers/common/actions.ATARI_GAME_ACTIONS`에 게임별 정확한 목록이 있고,
  `num_actions`도 여기서 그대로 계산합니다 — 더 이상 수동으로 맞출 값이 없습니다.

⚠ 실시간성 주의: README의 Oasis 절과 동일 — `WM_DENOISE_STEPS`로 denoising step 수를
  줄이는 게 가장 큰 레버입니다 (기본 3, 논문 기본값과 동일).
"""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from ..common.actions import ATARI_GAME_ACTIONS, ATARI_GAMES, Action, to_atari
from ..common.base import WorldModel

log = logging.getLogger("diamond-atari")

DIAMOND_REPO = os.getenv("WM_DIAMOND_ATARI_REPO", "/opt/diamond")
WEIGHTS_DIR = os.getenv("WM_DIAMOND_ATARI_WEIGHTS_DIR", "/weights/diamond_atari")
DEVICE = os.getenv("WM_DEVICE", "cuda:0")
PROMPT_PATH = os.getenv("WM_DIAMOND_ATARI_PROMPT", "")
DEFAULT_GAME = os.getenv("WM_DIAMOND_ATARI_DEFAULT_GAME", "Breakout")

# DIAMOND denoiser step 수. 논문/저장소 기본은 3.
DENOISE_STEPS = int(os.getenv("WM_DENOISE_STEPS", "3"))

_MODEL_PREFIX = "diamond-atari-"
_SLUG_TO_GAME = {g.lower(): g for g in ATARI_GAMES}


def _game_from_model_id(model_id: str) -> str:
    """`diamond-atari-<슬러그>` → 정확한 대소문자의 게임 이름(체크포인트 파일명과 동일)."""
    if model_id.startswith(_MODEL_PREFIX):
        slug = model_id[len(_MODEL_PREFIX):]
        game = _SLUG_TO_GAME.get(slug)
        if game is None:
            raise ValueError(
                f"알 수 없는 Atari 게임: '{slug}'. 지원 목록: "
                + ", ".join(g.lower() for g in ATARI_GAMES)
            )
        return game
    return DEFAULT_GAME


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

    def __init__(self, game: str, ckpt: str) -> None:
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        self.game = game
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

        num_actions = len(ATARI_GAME_ACTIONS[game])
        agent_cfg = instantiate(cfg.agent, num_actions=num_actions)
        self.context = agent_cfg.denoiser.inner_model.num_steps_conditioning

        log.info("Denoiser 로딩: game=%s ckpt=%s img_size=%d context=%d num_actions=%d",
                 game, ckpt, self.img_size, self.context, num_actions)
        self.denoiser = Denoiser(agent_cfg.denoiser).to(self.device).eval()
        sd = torch.load(ckpt, map_location=self.device)
        self.denoiser.load_state_dict(extract_state_dict(sd, "denoiser"))

        sampler_cfg = instantiate(cfg.world_model_env.diffusion_sampler, num_steps_denoising=DENOISE_STEPS)
        self.sampler = DiffusionSampler(self.denoiser, sampler_cfg)

        log.info("DIAMOND(Atari) 준비 완료. game=%s denoise_steps=%d", game, DENOISE_STEPS)

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
        self.act_buf.append(to_atari(action, self.game))

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

    game = _game_from_model_id(model_id)
    ckpt = os.path.join(WEIGHTS_DIR, f"{game}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt}\n"
            "  hf download eloialonso/diamond --include \"atari_100k/models/*\" "
            f"--local-dir {WEIGHTS_DIR}\n"
            "  (내려받으면 <디렉터리>/atari_100k/models/*.pt 로 들어오니, 그 폴더를 통째로\n"
            f"   {WEIGHTS_DIR} 로 지정하거나 파일들을 옮기세요)\n"
            "  파이프라인만 확인하려면 WM_DUMMY=1 로 실행하세요."
        )
    return DiamondAtariWorldModel(game, ckpt)
