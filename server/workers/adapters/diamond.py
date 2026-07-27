"""
DIAMOND 어댑터 (CS:GO / Atari).

DIAMOND는 EDM 기반 diffusion world model입니다. 공식 리포의
`src/models/diffusion/` 아래 denoiser와 sampler를 그대로 재사용하면 됩니다.

두 변종을 한 컨테이너에서 서빙합니다 (torch 의존성이 같으므로):
  * diamond-csgo  — 280×150, 연속 액션(키+마우스 bin), 무겁고 느림
  * diamond-atari — 64×64, 이산 액션 1개, 가벼워서 부하 테스트에 적합
"""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from ..common.actions import Action, csgo_vector, to_atari
from ..common.base import WorldModel

log = logging.getLogger("diamond")

CKPT_CSGO = os.getenv("WM_DIAMOND_CSGO_CKPT", "/weights/diamond_csgo.pt")
CKPT_ATARI = os.getenv("WM_DIAMOND_ATARI_CKPT", "/weights/diamond_atari.pt")
DEVICE = os.getenv("WM_DEVICE", "cuda")

# DIAMOND의 denoiser step 수. 논문 기본은 3이고, 늘릴수록 품질↑ 속도↓.
DENOISE_STEPS = int(os.getenv("WM_DENOISE_STEPS", "3"))
# 조건으로 넣는 과거 프레임 수 (num_steps_conditioning)
CONTEXT = int(os.getenv("WM_DIAMOND_CONTEXT", "4"))


class DiamondWorldModel(WorldModel):
    def __init__(self, variant: str) -> None:
        import torch

        self.torch = torch
        self.variant = variant
        self.device = torch.device(DEVICE)
        self.is_atari = variant.endswith("atari")

        self.fps = int(os.getenv("WM_FPS_ATARI", "30")) if self.is_atari \
            else int(os.getenv("WM_FPS_CSGO", "10"))
        self.quality = int(os.getenv("WM_JPEG_QUALITY", "80"))

        ckpt = CKPT_ATARI if self.is_atari else CKPT_CSGO

        # ------------------------------------------------------------------
        # TODO(1): denoiser 로드
        # ------------------------------------------------------------------
        # from models.diffusion import Denoiser, DiffusionSampler
        # sd = torch.load(ckpt, map_location="cpu")
        # self.denoiser = Denoiser(sd["config"]).to(self.device).eval()
        # self.denoiser.load_state_dict(sd["denoiser"])
        # self.sampler = DiffusionSampler(self.denoiser, num_steps=DENOISE_STEPS)
        raise NotImplementedError(
            f"DIAMOND 체크포인트를 연결하세요 (diamond.py TODO 1~3). ckpt={ckpt}\n"
            "먼저 파이프라인만 확인하려면 WM_DUMMY=1로 실행하세요."
        )

    def _encode_action(self, action: Action):
        """변종별 액션 인코딩."""
        torch = self.torch
        if self.is_atari:
            idx = to_atari(action)
            return torch.tensor([idx], device=self.device, dtype=torch.long)
        vec = csgo_vector(action)          # 길이 62
        return torch.from_numpy(vec).to(self.device).unsqueeze(0)

    def reset(self) -> np.ndarray:
        torch = self.torch

        # ------------------------------------------------------------------
        # TODO(2): 실제 데이터셋의 시작 프레임으로 버퍼 초기화
        # ------------------------------------------------------------------
        # DIAMOND는 num_steps_conditioning개의 실제 프레임이 있어야 시작합니다.
        # 공식 리포의 play.py처럼 데이터셋에서 한 궤적의 앞부분을 가져오세요.
        #
        # obs0 = load_start_frames(self.variant, CONTEXT)   # (CONTEXT, C, H, W)
        # self.obs_buf = deque(list(obs0), maxlen=CONTEXT)
        # self.act_buf = deque([self._encode_action(Action())] * CONTEXT,
        #                      maxlen=CONTEXT)
        # return obs0[-1].cpu().numpy()
        raise NotImplementedError

    def step(self, action: Action) -> np.ndarray:
        torch = self.torch
        act = self._encode_action(action)

        # ------------------------------------------------------------------
        # TODO(3): 다음 관측 샘플링
        # ------------------------------------------------------------------
        # self.act_buf.append(act)
        # with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        #     obs_ctx = torch.stack(list(self.obs_buf)).unsqueeze(0)
        #     act_ctx = torch.stack(list(self.act_buf), dim=1)
        #     next_obs = self.sampler.sample(obs=obs_ctx, act=act_ctx)[0]
        # self.obs_buf.append(next_obs)
        # return next_obs.float().cpu().numpy()   # CHW, [-1,1] — 런타임이 정규화
        raise NotImplementedError

    def close(self) -> None:
        if getattr(self, "torch", None) is not None:
            self.torch.cuda.empty_cache()


def build(model_id: str) -> WorldModel:
    if os.getenv("WM_DUMMY") == "1":
        from .dummy import make_dummy
        log.warning("WM_DUMMY=1 — 더미 모델로 기동합니다 (실제 추론 아님)")
        return make_dummy(model_id)

    ckpt = CKPT_ATARI if model_id.endswith("atari") else CKPT_CSGO
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt}\n"
            f"  - 가중치 경로를 WM_DIAMOND_*_CKPT로 지정하거나\n"
            f"  - 파이프라인 확인만 하려면 WM_DUMMY=1로 실행하세요."
        )
    return DiamondWorldModel(model_id)
