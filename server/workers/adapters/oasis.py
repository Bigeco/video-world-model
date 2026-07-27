"""
Oasis 어댑터 (Minecraft world model).

체크포인트를 연결하는 지점은 아래 TODO 세 군데뿐입니다.
가중치가 없거나 WM_DUMMY=1이면 자동으로 더미로 떨어집니다.

Oasis 계열 모델의 공통 구조:
  * 최근 N개 프레임을 VAE로 잠재(latent)로 인코딩해 컨텍스트로 유지
  * 액션 시퀀스를 조건으로 다음 프레임 잠재를 diffusion으로 샘플링
  * 디코딩해서 RGB 프레임 반환
  * 컨텍스트 윈도우를 슬라이딩하며 자기회귀 반복

실시간성의 핵심은 **diffusion step 수**입니다. 학습 시 50 step이라도 추론은
4~8 step으로 줄여야 20FPS가 나옵니다. WM_DIFFUSION_STEPS로 조절하세요.
"""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from ..common.actions import Action, minecraft_vector
from ..common.base import WorldModel

log = logging.getLogger("oasis")

CKPT = os.getenv("WM_OASIS_CKPT", "/weights/oasis500m.pt")
VAE_CKPT = os.getenv("WM_OASIS_VAE", "/weights/vit-l-20.pt")
CONTEXT = int(os.getenv("WM_CONTEXT", "16"))
STEPS = int(os.getenv("WM_DIFFUSION_STEPS", "8"))
DEVICE = os.getenv("WM_DEVICE", "cuda")


class OasisWorldModel(WorldModel):
    fps = int(os.getenv("WM_FPS_OASIS", "20"))
    quality = int(os.getenv("WM_JPEG_QUALITY", "80"))

    def __init__(self) -> None:
        import torch  # 지연 임포트 — 더미 경로에서는 torch가 필요 없다

        self.torch = torch
        self.device = torch.device(DEVICE)

        # ------------------------------------------------------------------
        # TODO(1): 모델과 VAE 로드
        # ------------------------------------------------------------------
        # from oasis.models import DiT, VAE          # 실제 리포의 임포트로 교체
        # self.model = DiT.from_pretrained(CKPT).to(self.device).eval()
        # self.vae   = VAE.from_pretrained(VAE_CKPT).to(self.device).eval()
        # self.model = self.model.to(memory_format=torch.channels_last).half()
        raise NotImplementedError(
            "Oasis 체크포인트를 연결하세요 (oasis.py TODO 1~3). "
            "먼저 파이프라인만 확인하려면 WM_DUMMY=1로 실행하세요."
        )

    # ----------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        torch = self.torch

        # ------------------------------------------------------------------
        # TODO(2): 시작 프레임으로 컨텍스트 초기화
        # ------------------------------------------------------------------
        # 대부분의 구현은 실제 게임 스크린샷 한 장을 seed로 씁니다.
        # 순수 노이즈로 시작하면 몇 초간 형태가 잡히지 않습니다.
        #
        # seed = load_seed_image()                       # HWC uint8
        # with torch.no_grad():
        #     latent = self.vae.encode(self._to_tensor(seed)).latent_dist.mode()
        # self.latents  = deque([latent] * CONTEXT, maxlen=CONTEXT)
        # self.actions  = deque([torch.zeros(1, 13, device=self.device)] * CONTEXT,
        #                       maxlen=CONTEXT)
        # return seed
        raise NotImplementedError

    def step(self, action: Action) -> np.ndarray:
        torch = self.torch

        # 브라우저 입력 → 13차원 액션 벡터 (버튼 11 + 카메라 2)
        vec = minecraft_vector(action)
        act = torch.from_numpy(vec).to(self.device).unsqueeze(0)

        # ------------------------------------------------------------------
        # TODO(3): 다음 프레임 잠재를 샘플링하고 디코딩
        # ------------------------------------------------------------------
        # self.actions.append(act)
        # with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        #     ctx_latents = torch.stack(list(self.latents), dim=1)
        #     ctx_actions = torch.stack(list(self.actions), dim=1)
        #     next_latent = self.model.sample(
        #         context=ctx_latents, actions=ctx_actions, num_steps=STEPS,
        #     )
        #     frame = self.vae.decode(next_latent).sample
        # self.latents.append(next_latent)
        # return frame[0].float().cpu().numpy()        # CHW float, 런타임이 정규화
        raise NotImplementedError

    def close(self) -> None:
        if getattr(self, "torch", None) is not None:
            self.torch.cuda.empty_cache()


def build(model_id: str) -> WorldModel:
    """워커 팩토리. 가중치가 없으면 더미로 자동 폴백."""
    if os.getenv("WM_DUMMY") == "1":
        from .dummy import make_dummy
        log.warning("WM_DUMMY=1 — 더미 모델로 기동합니다 (실제 추론 아님)")
        return make_dummy(model_id)

    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {CKPT}\n"
            f"  - 가중치를 받아 WM_OASIS_CKPT에 경로를 지정하거나\n"
            f"  - 파이프라인 확인만 하려면 WM_DUMMY=1로 실행하세요."
        )
    return OasisWorldModel()
