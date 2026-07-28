"""
Oasis 어댑터 — open-oasis (etched-ai) 실제 추론 코드에 맞춰 구현.

이 파일은 저장소의 generate.py 를 **상호작용용 상태 루프**로 바꾼 것입니다.
generate.py 는 액션 시퀀스 전체를 미리 받아 32프레임을 한 번에 만들어 video.mp4 로
저장하는 오프라인 스크립트입니다. 우리는 매 틱 액션 하나를 받아 다음 프레임 하나만
생성해야 하므로, 잠재(latent) 컨텍스트와 액션 히스토리를 self 에 들고 다니면서
generate.py 의 diffusion-forcing 샘플링 루프를 프레임 단위로 돌립니다.

전제:
  * open-oasis 저장소가 PYTHONPATH 에 있어야 합니다 (Dockerfile 주석 참고).
    거기서 dit, vae, utils 를 임포트합니다.
  * 체크포인트 2개: oasis500m.safetensors (DiT), vit-l-20.safetensors (VAE).
  * 시작 프레임(prompt) 이미지 1장. 저장소의 sample_data/sample_image_0.png 로 시작해도 됩니다.

가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백합니다.

⚠ 실시간성 주의: generate.py 는 프레임당 ddim_steps(기본 10) 번, 매번 컨텍스트
윈도우 전체에 대해 DiT forward 를 돕니다. 단일 GPU에서 360×640 · 20fps 는 그대로는
어렵습니다. WM_DIFFUSION_STEPS 를 4~8 로 줄이고, 필요하면 해상도·컨텍스트를 낮추세요.
자세한 튜닝은 server/README 의 Oasis 절 참고.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..common.actions import Action, oasis_vector
from ..common.base import WorldModel

log = logging.getLogger("oasis")

OASIS_CKPT = os.getenv("WM_OASIS_CKPT", "/weights/oasis500m.safetensors")
VAE_CKPT = os.getenv("WM_OASIS_VAE", "/weights/vit-l-20.safetensors")
PROMPT_PATH = os.getenv("WM_OASIS_PROMPT", "/weights/prompt.png")
DEVICE = os.getenv("WM_DEVICE", "cuda:0")

# 학습 시 총 노이즈 레벨. generate.py 상수와 동일하게 유지.
MAX_NOISE_LEVEL = 1000
NOISE_ABS_MAX = 20.0
STABILIZATION_LEVEL = 15

# 실시간을 위한 조절 손잡이
DDIM_STEPS = int(os.getenv("WM_DIFFUSION_STEPS", "8"))   # generate.py 기본 10
SCALING_FACTOR = 0.07843137255                            # generate.py 와 동일


class OasisWorldModel(WorldModel):
    fps = int(os.getenv("WM_FPS_OASIS", "20"))
    quality = int(os.getenv("WM_JPEG_QUALITY", "80"))

    def __init__(self) -> None:
        import torch
        from einops import rearrange
        from safetensors.torch import load_model

        # open-oasis 저장소 모듈
        from dit import DiT_models
        from vae import VAE_models
        from utils import load_prompt, sigmoid_beta_schedule

        self.torch = torch
        self.rearrange = rearrange
        self._load_prompt = load_prompt
        self.device = torch.device(DEVICE)

        # --- DiT 로드 (generate.py 그대로) ---
        log.info("DiT 로딩: %s", OASIS_CKPT)
        model = DiT_models["DiT-S/2"]()
        if OASIS_CKPT.endswith(".pt"):
            model.load_state_dict(torch.load(OASIS_CKPT, weights_only=True), strict=False)
        else:
            load_model(model, OASIS_CKPT)
        self.model = model.to(self.device).eval()

        # --- VAE 로드 ---
        log.info("VAE 로딩: %s", VAE_CKPT)
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        if VAE_CKPT.endswith(".pt"):
            vae.load_state_dict(torch.load(VAE_CKPT, weights_only=True))
        else:
            load_model(vae, VAE_CKPT)
        self.vae = vae.to(self.device).eval()

        # --- diffusion 스케줄 (generate.py 그대로) ---
        self.noise_range = torch.linspace(-1, MAX_NOISE_LEVEL - 1, DDIM_STEPS + 1)
        betas = sigmoid_beta_schedule(MAX_NOISE_LEVEL).float().to(self.device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = rearrange(acp, "T -> T 1 1 1")

        self.max_frames = getattr(self.model, "max_frames", 16)
        self._prompt_latent = None      # 시작 프레임 잠재 (reset 때 캐시)
        log.info("Oasis 준비 완료. ddim_steps=%d max_frames=%d", DDIM_STEPS, self.max_frames)

    # ----------------------------------------------------------------------

    def _encode_prompt(self):
        """시작 프레임 1장을 잠재로 인코딩. reset 마다 재사용하도록 캐시."""
        torch = self.torch
        rearrange = self.rearrange
        if self._prompt_latent is not None:
            return self._prompt_latent.clone()

        # (1, 1, C, 360, 640), [0,1]
        x = self._load_prompt(PROMPT_PATH, n_prompt_frames=1).to(self.device)
        self._H, self._W = x.shape[-2:]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.half):
            x = self.vae.encode(x * 2 - 1).mean * SCALING_FACTOR
        x = rearrange(x, "(b t) (h w) c -> b t c h w", t=1,
                      h=self._H // self.vae.patch_size, w=self._W // self.vae.patch_size)
        self._prompt_latent = x
        return x.clone()

    def reset(self) -> np.ndarray:
        torch = self.torch
        # 잠재 컨텍스트를 시작 프레임 하나로 초기화
        self.x = self._encode_prompt()                              # (1, 1, c, h, w)
        # 첫 프레임에 대응하는 액션(전부 0)
        self.actions = torch.zeros(
            (1, 1, len(oasis_vector(Action()))), device=self.device
        )
        self.frame_idx = 0
        return self._decode_last()

    def step(self, action: Action) -> np.ndarray:
        torch = self.torch
        rearrange = self.rearrange
        acp = self.alphas_cumprod

        # 1) 액션 히스토리에 이번 입력 추가 (25차원)
        vec = torch.from_numpy(oasis_vector(action)).to(self.device)
        self.actions = torch.cat([self.actions, vec.view(1, 1, -1)], dim=1)

        # 2) 새 프레임 자리에 노이즈를 붙인다 (generate.py 와 동일)
        i = self.x.shape[1]                          # 새로 만들 프레임의 인덱스
        chunk = torch.randn((1, 1, *self.x.shape[-3:]), device=self.device)
        chunk = torch.clamp(chunk, -NOISE_ABS_MAX, NOISE_ABS_MAX)
        self.x = torch.cat([self.x, chunk], dim=1)
        start = max(0, i + 1 - self.max_frames)      # 슬라이딩 윈도우

        # 3) DDIM 역방향 샘플링 (generate.py 루프를 프레임 1개에 대해 수행)
        for nidx in reversed(range(1, DDIM_STEPS + 1)):
            t_ctx = torch.full((1, i), STABILIZATION_LEVEL - 1, dtype=torch.long, device=self.device)
            t = torch.full((1, 1), self.noise_range[nidx], dtype=torch.long, device=self.device)
            t_next = torch.full((1, 1), self.noise_range[nidx - 1], dtype=torch.long, device=self.device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = self.x[:, start:].clone()
            t = t[:, start:]
            t_next = t_next[:, start:]

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.half):
                v = self.model(x_curr, t, self.actions[:, start : i + 1])

            x_start = acp[t].sqrt() * x_curr - (1 - acp[t]).sqrt() * v
            x_noise = ((1 / acp[t]).sqrt() * x_curr - x_start) / (1 / acp[t] - 1).sqrt()
            alpha_next = acp[t_next].clone()
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if nidx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            self.x[:, -1:] = x_pred[:, -1:]

        # 4) 컨텍스트가 너무 길어지면 앞을 버려 계산량을 묶어둔다
        if self.x.shape[1] > self.max_frames:
            drop = self.x.shape[1] - self.max_frames
            self.x = self.x[:, drop:]
            self.actions = self.actions[:, drop:]

        self.frame_idx += 1
        return self._decode_last()

    def _decode_last(self) -> np.ndarray:
        """마지막 잠재 프레임 1장만 디코딩해서 RGB 로 반환."""
        torch = self.torch
        rearrange = self.rearrange
        last = self.x[:, -1:]                                    # (1, 1, c, h, w)
        z = rearrange(last, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            img = (self.vae.decode(z / SCALING_FACTOR) + 1) / 2  # (1, c, H, W), [0,1]
        img = torch.clamp(img, 0, 1)[0]
        return img.float().cpu().numpy()                         # CHW, 런타임이 정규화

    def close(self) -> None:
        if getattr(self, "torch", None) is not None:
            self.torch.cuda.empty_cache()


def build(model_id: str) -> WorldModel:
    """워커 팩토리. 가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백."""
    if os.getenv("WM_DUMMY") == "1":
        from .dummy import make_dummy
        log.warning("WM_DUMMY=1 — 더미 모델로 기동합니다 (실제 추론 아님)")
        return make_dummy(model_id)

    missing = [p for p in (OASIS_CKPT, VAE_CKPT) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "체크포인트를 찾을 수 없습니다: " + ", ".join(missing) + "\n"
            "  huggingface-cli download Etched/oasis-500m oasis500m.safetensors\n"
            "  huggingface-cli download Etched/oasis-500m vit-l-20.safetensors\n"
            "  로 받아 WM_OASIS_CKPT / WM_OASIS_VAE 에 경로를 지정하세요.\n"
            "  파이프라인만 확인하려면 WM_DUMMY=1 로 실행하세요."
        )
    return OasisWorldModel()