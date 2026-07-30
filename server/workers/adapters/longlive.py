"""
LongLive 어댑터 (NVLabs/LongLive, ``v1.0`` 브랜치 — LongLive-1.3B, 실시간 인터랙티브 롱비디오).

⚠ models/LongLive 로컬 clone은 원래 ``main`` 브랜치(=LongLive 2.0, 5B, 오프라인 배치
생성 파이프라인 — ``inference.py`` 로 전체 비디오를 한 번에 만들어 mp4 로 저장하는 구조라
프레임 단위 스트리밍 서버에 맞지 않습니다)였습니다. 이 프로젝트가 필요로 하는 것은 실시간
인터랙티브 스트리밍이 가능한 ``v1.0`` 브랜치(LongLive-1.3B, ``interactive_inference.py`` /
``pipeline/interactive_causal_inference.py``)입니다.

    git -C models/LongLive checkout v1.0

로 브랜치를 바꿔둔 상태를 전제로 합니다 (이미 이 리포에서 전환해뒀습니다).

LongLive는 오아시스/다이아몬드처럼 WASD로 캐릭터를 조작하는 게임형 월드모델이 아니라
"텍스트 프롬프트로 다음 장면을 계속 지시하는" 롱비디오 생성기입니다. 그래서 이 어댑터는
프론트엔드의 이동/마우스 액션이 아니라 hotbar 슬롯(1~9)을 "지금 재생 중인 프롬프트" 선택
스위치로 씁니다 — ``WM_LONGLIVE_PROMPTS`` 에 ``|`` 로 구분해 넣은 문장 중 하나를 hotbar
숫자로 고르면 다음 블록부터 그 프롬프트로 전환됩니다. 전환 시 cross-attention 캐시만
리셋하고 self-attention KV 캐시(시각적 연속성)는 유지합니다 — 원 코드의 ``global_sink``
정책과 동일합니다.

이 파일이 하는 일 — ``pipeline/interactive_causal_inference.py`` 의
``InteractiveCausalInferencePipeline.inference()`` 는 프롬프트 시퀀스 전체를 받아 고정
길이 비디오를 한 번에 만들고 마지막에 VAE로 통째로 디코딩하는 **오프라인** 함수입니다.
우리는 매 틱 프레임 하나씩 내보내야 하므로, 그 함수 내부의 "블록 하나 denoising" 루프와
"프롬프트 전환 시 재캐시(recache)" 로직만 그대로 떼어내 self 에 상태(KV 캐시, latent
히스토리, 프레임 큐)로 들고 다니면서 블록 단위로 나눠 돌립니다. VAE 디코딩도
``use_cache=True`` (causal VAE의 스트리밍 디코드)로 블록이 끝날 때마다 바로 수행해
프레임 큐에 쌓아둡니다. 원본 오프라인 스크립트는 재캐시할 때 latent 전체 히스토리를
고정 길이로 미리 할당해두지만, 우리는 세션이 몇 분이고 계속될 수 있으므로 최근
``local_attn_size`` 프레임만 담는 롤링 버퍼(``self.latent_hist``)를 대신 씁니다.

전제:
  * LongLive 저장소(v1.0 브랜치)가 PYTHONPATH 에 있어야 합니다.
  * 저장소 코드 자체가 가중치를 **상대경로**로 찾습니다 — ``WanTextEncoder`` /
    ``WanVAEWrapper`` / ``CausalWanModel.from_pretrained`` 내부에
    ``"wan_models/Wan2.1-T2V-1.3B/..."`` 가 하드코딩돼 있습니다. 그래서 이 어댑터는
    모델을 생성하기 전에 워커 프로세스의 cwd 를 저장소 루트로 바꿔둡니다
    (``WM_LONGLIVE_REPO``).
  * 가중치가 두 종류 따로 필요합니다 (LongLive 체크포인트 하나만으론 안 됩니다):
      1) Wan2.1-T2V-1.3B 베이스(T5 텍스트 인코더 + VAE + 토크나이저) — LongLive가 아니라
         원본 Wan2.1 저장소 것입니다:
           huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \\
             --local-dir <WM_LONGLIVE_REPO>/wan_models/Wan2.1-T2V-1.3B
      2) LongLive-1.3B 체크포인트(generator + LoRA):
           huggingface-cli download Efficient-Large-Model/LongLive \\
             --local-dir <WM_LONGLIVE_REPO>/longlive_models

가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백합니다.

⚠ 실시간성 주의: 블록(``num_frame_per_block`` latent 프레임, 기본 설정 3) 하나를 만들
때마다 ``denoising_step_list`` 단계 수만큼 forward + 캐시 갱신 1회를 더 돕니다. 논문
기준 H100 단일 GPU에서 약 20fps입니다. 그보다 느린 GPU라면
``configs/longlive_interactive_inference.yaml`` 의 ``denoising_step_list`` 길이나
``num_frame_per_block`` 을 줄이세요.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Dict, Optional

import numpy as np

from ..common.actions import Action, longlive_prompt_index
from ..common.base import WorldModel

log = logging.getLogger("longlive")

LONGLIVE_REPO = os.getenv("WM_LONGLIVE_REPO", "/opt/LongLive")
CONFIG_PATH = os.getenv("WM_LONGLIVE_CONFIG", "configs/longlive_interactive_inference.yaml")
CKPT_OVERRIDE = os.getenv("WM_LONGLIVE_CKPT", "")
LORA_CKPT_OVERRIDE = os.getenv("WM_LONGLIVE_LORA_CKPT", "")
DEVICE = os.getenv("WM_DEVICE", "cuda:0")

# Wan2.1-T2V-1.3B 잠재 공간 크기. configs/default_config.yaml 의 height=480/width=832 를
# VAE stride (4, 8, 8) 로 나눈 값 — 아키텍처가 고정이라 모델을 바꾸지 않는 한 상수입니다.
LATENT_C = 16
LATENT_H = int(os.getenv("WM_LONGLIVE_LATENT_H", "60"))
LATENT_W = int(os.getenv("WM_LONGLIVE_LATENT_W", "104"))

DEFAULT_PROMPT = (
    "A scenic drone shot flying over a lush green mountain valley at sunrise, "
    "cinematic, photorealistic."
)
PROMPTS_RAW = os.getenv("WM_LONGLIVE_PROMPTS", DEFAULT_PROMPT)


class LongLiveWorldModel(WorldModel):
    fps = int(os.getenv("WM_FPS_LONGLIVE", "16"))
    quality = int(os.getenv("WM_JPEG_QUALITY", "80"))

    def __init__(self) -> None:
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.device = torch.device(DEVICE)

        if not os.path.isdir(LONGLIVE_REPO):
            raise FileNotFoundError(f"LongLive 저장소를 찾을 수 없습니다: {LONGLIVE_REPO}")
        os.chdir(LONGLIVE_REPO)   # wan_models/... 상대경로 로드를 위해 cwd 고정

        from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline

        cfg_path = CONFIG_PATH if os.path.isabs(CONFIG_PATH) else os.path.join(LONGLIVE_REPO, CONFIG_PATH)
        cfg = OmegaConf.load(cfg_path)
        if CKPT_OVERRIDE:
            cfg.generator_ckpt = CKPT_OVERRIDE
        if LORA_CKPT_OVERRIDE:
            cfg.lora_ckpt = LORA_CKPT_OVERRIDE
        self.cfg = cfg

        log.info("LongLive 파이프라인 초기화: config=%s", cfg_path)
        pipe = InteractiveCausalInferencePipeline(cfg, device=self.device)

        log.info("generator 체크포인트 로딩: %s", cfg.generator_ckpt)
        state_dict = torch.load(cfg.generator_ckpt, map_location="cpu")
        use_ema = bool(cfg.get("use_ema", False))
        raw = state_dict["generator_ema" if use_ema else "generator"]
        if use_ema:
            raw = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw.items()}
            missing, unexpected = pipe.generator.load_state_dict(raw, strict=False)
            if missing or unexpected:
                log.warning("generator 로드: missing=%d unexpected=%d", len(missing), len(unexpected))
        else:
            pipe.generator.load_state_dict(raw)

        if cfg.get("adapter", None):
            from utils.lora_utils import configure_lora_for_model
            log.info("LoRA 적용: %s", cfg.adapter)
            pipe.generator.model = configure_lora_for_model(
                pipe.generator.model, model_name="generator",
                lora_config=cfg.adapter, is_main_process=True,
            )
            lora_ckpt = cfg.get("lora_ckpt", None)
            if lora_ckpt:
                import peft
                log.info("LoRA 체크포인트 로딩: %s", lora_ckpt)
                lora_sd = torch.load(lora_ckpt, map_location="cpu")
                if isinstance(lora_sd, dict) and "generator_lora" in lora_sd:
                    lora_sd = lora_sd["generator_lora"]
                peft.set_peft_model_state_dict(pipe.generator.model, lora_sd)

        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.generator.to(device=self.device)
        pipe.vae.to(device=self.device)
        pipe.generator.model.eval().requires_grad_(False)
        self.pipe = pipe

        self.prompts = [p.strip() for p in PROMPTS_RAW.split("|") if p.strip()] or [DEFAULT_PROMPT]
        self._cond_cache: Dict[int, dict] = {}

        log.info("LongLive 준비 완료. prompts=%d block=%d local_attn=%d",
                 len(self.prompts), pipe.num_frame_per_block, pipe.local_attn_size)

    # ----------------------------------------------------------------------

    def _encode_prompt(self, idx: int) -> dict:
        if idx not in self._cond_cache:
            with self.torch.no_grad():
                self._cond_cache[idx] = self.pipe.text_encoder(text_prompts=[self.prompts[idx]])
        return self._cond_cache[idx]

    def _select_prompt_idx(self, action: Action) -> int:
        return longlive_prompt_index(action, len(self.prompts))

    def reset(self) -> np.ndarray:
        torch = self.torch
        pipe = self.pipe

        kv_size = (pipe.local_attn_size * pipe.frame_seq_length) if pipe.local_attn_size != -1 else 32760
        pipe._initialize_kv_cache(1, dtype=torch.bfloat16, device=self.device, kv_cache_size_override=kv_size)
        pipe._initialize_crossattn_cache(1, dtype=torch.bfloat16, device=self.device)
        pipe.generator.model.local_attn_size = pipe.local_attn_size
        pipe._set_all_modules_max_attention_size(pipe.local_attn_size)
        pipe.vae.model.clear_cache()   # 스트리밍(cached_decode) 상태 초기화

        self.current_start_frame = 0
        self.prompt_idx = 0
        self.latent_hist: Optional["torch.Tensor"] = None
        self.frame_queue: "deque" = deque()

        self._run_block(self._encode_prompt(self.prompt_idx))
        return self.frame_queue.popleft()

    def step(self, action: Action) -> np.ndarray:
        new_idx = self._select_prompt_idx(action)
        if new_idx != self.prompt_idx:
            self.prompt_idx = new_idx
            self._recache(self._encode_prompt(self.prompt_idx))

        if not self.frame_queue:
            self._run_block(self._encode_prompt(self.prompt_idx))
        return self.frame_queue.popleft()

    # ----------------------------------------------------------------------

    def _run_block(self, cond: dict) -> None:
        """블록 하나(num_frame_per_block latent 프레임)를 생성해 프레임 큐에 채운다.
        CausalInferencePipeline.inference() 의 "Step 2.1 spatial denoising loop" +
        "Step 2.3 클린 컨텍스트로 캐시 갱신" 부분과 동일한 로직."""
        torch = self.torch
        pipe = self.pipe
        n = pipe.num_frame_per_block

        noise = torch.randn(1, n, LATENT_C, LATENT_H, LATENT_W, device=self.device, dtype=torch.bfloat16)
        noisy_input = noise
        denoised_pred = None
        timestep = None
        for i, t in enumerate(pipe.denoising_step_list):
            timestep = torch.ones(1, n, device=self.device, dtype=torch.int64) * t
            with torch.no_grad():
                _, denoised_pred = pipe.generator(
                    noisy_image_or_video=noisy_input, conditional_dict=cond, timestep=timestep,
                    kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                    current_start=self.current_start_frame * pipe.frame_seq_length,
                )
            if i < len(pipe.denoising_step_list) - 1:
                next_t = pipe.denoising_step_list[i + 1]
                noisy_input = pipe.scheduler.add_noise(
                    denoised_pred.flatten(0, 1), torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_t * torch.ones(n, device=self.device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])

        context_timestep = torch.ones_like(timestep) * pipe.args.context_noise
        with torch.no_grad():
            pipe.generator(
                noisy_image_or_video=denoised_pred, conditional_dict=cond, timestep=context_timestep,
                kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                current_start=self.current_start_frame * pipe.frame_seq_length,
            )

        self.current_start_frame += n
        self._push_history(denoised_pred)

        with torch.no_grad():
            px = pipe.vae.decode_to_pixel(denoised_pred, use_cache=True)   # (1, T, C, H, W)
        px = (px * 0.5 + 0.5).clamp(0, 1)[0]
        for f in px:
            self.frame_queue.append(f.float().cpu().numpy())              # CHW, [0,1]

    def _push_history(self, denoised_pred) -> None:
        """_recache 에 쓸, 최근 local_attn_size 프레임만 담는 롤링 latent 버퍼."""
        pipe = self.pipe
        self.latent_hist = denoised_pred if self.latent_hist is None \
            else self.torch.cat([self.latent_hist, denoised_pred], dim=1)
        if pipe.local_attn_size != -1 and self.latent_hist.shape[1] > pipe.local_attn_size:
            self.latent_hist = self.latent_hist[:, -pipe.local_attn_size:]

    def _recache(self, new_cond: dict) -> None:
        """프롬프트 전환 — InteractiveCausalInferencePipeline._recache_after_switch 를
        무제한 길이 세션에 맞게 이식. 원본은 오프라인 스크립트가 미리 할당해둔 전체
        latent 버퍼(``output``)에서 슬라이스하지만, 우리는 끝없이 이어지는 세션이라
        최근 프레임만 담은 롤링 버퍼(self.latent_hist)를 대신 쓴다."""
        torch = self.torch
        pipe = self.pipe

        if not pipe.global_sink:
            for cache in pipe.kv_cache1:
                cache["k"].zero_()
                cache["v"].zero_()
        for blk in pipe.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        if self.current_start_frame == 0 or self.latent_hist is None:
            return

        num_recache = self.latent_hist.shape[1]
        if pipe.local_attn_size != -1:
            num_recache = min(num_recache, pipe.local_attn_size)
        recache_start = self.current_start_frame - num_recache
        frames = self.latent_hist[:, -num_recache:]

        block_mask = pipe.generator.model._prepare_blockwise_causal_attn_mask(
            device=frames.device, num_frames=num_recache, frame_seqlen=pipe.frame_seq_length,
            num_frame_per_block=pipe.num_frame_per_block, local_attn_size=pipe.local_attn_size,
        )
        pipe.generator.model.block_mask = block_mask
        context_timestep = torch.ones([1, num_recache], device=frames.device, dtype=torch.int64) * pipe.args.context_noise
        with torch.no_grad():
            pipe.generator(
                noisy_image_or_video=frames, conditional_dict=new_cond, timestep=context_timestep,
                kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache,
                current_start=recache_start * pipe.frame_seq_length,
                sink_recache_after_switch=not pipe.global_sink,
            )
        for blk in pipe.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def close(self) -> None:
        self.pipe = None
        self.latent_hist = None
        self.frame_queue = None
        self._cond_cache = None
        if getattr(self, "torch", None) is not None:
            self.torch.cuda.empty_cache()


def build(model_id: str) -> WorldModel:
    """워커 팩토리. 가중치가 없거나 WM_DUMMY=1 이면 더미로 폴백."""
    if os.getenv("WM_DUMMY") == "1":
        from .dummy import make_dummy
        log.warning("WM_DUMMY=1 — 더미 모델로 기동합니다 (실제 추론 아님)")
        return make_dummy(model_id)

    wan_base = os.path.join(LONGLIVE_REPO, "wan_models", "Wan2.1-T2V-1.3B")
    ckpt = CKPT_OVERRIDE or os.path.join(LONGLIVE_REPO, "longlive_models", "models", "longlive_base.pt")
    missing = [p for p in (wan_base, ckpt) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "가중치를 찾을 수 없습니다: " + ", ".join(missing) + "\n"
            "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "
            f"{LONGLIVE_REPO}/wan_models/Wan2.1-T2V-1.3B\n"
            "  huggingface-cli download Efficient-Large-Model/LongLive --local-dir "
            f"{LONGLIVE_REPO}/longlive_models\n"
            "  파이프라인만 확인하려면 WM_DUMMY=1 로 실행하세요."
        )
    return LongLiveWorldModel()
