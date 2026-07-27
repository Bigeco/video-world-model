"""프레임 인코딩 + 와이어 패킹."""

from __future__ import annotations

import struct
from typing import Callable

import numpy as np

# JPEG 인코더: cv2가 가장 빠르지만 없으면 PIL로 떨어진다.
_encoder: Callable[[np.ndarray, int], bytes]

try:
    import cv2  # type: ignore

    def _encode(rgb: np.ndarray, quality: int) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode 실패")
        return buf.tobytes()

    _encoder = _encode
    BACKEND = "cv2"

except ImportError:  # pragma: no cover - 환경에 따라 갈림
    import io

    from PIL import Image  # type: ignore

    def _encode(rgb: np.ndarray, quality: int) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    _encoder = _encode
    BACKEND = "pillow"


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """모델 출력을 HxWx3 uint8 RGB로 정규화한다.

    받아들이는 형태:
      - float 배열, 값 범위 [0,1] 또는 [-1,1]
      - uint8 배열
      - CHW (채널 우선) 또는 HWC
      - 그레이스케일 HxW → 3채널 복제
    """
    arr = np.asarray(frame)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))          # CHW → HWC
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo = float(arr.min()) if arr.size else 0.0
        if lo < -0.01:                               # [-1,1] 범위로 판단
            arr = (arr + 1.0) * 127.5
        elif arr.max() <= 1.001:                     # [0,1] 범위로 판단
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr[:, :, :3])


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    return _encoder(to_uint8_rgb(frame), quality)


def pack_frame(frame_id: int, ack_seq: int, jpeg: bytes) -> bytes:
    """
    와이어 포맷 — 프론트엔드 handleFrame()과 반드시 일치해야 한다.

        바이트 0..3   uint32 big-endian   frame_id
        바이트 4..7   uint32 big-endian   ack_seq
        바이트 8..    JPEG 원본

    ack_seq는 이 프레임 생성에 실제로 반영된 마지막 action의 seq다.
    이 값이 있어야 브라우저가 입력→화면 왕복 지연을 계산할 수 있다.
    """
    return struct.pack(">II", frame_id & 0xFFFFFFFF, ack_seq & 0xFFFFFFFF) + jpeg
