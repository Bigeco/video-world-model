"""
워커 진입점.

    WM_MODEL=oasis   python -m workers.run
    WM_MODEL=diamond python -m workers.run
    WM_MODEL=oasis WM_DUMMY=1 python -m workers.run     # 가중치 없이

WM_MODEL은 *어댑터*를 고르고, 실제 변종(diamond-csgo / diamond-atari)은
게이트웨이가 세션마다 select 메시지로 알려줍니다.
"""

from __future__ import annotations

import logging
import os

import uvicorn

from .common.server import create_app

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)

ADAPTER = os.getenv("WM_MODEL", "oasis")
DEFAULT_MODEL = os.getenv("WM_DEFAULT_MODEL", "oasis" if ADAPTER == "oasis" else "diamond-csgo")


def factory(model_id: str):
    if ADAPTER == "oasis":
        from .adapters.oasis import build
    elif ADAPTER == "diamond":
        from .adapters.diamond import build
    elif ADAPTER == "dummy":
        from .adapters.dummy import make_dummy as build
    else:
        raise ValueError(f"알 수 없는 WM_MODEL: {ADAPTER}")
    return build(model_id)


app = create_app(factory, DEFAULT_MODEL)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("WM_HOST", "0.0.0.0"),
        port=int(os.getenv("WM_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        ws_max_size=16 * 1024 * 1024,
    )
