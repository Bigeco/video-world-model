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
from importlib import import_module

import uvicorn

from .common.server import create_app

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)

ADAPTER = os.getenv("WM_MODEL", "oasis")
DEFAULT_MODEL = os.getenv("WM_DEFAULT_MODEL", ADAPTER if ADAPTER != "dummy" else "oasis")
FACTORY_PATH = os.getenv("VWM_MODEL_FACTORY", "")


def _private_factory():
    if not FACTORY_PATH:
        return None
    module_name, separator, function_name = FACTORY_PATH.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("VWM_MODEL_FACTORY must use module:function format")
    return getattr(import_module(module_name), function_name)


def factory(model_id: str):
    private = _private_factory()
    if private is not None:
        return private(model_id, adapter=ADAPTER)
    if ADAPTER == "dummy" or os.getenv("WM_DUMMY", "0") == "1":
        from .adapters.dummy import make_dummy
        return make_dummy(model_id)
    raise RuntimeError(
        "Real model implementations are private. Set "
        "VWM_MODEL_FACTORY=vwm.models.registry:create_model locally."
    )


app = create_app(factory, DEFAULT_MODEL)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("WM_HOST", "0.0.0.0"),
        port=int(os.getenv("WM_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        ws_max_size=16 * 1024 * 1024,
    )
