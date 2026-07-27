"""
워커 공통 런타임.

게이트웨이만 여기에 붙습니다. 브라우저가 직접 붙는 일은 없어야 합니다
(이 포트는 컨테이너 내부 네트워크에만 노출하세요).

핵심 설계 — **최신 액션 우선(latest-wins)**:
world model은 고정 FPS로 자기회귀 생성합니다. 액션을 큐에 쌓아두고 순서대로
소비하면 사용자 입력과 화면이 점점 벌어집니다. 그래서 큐를 두지 않고 항상
가장 최근 액션만 반영하고, 그 액션의 seq를 프레임 헤더에 실어 되돌려줍니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .actions import NEUTRAL, Action
from .base import WorldModel
from .encode import BACKEND, encode_jpeg, pack_frame

log = logging.getLogger("worker")


class SessionInput:
    """현재 눌려 있는 입력 상태. 액션은 쌓지 않고 덮어쓴다."""

    def __init__(self) -> None:
        self.current: Action = NEUTRAL
        self.reset_requested = False

    def apply(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "action":
            incoming = Action.from_json(msg)
            # 마우스 delta는 누적한다. 전송 주기가 생성 FPS보다 빠를 때
            # 중간 입력이 통째로 버려지는 것을 막기 위함.
            self.current = Action(
                seq=incoming.seq,
                keys=incoming.keys,
                dx=self.current.dx + incoming.dx,
                dy=self.current.dy + incoming.dy,
                left=incoming.left,
                right=incoming.right,
                hotbar=incoming.hotbar,
            )
        elif kind == "reset":
            self.reset_requested = True

    def take(self) -> Action:
        """추론에 쓸 액션을 꺼내고, 상대 마우스 이동은 소진시킨다."""
        action = self.current
        self.current = self.current.consumed()
        return action


ModelFactory = Callable[[str], WorldModel]


def create_app(factory: ModelFactory, default_model: str) -> FastAPI:
    """
    factory(model_id) -> WorldModel

    프로세스 기동 시 모델을 한 번 로드해두고 세션마다 reset()만 하는 것을
    권장합니다. 체크포인트 로딩은 수십 초가 걸리기도 해서, 세션마다 로드하면
    사용자가 그 시간을 그대로 기다리게 됩니다.
    """
    app = FastAPI(title=f"World Model Worker ({default_model})")
    state: dict = {"models": {}, "lock": asyncio.Lock()}

    async def get_model(model_id: str) -> WorldModel:
        async with state["lock"]:
            if model_id not in state["models"]:
                log.info("모델 로딩 model=%s", model_id)
                t0 = time.monotonic()
                loop = asyncio.get_running_loop()
                state["models"][model_id] = await loop.run_in_executor(
                    None, factory, model_id
                )
                log.info("모델 로딩 완료 model=%s %.1fs", model_id, time.monotonic() - t0)
            return state["models"][model_id]

    @app.on_event("startup")
    async def _warmup() -> None:
        log.info("JPEG 백엔드: %s", BACKEND)
        try:
            await get_model(default_model)
        except Exception:
            log.exception("워밍업 실패 — 첫 세션에서 다시 시도합니다")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "loaded": list(state["models"].keys())}

    @app.websocket("/session")
    async def session(ws: WebSocket) -> None:
        await ws.accept()
        model_id = default_model
        inputs = SessionInput()
        # 세션당 스레드 1개. 추론은 여기서 돌고 이벤트 루프는 자유롭게 둔다.
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        stop = asyncio.Event()

        async def reader() -> None:
            nonlocal model_id
            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "select":
                        model_id = msg.get("model") or default_model
                    elif msg.get("type") == "init":
                        model_id = msg.get("model") or model_id
                    else:
                        inputs.apply(msg)
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                stop.set()

        reader_task = asyncio.create_task(reader())

        try:
            # select/init이 도착할 짧은 여유를 준다 (게이트웨이가 즉시 보냄).
            await asyncio.sleep(0.05)
            model = await get_model(model_id)
            loop = asyncio.get_running_loop()

            frame = await loop.run_in_executor(pool, model.reset)
            frame_id = 0
            ack = 0
            period = 1.0 / max(1, model.fps)
            next_at = time.monotonic()

            while not stop.is_set():
                jpeg = await loop.run_in_executor(pool, encode_jpeg, frame, model.quality)
                try:
                    await ws.send_bytes(pack_frame(frame_id, ack, jpeg))
                except (WebSocketDisconnect, RuntimeError):
                    break
                frame_id += 1

                if inputs.reset_requested:
                    inputs.reset_requested = False
                    frame = await loop.run_in_executor(pool, model.reset)
                    continue

                action = inputs.take()
                ack = action.seq
                frame = await loop.run_in_executor(pool, model.step, action)

                # 페이싱: 추론이 목표 FPS보다 빠를 때만 쉰다.
                next_at += period
                delay = next_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_at = time.monotonic()   # 뒤처졌으면 따라잡지 않고 리셋

        except Exception:
            log.exception("세션 오류 model=%s", model_id)
            try:
                await ws.send_text(json.dumps({"type": "error", "message": "추론 실패"}))
            except Exception:
                pass
        finally:
            stop.set()
            reader_task.cancel()
            pool.shutdown(wait=False)
            try:
                await ws.close()
            except Exception:
                pass

    return app
