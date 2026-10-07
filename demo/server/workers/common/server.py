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
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .actions import NEUTRAL, Action
from .base import WorldModel
from .encode import BACKEND, encode_jpeg, pack_frame
from .stats import SessionStats

log = logging.getLogger("worker")

# 기본은 지연 로딩: 프로세스가 뜬다고 바로 GPU를 잡지 않고, 첫 세션이 실제로
# 붙을 때 로드한다. "실행" 버튼을 누르기 전엔 GPU를 점유하지 않는다.
# 1로 켜면 기존처럼 기동 시 미리 로드해둔다(첫 세션이 로딩을 안 기다림).
EAGER_WARMUP = os.getenv("WM_EAGER_WARMUP", "0") == "1"

# 세션이 끝나서 그 모델을 쓰는 사람이 아무도 없어지면(active==0) 바로 GPU에서
# 내린다. 0으로 주면 언로드하지 않는다(한 번 로드되면 프로세스가 죽을 때까지
# 유지 — 재접속이 잦을 때 재로딩을 피하고 싶으면 이 값을 켜라).
IMMEDIATE_UNLOAD = os.getenv("WM_MODEL_IMMEDIATE_UNLOAD", "1") == "1"


class SessionInput:
    """현재 눌려 있는 입력 상태. 액션은 쌓지 않고 덮어쓴다."""

    def __init__(self) -> None:
        self.current: Action = NEUTRAL
        self.reset_requested = False
        self.save_frame_requested = False

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

    기본은 지연 로딩(WM_EAGER_WARMUP=0): 첫 세션이 실제로 붙을 때만 로드한다.
    WM_MODEL_IMMEDIATE_UNLOAD=1(기본)이면 그 모델을 쓰는 세션이 하나도 안
    남는 순간 바로 GPU에서 내린다. 세션이 겹쳐 있는 동안(active>0)엔 절대
    내리지 않는다.
    """
    app = FastAPI(title=f"World Model Worker ({default_model})")
    state: dict = {
        "models": {},   # model_id -> WorldModel (로드된 것만)
        "lock": asyncio.Lock(),
        "active": {},    # model_id -> 현재 진행 중인 세션 수
    }

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

    async def _unload(model_id: str) -> None:
        async with state["lock"]:
            model = state["models"].pop(model_id, None)
        if model is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, model.close)
        except Exception:
            log.exception("모델 언로드 중 close() 실패 model=%s", model_id)
        del model
        log.info("모델 언로드(GPU 메모리 반환) model=%s", model_id)

    @app.on_event("startup")
    async def _on_startup() -> None:
        log.info("JPEG 백엔드: %s", BACKEND)
        if EAGER_WARMUP:
            try:
                await get_model(default_model)
            except Exception:
                log.exception("워밍업 실패 — 첫 세션에서 다시 시도합니다")
        else:
            log.info("WM_EAGER_WARMUP=0 — 첫 세션이 붙을 때 모델을 로드합니다 (기동 시 GPU를 잡지 않음)")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "loaded": list(state["models"].keys())}

    @app.websocket("/session")
    async def session(ws: WebSocket) -> None:
        await ws.accept()
        model_id = default_model
        inputs = SessionInput()
        # reader가 시작하자마자 도착하는 action도 놓치지 않도록 미리 만들어둔다.
        # model_id가 select/init으로 바뀌면 아래에서 stats.model_id도 맞춰준다.
        stats: Optional[SessionStats] = SessionStats(model_id)
        # 세션당 스레드 1개. 추론은 여기서 돌고 이벤트 루프는 자유롭게 둔다.
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        stop = asyncio.Event()
        active_counted = False   # get_model() 이후에만 True — 실패한 세션은 카운트하지 않는다

        async def reader() -> None:
            nonlocal model_id
            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    kind = msg.get("type")
                    if kind == "select":
                        model_id = msg.get("model") or default_model
                    elif kind == "init":
                        model_id = msg.get("model") or model_id
                    elif kind == "save_frame":
                        inputs.save_frame_requested = True
                    elif kind == "client_metric":
                        rtt = msg.get("rtt")
                        if stats is not None and isinstance(rtt, (int, float)) and rtt > 0:
                            stats.log_client_rtt(float(rtt))
                    else:
                        inputs.apply(msg)
                        if kind == "action" and stats is not None:
                            stats.log_action(Action.from_json(msg))
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                stop.set()

        reader_task = asyncio.create_task(reader())

        try:
            # select/init이 도착할 짧은 여유를 준다 (게이트웨이가 즉시 보냄).
            await asyncio.sleep(0.05)
            model = await get_model(model_id)
            async with state["lock"]:
                state["active"][model_id] = state["active"].get(model_id, 0) + 1
            active_counted = True
            loop = asyncio.get_running_loop()
            stats.model_id = model_id     # select/init으로 바뀌었을 수 있으니 최종값으로 맞춘다

            t0 = time.monotonic()
            frame = await loop.run_in_executor(pool, model.reset)
            stats.log_infer((time.monotonic() - t0) * 1000)
            frame_id = 0
            ack = 0
            period = 1.0 / max(1, model.fps)
            next_at = time.monotonic()

            while not stop.is_set():
                t0 = time.monotonic()
                jpeg = await loop.run_in_executor(pool, encode_jpeg, frame, model.quality)
                stats.log_encode((time.monotonic() - t0) * 1000)
                try:
                    await ws.send_bytes(pack_frame(frame_id, ack, jpeg))
                except (WebSocketDisconnect, RuntimeError):
                    break
                frame_id += 1
                stats.frame_count = frame_id

                if inputs.save_frame_requested:
                    inputs.save_frame_requested = False
                    path = stats.save_frame(frame)
                    stats.write()
                    log.info("프레임 저장 model=%s path=%s", model_id, path)

                if inputs.reset_requested:
                    inputs.reset_requested = False
                    t0 = time.monotonic()
                    frame = await loop.run_in_executor(pool, model.reset)
                    stats.log_infer((time.monotonic() - t0) * 1000)
                    continue

                action = inputs.take()
                ack = action.seq
                t0 = time.monotonic()
                frame = await loop.run_in_executor(pool, model.step, action)
                stats.log_infer((time.monotonic() - t0) * 1000)

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
            if active_counted:
                async with state["lock"]:
                    left = max(0, state["active"].get(model_id, 1) - 1)
                    state["active"][model_id] = left
                if left == 0 and IMMEDIATE_UNLOAD:
                    await _unload(model_id)
            if stats is not None and (stats.frame_count or stats.actions):
                try:
                    path = stats.write()
                    log.info("세션 통계 저장 model=%s path=%s", model_id, path)
                except Exception:
                    log.exception("세션 통계 저장 실패 model=%s", model_id)
            try:
                await ws.close()
            except Exception:
                pass

    return app
