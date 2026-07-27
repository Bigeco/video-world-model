"""
게이트웨이 — 브라우저가 붙는 단일 진입점.

역할:
  1. 액세스 토큰 검증 (상수시간 비교)
  2. 모델별 세션 대기열 관리 (GPU 1장 = 동시 1~2세션)
  3. 세션 TTL 강제
  4. 워커 컨테이너로 WebSocket 양방향 프록시

브라우저는 이 서버에만 연결합니다. 워커 포트는 절대 외부에 노출하지 마세요.
프로토콜 명세는 최상위 README.md 참조.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import websockets
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger("gateway")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

# close code — 프론트엔드가 이 값들을 그대로 해석합니다. 바꾸지 마세요.
CLOSE_BAD_TOKEN = 4001
CLOSE_SESSION_OVER = 4008
CLOSE_SERVER_FULL = 4029
CLOSE_WORKER_DOWN = 4011


def _env_list(name: str, default: str = "") -> List[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


TOKENS = set(_env_list("WM_TOKENS"))
ALLOWED_ORIGINS = _env_list("WM_ALLOWED_ORIGINS", "*")

SESSION_TTL = int(os.getenv("WM_SESSION_TTL", "120"))          # 세션당 초
MAX_QUEUE = int(os.getenv("WM_MAX_QUEUE", "12"))               # 모델당 대기열 상한
IDLE_TIMEOUT = int(os.getenv("WM_IDLE_TIMEOUT", "30"))         # 입력 없으면 회수

# model_id -> (워커 주소, 동시 세션 수)
WORKERS: Dict[str, str] = {
    "oasis":         os.getenv("WM_WORKER_OASIS", "ws://oasis:8000/session"),
    "diamond-csgo":  os.getenv("WM_WORKER_DIAMOND_CSGO", "ws://diamond:8000/session"),
    "diamond-atari": os.getenv("WM_WORKER_DIAMOND_ATARI", "ws://diamond:8000/session"),
}
CAPACITY: Dict[str, int] = {
    "oasis":         int(os.getenv("WM_CAP_OASIS", "1")),
    "diamond-csgo":  int(os.getenv("WM_CAP_DIAMOND_CSGO", "1")),
    "diamond-atari": int(os.getenv("WM_CAP_DIAMOND_ATARI", "2")),
}


def check_token(token: str) -> bool:
    """상수시간 비교. 토큰을 하나도 설정하지 않으면 개발 모드로 간주해 통과시킵니다."""
    if not TOKENS:
        log.warning("WM_TOKENS 미설정 — 인증 없이 동작 중입니다 (개발 모드)")
        return True
    return any(hmac.compare_digest(token, t) for t in TOKENS)


# --------------------------------------------------------------------------
# 대기열
# --------------------------------------------------------------------------

@dataclass
class Waiter:
    fut: asyncio.Future
    notify: "asyncio.Queue[dict]"


class ModelQueue:
    """모델 하나에 대한 입장 제어. 선착순, 취소 안전."""

    def __init__(self, model_id: str, capacity: int):
        self.model_id = model_id
        self.capacity = max(1, capacity)
        self.active = 0
        self.waiters: Deque[Waiter] = deque()
        self.lock = asyncio.Lock()
        self.durations: Deque[float] = deque(maxlen=8)

    @property
    def avg_session(self) -> float:
        if self.durations:
            return sum(self.durations) / len(self.durations)
        return float(SESSION_TTL)

    def depth(self) -> int:
        return len(self.waiters)

    async def acquire(self, notify: "asyncio.Queue[dict]") -> bool:
        """입장할 때까지 대기. 자리가 없으면 False (대기열 포화)."""
        async with self.lock:
            if self.active < self.capacity and not self.waiters:
                self.active += 1
                return True
            if len(self.waiters) >= MAX_QUEUE:
                return False
            waiter = Waiter(fut=asyncio.get_running_loop().create_future(), notify=notify)
            self.waiters.append(waiter)

        await self._broadcast()
        try:
            await waiter.fut
            return True
        except asyncio.CancelledError:
            async with self.lock:
                try:
                    self.waiters.remove(waiter)
                except ValueError:
                    # 이미 입장 처리된 뒤 취소됨 → 자리를 반납해야 함
                    if waiter.fut.done() and not waiter.fut.cancelled():
                        self.active -= 1
                        await self._promote_locked()
            await self._broadcast()
            raise

    async def release(self, duration: Optional[float] = None) -> None:
        async with self.lock:
            self.active = max(0, self.active - 1)
            if duration is not None:
                self.durations.append(duration)
            await self._promote_locked()
        await self._broadcast()

    async def _promote_locked(self) -> None:
        """lock을 이미 쥔 상태에서 호출. 빈 자리만큼 앞에서부터 입장시킴."""
        while self.active < self.capacity and self.waiters:
            nxt = self.waiters.popleft()
            if nxt.fut.done():
                continue
            self.active += 1
            nxt.fut.set_result(True)

    async def _broadcast(self) -> None:
        """대기 중인 모두에게 현재 순번을 알림."""
        async with self.lock:
            snapshot = list(self.waiters)
            avg = self.avg_session
        for i, w in enumerate(snapshot):
            msg = {
                "type": "queue",
                "position": i + 1,
                "ahead": i,
                "eta": round(avg * (i + 1)),
            }
            try:
                w.notify.put_nowait(msg)
            except asyncio.QueueFull:
                pass


QUEUES: Dict[str, ModelQueue] = {
    mid: ModelQueue(mid, CAPACITY.get(mid, 1)) for mid in WORKERS
}


# --------------------------------------------------------------------------
# 앱
# --------------------------------------------------------------------------

app = FastAPI(title="World Model Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """브라우저로 이 주소를 열었을 때 뭘 해야 하는지 알려준다.

    게이트웨이는 API 서버라서 화면이 없다. 웹사이트는 index.html을 따로 열어야 한다.
    """
    return {
        "service": "World Model Gateway",
        "note": "여기는 API 서버입니다. 웹사이트 화면은 index.html을 브라우저로 여세요.",
        "endpoints": {
            "GET /healthz": "상태 확인",
            "GET /models": "모델 목록과 대기열 상태",
            "WS  /ws/{model_id}?token=...": "세션 (브라우저 주소창으로는 열 수 없음)",
        },
        "models": list(WORKERS.keys()),
        "auth": "토큰 설정됨" if TOKENS else "토큰 미설정 — 인증 없이 동작 중",
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": time.time()}


@app.get("/models")
async def models():
    return {
        "models": [
            {
                "id": mid,
                "capacity": q.capacity,
                "active": q.active,
                "queued": q.depth(),
                "eta": round(q.avg_session * (q.depth() + 1)) if q.depth() else 0,
            }
            for mid, q in QUEUES.items()
        ],
        "session_ttl": SESSION_TTL,
    }


@app.websocket("/ws/{model_id}")
async def ws_session(ws: WebSocket, model_id: str, token: str = Query(default="")):
    if model_id not in WORKERS:
        await ws.close(code=1008)
        return

    if not check_token(token):
        await ws.accept()
        await ws.close(code=CLOSE_BAD_TOKEN)
        log.info("token rejected model=%s", model_id)
        return

    await ws.accept()
    queue = QUEUES[model_id]

    # 클라이언트 수신은 항상 이 태스크 하나만 담당한다.
    # 대기 중에 온 메시지(init 등)는 inbox에 쌓였다가 입장 후 워커로 전달된다.
    inbox: "asyncio.Queue[Optional[str]]" = asyncio.Queue(maxsize=256)
    notify: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=32)

    async def reader():
        try:
            while True:
                msg = await ws.receive_text()
                if inbox.full():          # 오래된 액션은 버린다 (최신 우선)
                    try:
                        inbox.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                inbox.put_nowait(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await inbox.put(None)

    async def notifier():
        """대기열 순번을 클라이언트로 흘려보냄."""
        try:
            while True:
                msg = await notify.get()
                await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    reader_task = asyncio.create_task(reader())
    notify_task = asyncio.create_task(notifier())
    admitted = False
    started = 0.0

    try:
        admitted = await queue.acquire(notify)
        if not admitted:
            await ws.close(code=CLOSE_SERVER_FULL)
            return

        notify_task.cancel()
        started = time.monotonic()
        log.info("session start model=%s active=%d", model_id, queue.active)

        try:
            await _run_session(ws, model_id, inbox)
        except _WorkerDown:
            await _safe_close(ws, CLOSE_WORKER_DOWN)
        except _SessionExpired:
            await _safe_close(ws, CLOSE_SESSION_OVER)

    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("session error model=%s", model_id)
        await _safe_close(ws, 1011)
    finally:
        notify_task.cancel()
        reader_task.cancel()
        if admitted:
            await queue.release(time.monotonic() - started if started else None)
            log.info("session end model=%s active=%d queued=%d",
                     model_id, queue.active, queue.depth())


class _WorkerDown(Exception):
    pass


class _SessionExpired(Exception):
    pass


async def _safe_close(ws: WebSocket, code: int) -> None:
    try:
        await ws.close(code=code)
    except Exception:
        pass


async def _run_session(ws: WebSocket, model_id: str, inbox: "asyncio.Queue") -> None:
    """워커에 연결하고 클라이언트 ↔ 워커를 양방향으로 중계."""
    url = WORKERS[model_id]
    try:
        worker = await asyncio.wait_for(
            websockets.connect(url, max_size=None, ping_interval=20), timeout=10
        )
    except Exception as e:
        log.error("worker 연결 실패 model=%s url=%s err=%s", model_id, url, e)
        raise _WorkerDown()

    async with worker:
        # 워커에 어떤 모델인지 알려준다 (한 컨테이너가 변종을 여러 개 서빙할 수 있음)
        await worker.send(json.dumps({"type": "select", "model": model_id}))
        await ws.send_text(json.dumps({"type": "ready", "ttl": SESSION_TTL}))

        last_input = time.monotonic()

        async def client_to_worker():
            nonlocal last_input
            while True:
                msg = await inbox.get()
                if msg is None:               # 클라이언트 연결 종료
                    return
                last_input = time.monotonic()
                await worker.send(msg)

        async def worker_to_client():
            async for msg in worker:
                if isinstance(msg, (bytes, bytearray)):
                    await ws.send_bytes(bytes(msg))
                else:
                    await ws.send_text(msg)

        async def watchdog():
            deadline = time.monotonic() + SESSION_TTL
            while True:
                await asyncio.sleep(1)
                now = time.monotonic()
                if now >= deadline:
                    raise _SessionExpired()
                if now - last_input > IDLE_TIMEOUT:
                    log.info("idle 회수 model=%s", model_id)
                    raise _SessionExpired()

        tasks = [
            asyncio.create_task(client_to_worker()),
            asyncio.create_task(worker_to_client()),
            asyncio.create_task(watchdog()),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                exc = t.exception()
                if exc:
                    raise exc
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
