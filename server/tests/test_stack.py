"""
통합 테스트 — 게이트웨이 + 워커를 실제로 띄우고 브라우저처럼 붙어본다.

    python tests/test_stack.py

더미 모델을 쓰므로 GPU도 체크포인트도 필요 없습니다.
프론트엔드가 의존하는 계약을 전부 검증합니다:
  * 잘못된 토큰 → close 4001
  * 두 번째 접속자 → queue 메시지, 앞 세션 종료 후 자동 승격
  * ready 메시지와 ttl
  * 바이너리 프레임 헤더 (uint32 BE frame_id + ack_seq)
  * ack_seq가 클라이언트 seq를 되돌려주는지 (지연 계산의 전제)
  * 세션 TTL 만료 → close 4008
  * 액션이 실제로 화면을 바꾸는지
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WM_DUMMY", "1")
os.environ["WM_TOKENS"] = "good-token"
os.environ["WM_SESSION_TTL"] = "6"
os.environ["WM_IDLE_TIMEOUT"] = "60"
os.environ["WM_CAP_OASIS"] = "1"
os.environ["WM_WORKER_OASIS"] = "ws://127.0.0.1:8931/session"
os.environ["WM_WORKER_DIAMOND_CSGO"] = "ws://127.0.0.1:8931/session"
os.environ["WM_WORKER_DIAMOND_ATARI"] = "ws://127.0.0.1:8931/session"
os.environ["WM_MODEL"] = "dummy"
os.environ["WM_DEFAULT_MODEL"] = "oasis"
os.environ["LOG_LEVEL"] = "WARNING"

import uvicorn                      # noqa: E402
import websockets                   # noqa: E402

GATEWAY_PORT = 8930
WORKER_PORT = 8931

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    mark = " ok " if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"   [{detail}]" if detail and not ok else ""))


class Server:
    def __init__(self, app, port):
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port,
                             log_level="error", ws_max_size=16 * 1024 * 1024)
        self.server = uvicorn.Server(cfg)
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(100):
            if self.server.started:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("서버 기동 실패")

    async def stop(self):
        self.server.should_exit = True
        if self.task:
            await asyncio.wait_for(self.task, timeout=10)


def gw_url(model="oasis", token="good-token"):
    return f"ws://127.0.0.1:{GATEWAY_PORT}/ws/{model}?token={token}"


def parse_frame(buf: bytes):
    frame_id, ack = struct.unpack(">II", buf[:8])
    return frame_id, ack, buf[8:]


async def recv_until(ws, kind: str, timeout=10.0):
    """지정한 종류가 나올 때까지 읽는다. kind='binary'면 첫 바이너리 프레임."""
    deadline = time.monotonic() + timeout
    seen = []
    while time.monotonic() < deadline:
        msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
        if isinstance(msg, (bytes, bytearray)):
            if kind == "binary":
                return bytes(msg)
            continue
        data = json.loads(msg)
        seen.append(data.get("type"))
        if data.get("type") == kind:
            return data
    raise asyncio.TimeoutError(f"{kind} 미수신, 받은 것: {seen}")


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

async def t_bad_token():
    try:
        async with websockets.connect(gw_url(token="wrong")) as ws:
            await ws.recv()
        check("잘못된 토큰은 거부된다", False, "연결이 유지됨")
    except websockets.exceptions.ConnectionClosed as e:
        check("잘못된 토큰 → close 4001", e.code == 4001, f"code={e.code}")


async def t_happy_path():
    async with websockets.connect(gw_url()) as ws:
        await ws.send(json.dumps({"type": "init", "model": "oasis", "fps": 20}))
        ready = await recv_until(ws, "ready")
        check("ready 수신", ready.get("type") == "ready")
        check("ready에 ttl 포함", isinstance(ready.get("ttl"), int) and ready["ttl"] > 0)

        raw = await recv_until(ws, "binary")
        fid, ack, jpeg = parse_frame(raw)
        check("바이너리 프레임 도착", len(jpeg) > 0, f"{len(jpeg)}B")
        check("JPEG 매직바이트(FFD8FF)", jpeg[:3] == b"\xff\xd8\xff", jpeg[:3].hex())
        check("첫 프레임 frame_id=0", fid == 0, f"fid={fid}")

        # 액션을 보내고 ack_seq가 되돌아오는지 — 지연 계산의 전제
        await ws.send(json.dumps({
            "type": "action", "seq": 777, "keys": ["KeyW"],
            "mouse": {"dx": 40, "dy": 0}, "buttons": {"left": False, "right": False},
            "hotbar": 1,
        }))
        got_ack = False
        for _ in range(40):
            raw = await recv_until(ws, "binary")
            _, ack, _ = parse_frame(raw)
            if ack == 777:
                got_ack = True
                break
        check("ack_seq가 클라이언트 seq를 되돌려준다", got_ack, f"마지막 ack={ack}")

        # 화면이 실제로 변하는지 (액션 반영 확인)
        frames = []
        for _ in range(6):
            await ws.send(json.dumps({
                "type": "action", "seq": 800, "keys": ["KeyW"],
                "mouse": {"dx": 25, "dy": 0}, "buttons": {}, "hotbar": 1,
            }))
            frames.append(parse_frame(await recv_until(ws, "binary"))[2])
        check("액션에 따라 프레임이 변한다", len(set(frames)) > 1,
              f"고유 프레임 {len(set(frames))}/6")

        # frame_id 단조 증가
        a = parse_frame(await recv_until(ws, "binary"))[0]
        b = parse_frame(await recv_until(ws, "binary"))[0]
        check("frame_id 단조 증가", b > a, f"{a} → {b}")


async def t_queue():
    """capacity=1이므로 두 번째 접속자는 대기열에 들어가야 한다."""
    first = await websockets.connect(gw_url())
    await first.send(json.dumps({"type": "init", "model": "oasis"}))
    await recv_until(first, "ready")

    second = await websockets.connect(gw_url())
    q = await recv_until(second, "queue", timeout=5)
    check("두 번째 접속자는 대기열로", q.get("type") == "queue")
    check("대기 순번 1번", q.get("position") == 1, f"position={q.get('position')}")
    check("앞에 0명 (현재 세션은 active)", q.get("ahead") == 0, f"ahead={q.get('ahead')}")
    check("eta 제공", isinstance(q.get("eta"), (int, float)))

    # 첫 세션을 끊으면 두 번째가 승격되어야 한다
    await first.close()
    ready = await recv_until(second, "ready", timeout=10)
    check("앞 세션 종료 시 자동 승격", ready.get("type") == "ready")
    await second.close()


async def t_ttl():
    """WM_SESSION_TTL=6초로 설정했으므로 그 뒤 4008로 닫혀야 한다."""
    ws = await websockets.connect(gw_url())
    await ws.send(json.dumps({"type": "init", "model": "oasis"}))
    await recv_until(ws, "ready")
    t0 = time.monotonic()
    code = None
    try:
        while time.monotonic() - t0 < 15:
            await ws.send(json.dumps({"type": "action", "seq": 1, "keys": [],
                                      "mouse": {"dx": 0, "dy": 0}, "buttons": {}}))
            await asyncio.wait_for(ws.recv(), timeout=12)
    except websockets.exceptions.ConnectionClosed as e:
        code = e.code
    elapsed = time.monotonic() - t0
    check("세션 TTL 만료 → close 4008", code == 4008, f"code={code}")
    check("TTL 시점이 설정값 근처", 5 <= elapsed <= 11, f"{elapsed:.1f}s")


async def t_unknown_model():
    try:
        async with websockets.connect(gw_url(model="nope")) as ws:
            await ws.recv()
        check("존재하지 않는 모델은 거부", False)
    except Exception as e:
        code = getattr(e, "code", None)
        check("존재하지 않는 모델은 거부", code in (1008, 1006) or code is None, f"code={code}")


def t_action_mapping():
    """액션 매핑은 순수 함수라 서버 없이 검증."""
    from workers.common.actions import (Action, csgo_vector, minecraft_vector,
                                        to_atari, to_minecraft)

    a = Action(seq=1, keys=frozenset({"KeyW", "ShiftLeft"}), dx=100, dy=-50, left=True)
    mc = to_minecraft(a)
    check("Minecraft: W→forward", mc["forward"] == 1)
    check("Minecraft: Shift→sneak", mc["sneak"] == 1)
    check("Minecraft: 좌클릭→attack", mc["attack"] == 1)
    check("Minecraft: 카메라 ±20도 클램프",
          abs(mc["camera"][0]) <= 20 and abs(mc["camera"][1]) <= 20, str(mc["camera"]))

    big = to_minecraft(Action(dx=99999, dy=-99999))
    check("Minecraft: 과도한 마우스 입력도 클램프",
          abs(big["camera"][0]) == 20 and abs(big["camera"][1]) == 20)

    check("Minecraft 벡터 길이 13", minecraft_vector(a).shape == (13,))
    check("CS:GO 벡터 길이 62", csgo_vector(a).shape == (62,),
          str(csgo_vector(a).shape))

    check("Atari: NOOP", to_atari(Action()) == 0)
    check("Atari: FIRE", to_atari(Action(keys=frozenset({"Space"}))) == 1)
    check("Atari: UP", to_atari(Action(keys=frozenset({"ArrowUp"}))) == 2)
    check("Atari: UP+FIRE", to_atari(Action(keys=frozenset({"ArrowUp", "Space"}))) == 10)
    check("Atari: 상충 입력(좌+우) 상쇄",
          to_atari(Action(keys=frozenset({"ArrowLeft", "ArrowRight"}))) == 0)

    consumed = a.consumed()
    check("consumed(): 마우스 delta 소진", consumed.dx == 0 and consumed.dy == 0)
    check("consumed(): 키는 유지", consumed.keys == a.keys)


def t_encode():
    import numpy as np
    from workers.common.encode import encode_jpeg, pack_frame, to_uint8_rgb

    check("uint8 HWC 통과", to_uint8_rgb(np.zeros((4, 5, 3), np.uint8)).shape == (4, 5, 3))
    check("CHW → HWC 변환", to_uint8_rgb(np.zeros((3, 4, 5), np.float32)).shape == (4, 5, 3))
    check("그레이스케일 → 3채널", to_uint8_rgb(np.zeros((4, 5), np.uint8)).shape == (4, 5, 3))

    f01 = to_uint8_rgb(np.ones((2, 2, 3), np.float32))            # [0,1] 범위
    check("[0,1] float → 255 스케일", int(f01.max()) == 255, str(f01.max()))
    fneg = to_uint8_rgb(np.full((2, 2, 3), -1.0, np.float32))     # [-1,1] 범위
    check("[-1,1] float → 0 매핑", int(fneg.max()) == 0, str(fneg.max()))

    jpeg = encode_jpeg(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    check("JPEG 인코딩", jpeg[:3] == b"\xff\xd8\xff")
    packed = pack_frame(0xFFFFFFFF, 12345, jpeg)
    fid, ack = struct.unpack(">II", packed[:8])
    check("pack_frame 라운드트립", fid == 0xFFFFFFFF and ack == 12345)
    check("uint32 오버플로 랩어라운드", struct.unpack(">II", pack_frame(2**32 + 5, 0, b"")[:8])[0] == 5)


async def main():
    from gateway.app import app as gateway_app
    from workers.run import app as worker_app

    print("\n[1] 순수 함수 — 액션 매핑")
    t_action_mapping()
    print("\n[2] 순수 함수 — 프레임 인코딩")
    t_encode()

    worker = Server(worker_app, WORKER_PORT)
    gateway = Server(gateway_app, GATEWAY_PORT)
    await worker.start()
    await gateway.start()
    await asyncio.sleep(0.5)

    try:
        print("\n[3] 인증")
        await t_bad_token()
        await t_unknown_model()
        print("\n[4] 정상 세션")
        await t_happy_path()
        print("\n[5] 대기열")
        await t_queue()
        print("\n[6] 세션 TTL")
        await t_ttl()
    finally:
        await gateway.stop()
        await worker.stop()

    print(f"\n{'='*52}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
