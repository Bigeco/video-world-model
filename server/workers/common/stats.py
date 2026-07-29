"""
세션 통계/로그 — 지연이 GPU inference 때문인지, 웹 전송(네트워크) 때문인지
구분하기 위한 기록기.

측정 방식:
  * GPU inference: model.reset()/model.step() 호출을 서버에서 직접 wall-clock으로 잰다.
  * JPEG 인코딩: encode_jpeg() 호출도 마찬가지로 직접 잰다.
  * 웹 전송(network): 서버가 직접 잴 수 없다(클라이언트까지 왕복해야 함). 대신
    클라이언트가 주기적으로 보고하는 RTT(액션 전송 → 그 액션이 반영된 프레임 수신)에서
    위에서 잰 서버 처리 시간(inference + encode)을 빼서 근사한다.
      network_estimate_ms ≈ client_round_trip.mean - (gpu_inference.mean + jpeg_encode.mean)
    게이트웨이 중계 + 실제 네트워크 왕복 + 약간의 스케줄링 지터가 여기 섞여 들어간다.

세션마다 playground/<model_id>/<시작시각>/ 아래에:
  * stats.json      — 누적 통계 (저장 요청마다, 그리고 세션 종료 시 갱신)
  * frame_NNNNNN_*.png — "현재 프레임 저장" 요청이 올 때마다 원본 해상도로 저장
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .actions import Action
from .encode import to_uint8_rgb

log = logging.getLogger("stats")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# workers/common → workers → server → VideoWorldModel, 그 아래 playground/
_DEFAULT_PLAYGROUND = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "playground"))
PLAYGROUND_DIR = os.getenv("WM_PLAYGROUND_DIR", _DEFAULT_PLAYGROUND)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": round(sum(values) / len(values), 2),
        "p50_ms": round(_percentile(values, 0.50), 2),
        "p95_ms": round(_percentile(values, 0.95), 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


class SessionStats:
    """세션 하나의 타이밍 / 액션 / RTT 로그. asyncio 이벤트 루프 안에서만 건드릴 것."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.started_at = time.time()
        self._t0 = time.monotonic()
        self.actions: List[Dict[str, Any]] = []
        self.infer_ms: List[float] = []       # model.reset()/model.step()
        self.encode_ms: List[float] = []      # JPEG 인코딩
        self.client_rtt_ms: List[float] = []  # 클라이언트가 보고한 왕복 지연
        self.frame_count = 0
        self._session_dir: Optional[str] = None

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def log_action(self, action: Action) -> None:
        self.actions.append({
            "t": round(self.elapsed(), 3),
            "seq": action.seq,
            "keys": sorted(action.keys),
            "dx": action.dx, "dy": action.dy,
            "left": action.left, "right": action.right,
            "hotbar": action.hotbar,
        })

    def log_infer(self, ms: float) -> None:
        self.infer_ms.append(ms)

    def log_encode(self, ms: float) -> None:
        self.encode_ms.append(ms)

    def log_client_rtt(self, ms: float) -> None:
        self.client_rtt_ms.append(ms)

    def session_dir(self) -> str:
        if self._session_dir is None:
            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started_at))
            self._session_dir = os.path.join(PLAYGROUND_DIR, self.model_id, ts)
            os.makedirs(self._session_dir, exist_ok=True)
        return self._session_dir

    def save_frame(self, frame: np.ndarray, tag: str = "manual") -> str:
        """현재 프레임을 원본 해상도 PNG로 저장하고 경로를 반환."""
        rgb = to_uint8_rgb(frame)
        path = os.path.join(self.session_dir(), f"frame_{self.frame_count:06d}_{tag}.png")
        Image.fromarray(rgb).save(path)
        return path

    def to_dict(self) -> Dict[str, Any]:
        mean_infer = sum(self.infer_ms) / len(self.infer_ms) if self.infer_ms else 0.0
        mean_encode = sum(self.encode_ms) / len(self.encode_ms) if self.encode_ms else 0.0
        mean_server = mean_infer + mean_encode
        mean_rtt = sum(self.client_rtt_ms) / len(self.client_rtt_ms) if self.client_rtt_ms else None
        network_ms = round(max(0.0, mean_rtt - mean_server), 2) if mean_rtt is not None else None

        return {
            "model": self.model_id,
            "started_at": self.started_at,
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_at)),
            "play_time_sec": round(self.elapsed(), 2),
            "frame_count": self.frame_count,
            "action_count": len(self.actions),
            "latency_breakdown_ms": {
                "gpu_inference": _summary(self.infer_ms),
                "jpeg_encode": _summary(self.encode_ms),
                "server_total_mean_ms": round(mean_server, 2) if (self.infer_ms or self.encode_ms) else None,
                "client_round_trip": _summary(self.client_rtt_ms),
                "network_estimate_mean_ms": network_ms,
                "note": ("network_estimate_mean_ms = client_round_trip.mean_ms - server_total_mean_ms. "
                         "게이트웨이 중계 + 실제 네트워크 왕복 + 스케줄링 지터를 합친 근사치입니다. "
                         "client_round_trip 표본이 없으면 null입니다(클라이언트가 아직 client_metric을 "
                         "보내지 않았거나 데모 모드)."),
            },
            "actions": self.actions,
        }

    def write(self) -> str:
        """session_dir/stats.json 에 현재까지의 누적 통계를 (덮어)쓴다."""
        path = os.path.join(self.session_dir(), "stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path
