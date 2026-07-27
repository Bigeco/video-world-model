"""
가중치 없이 전체 스택을 돌려보기 위한 더미 world model.

**실제 모델이 아닙니다.** 액션에 반응하는 절차적 씬을 numpy로 렌더링할 뿐입니다.
용도는 이것들입니다:

  * 체크포인트를 받기 전에 게이트웨이 ↔ 워커 ↔ 브라우저 경로를 끝까지 검증
  * 터널·TLS·CORS·대기열 설정이 맞는지 확인
  * 실제 네트워크에서의 지연/대역폭 측정 (모델을 로드하지 않고)

`WM_DUMMY=1`이면 어댑터가 실제 모델 대신 이걸 씁니다.
"""

from __future__ import annotations

import math

import numpy as np

from ..common.actions import Action
from ..common.base import WorldModel


class DummyWorldModel(WorldModel):
    """원근 투영 바닥 + 하늘. 카메라를 액션으로 조작할 수 있습니다."""

    def __init__(self, width: int = 320, height: int = 180, fps: int = 20,
                 palette: tuple = ((92, 140, 58), (124, 179, 66), (144, 202, 249)),
                 label: str = "DUMMY"):
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = 80
        self.palette = np.array(palette, dtype=np.float32)
        self.label = label
        self._rng = np.random.default_rng(0)
        self.reset()

    # -- 상태 ---------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.x = 0.0
        self.z = 0.0
        self.y = 1.6
        self.vy = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.t = 0
        return self._render()

    def step(self, action: Action) -> np.ndarray:
        dt = 1.0 / max(1, self.fps)

        # 시점
        self.yaw += action.dx * 0.0035
        self.pitch = float(np.clip(self.pitch - action.dy * 0.0035, -1.2, 1.2))

        # 이동
        fx = float(action.held("KeyD", "ArrowRight")) - float(action.held("KeyA", "ArrowLeft"))
        fz = float(action.held("KeyW", "ArrowUp")) - float(action.held("KeyS", "ArrowDown"))
        norm = math.hypot(fx, fz) or 1.0
        fx, fz = fx / norm, fz / norm

        speed = 2.2 if action.held("ShiftLeft") else 5.4
        if action.held("ControlLeft"):
            speed *= 1.9

        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        self.x += (fx * cy + fz * sy) * speed * dt
        self.z += (fz * cy - fx * sy) * speed * dt

        # 점프 + 중력
        if action.held("Space") and self.y <= 1.61:
            self.vy = 4.5
        self.vy -= 13.0 * dt
        self.y += self.vy * dt
        if self.y < 1.6:
            self.y, self.vy = 1.6, 0.0

        self.t += 1
        return self._render()

    # -- 렌더링 -------------------------------------------------------------

    def _render(self) -> np.ndarray:
        H, W = self.height, self.width
        focal = (H / 2) / math.tan(0.62)
        horizon = H / 2 + self.pitch * focal

        img = np.empty((H, W, 3), dtype=np.float32)

        # 하늘: 위에서 아래로 그라데이션
        rows = np.arange(H, dtype=np.float32)[:, None]
        sky_t = np.clip(rows / max(1.0, horizon), 0.0, 1.0)
        img[:] = self.palette[2] * (1 - sky_t)[..., None] + np.float32(210) * sky_t[..., None]

        # 바닥: 원근 투영 체커보드 (mode-7 방식)
        y_idx = np.arange(H, dtype=np.float32)
        ground_rows = np.where(y_idx > horizon + 0.5)[0]
        if ground_rows.size:
            yy = y_idx[ground_rows] - horizon                 # (R,)
            depth = (self.y * focal) / yy                      # 각 행의 월드 깊이
            xs = (np.arange(W, dtype=np.float32) - W / 2) / focal
            cam_x = depth[:, None] * xs[None, :]               # (R, W)
            cam_z = np.repeat(depth[:, None], W, axis=1)

            sy, cy = math.sin(self.yaw), math.cos(self.yaw)
            wx = self.x + cam_x * cy + cam_z * sy
            wz = self.z - cam_x * sy + cam_z * cy

            checker = ((np.floor(wx) + np.floor(wz)).astype(np.int64) & 1).astype(np.float32)
            base = self.palette[0][None, None, :] * (1 - checker)[..., None] + \
                   self.palette[1][None, None, :] * checker[..., None]

            fog = np.clip(1.0 - depth[:, None] / 60.0, 0.0, 1.0)[..., None]
            img[ground_rows] = base * fog + np.float32(200) * (1 - fog)

        # diffusion 아티팩트 흉내 — 미세 노이즈
        img += self._rng.normal(0.0, 3.0, size=img.shape).astype(np.float32)

        # 조준점
        cx, cyy = W // 2, H // 2
        img[cyy, cx - 4:cx + 5] = 255.0
        img[cyy - 4:cyy + 5, cx] = 255.0

        return np.clip(img, 0, 255).astype(np.uint8)


def make_dummy(model_id: str) -> DummyWorldModel:
    """모델별로 색과 해상도를 다르게 해서 어떤 워커에 붙었는지 눈으로 구분되게."""
    presets = {
        "oasis": dict(width=320, height=180, fps=20, label="OASIS",
                      palette=((92, 140, 58), (124, 179, 66), (144, 202, 249))),
        "diamond-csgo": dict(width=280, height=150, fps=10, label="DIAMOND/CSGO",
                             palette=((161, 135, 95), (200, 177, 138), (224, 214, 195))),
        "diamond-atari": dict(width=128, height=128, fps=30, label="DIAMOND/ATARI",
                              palette=((26, 35, 126), (216, 27, 96), (0, 172, 193))),
    }
    return DummyWorldModel(**presets.get(model_id, presets["oasis"]))
