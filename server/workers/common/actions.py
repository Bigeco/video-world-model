"""
브라우저 입력 → 모델별 액션 스페이스 변환.

프론트엔드는 KeyboardEvent.code 문자열을 그대로 보냅니다("KeyW", "ShiftLeft", ...).
키 이름을 모델 의미로 바꾸는 책임은 전부 여기 있습니다. 새 모델을 붙일 때는
이 파일에 매퍼 함수 하나만 추가하면 됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet

import numpy as np


@dataclass(frozen=True)
class Action:
    """프론트엔드가 보낸 원본 입력 한 틱."""
    seq: int = 0
    keys: FrozenSet[str] = frozenset()
    dx: float = 0.0          # 마지막 틱 이후 누적 마우스 상대 이동 (px)
    dy: float = 0.0
    left: bool = False       # 마우스 좌클릭
    right: bool = False
    hotbar: int = 1

    @staticmethod
    def from_json(msg: Dict[str, Any]) -> "Action":
        mouse = msg.get("mouse") or {}
        btn = msg.get("buttons") or {}
        return Action(
            seq=int(msg.get("seq", 0)),
            keys=frozenset(msg.get("keys") or []),
            dx=float(mouse.get("dx", 0.0)),
            dy=float(mouse.get("dy", 0.0)),
            left=bool(btn.get("left", False)),
            right=bool(btn.get("right", False)),
            hotbar=int(msg.get("hotbar", 1)),
        )

    def held(self, *codes: str) -> bool:
        return any(c in self.keys for c in codes)

    def consumed(self) -> "Action":
        """마우스 상대 이동은 한 번 쓰면 사라진다. 키는 눌린 채로 유지."""
        return Action(self.seq, self.keys, 0.0, 0.0, self.left, self.right, self.hotbar)


NEUTRAL = Action()


# ---------------------------------------------------------------------------
# Minecraft (Oasis) — VPT 스타일 이진 버튼 + 카메라 2축
# ---------------------------------------------------------------------------

def to_minecraft(a: Action, sensitivity: float = 0.15) -> Dict[str, Any]:
    """
    Oasis / VPT 계열이 기대하는 액션 딕셔너리.

    camera는 도(degree) 단위 [pitch, yaw]입니다. 브라우저의 픽셀 delta를
    sensitivity로 스케일하고 ±20도로 클램프합니다 (한 틱에 그 이상 도는
    입력은 학습 분포 밖이라 모델이 깨집니다).
    """
    return {
        "forward": int(a.held("KeyW", "ArrowUp")),
        "back":    int(a.held("KeyS", "ArrowDown")),
        "left":    int(a.held("KeyA", "ArrowLeft")),
        "right":   int(a.held("KeyD", "ArrowRight")),
        "jump":    int(a.held("Space")),
        "sneak":   int(a.held("ShiftLeft", "ShiftRight")),
        "sprint":  int(a.held("ControlLeft", "ControlRight")),
        "attack":  int(a.left),
        "use":     int(a.right),
        "drop":    int(a.held("KeyQ")),
        "inventory": int(a.held("KeyE")),
        "hotbar":  max(1, min(9, a.hotbar)),
        "camera":  [
            float(np.clip(a.dy * sensitivity, -20.0, 20.0)),   # pitch
            float(np.clip(a.dx * sensitivity, -20.0, 20.0)),   # yaw
        ],
    }


MINECRAFT_BUTTONS = [
    "forward", "back", "left", "right", "jump",
    "sneak", "sprint", "attack", "use", "drop", "inventory",
]


# ---------------------------------------------------------------------------
# Oasis(open-oasis) 정확한 액션 포맷
#
# open-oasis/utils.py 의 ACTION_KEYS 순서와 정확히 일치해야 합니다. 순서·개수가
# 어긋나면 모델이 엉뚱한 버튼을 눌린 것으로 해석해 화면이 무너집니다.
# 이 25차원 벡터가 generate.py 의 one_hot_actions() 출력과 동일한 규약입니다.
# ---------------------------------------------------------------------------

OASIS_ACTION_KEYS = [
    "inventory", "ESC",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right",
    "cameraX", "cameraY",
    "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop",
]


def oasis_vector(a: Action, sensitivity: float = 0.15) -> np.ndarray:
    """
    open-oasis 가 기대하는 25차원 액션 벡터. ACTION_KEYS 순서 그대로.

    카메라 인코딩은 utils.one_hot_actions() 을 그대로 따릅니다:
      - 브라우저 픽셀 delta → sensitivity 로 스케일 → ±20도 클램프
      - 정규화 카메라 값 = degrees / 20  (즉 [-1, 1])
    hotbar 는 선택된 슬롯 하나만 1인 원핫입니다.
    """
    cam_pitch = float(np.clip(a.dy * sensitivity, -20.0, 20.0)) / 20.0   # cameraX
    cam_yaw = float(np.clip(a.dx * sensitivity, -20.0, 20.0)) / 20.0     # cameraY
    slot = max(1, min(9, a.hotbar))

    values = {
        "inventory": float(a.held("KeyE")),
        "ESC": 0.0,
        "forward": float(a.held("KeyW", "ArrowUp")),
        "back": float(a.held("KeyS", "ArrowDown")),
        "left": float(a.held("KeyA", "ArrowLeft")),
        "right": float(a.held("KeyD", "ArrowRight")),
        "cameraX": cam_pitch,
        "cameraY": cam_yaw,
        "jump": float(a.held("Space")),
        "sneak": float(a.held("ShiftLeft", "ShiftRight")),
        "sprint": float(a.held("ControlLeft", "ControlRight")),
        "swapHands": float(a.held("KeyF")),
        "attack": float(a.left),
        "use": float(a.right),
        "pickItem": float(a.held("KeyX")),
        "drop": float(a.held("KeyQ")),
    }
    for n in range(1, 10):
        values[f"hotbar.{n}"] = 1.0 if slot == n else 0.0

    return np.asarray([values[k] for k in OASIS_ACTION_KEYS], dtype=np.float32)


# 하위호환용 별칭 (기존 13차원). 실제 Oasis 연결에는 oasis_vector 를 쓰세요.
def minecraft_vector(a: Action, sensitivity: float = 0.15) -> np.ndarray:
    """레거시 13차원 벡터. 실제 open-oasis 에는 oasis_vector(25차원)를 사용."""
    d = to_minecraft(a, sensitivity)
    buttons = [float(d[k]) for k in MINECRAFT_BUTTONS]
    return np.asarray(buttons + [d["camera"][0] / 20.0, d["camera"][1] / 20.0],
                      dtype=np.float32)


# ---------------------------------------------------------------------------
# CS:GO (DIAMOND) — 키 이진 + 마우스 이동 이산 bin
# ---------------------------------------------------------------------------

# DIAMOND CS:GO는 마우스 이동을 이산 bin으로 다룹니다. 원 논문 구현의
# bin 경계와 맞추세요 (csgo/action_processing.py 참조).
CSGO_MOUSE_BINS = np.array(
    [-1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2, -0.5,
     0.5, 2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000],
    dtype=np.float32,
)

CSGO_KEYS = [
    "KeyW", "KeyA", "KeyS", "KeyD", "Space",
    "ControlLeft", "ShiftLeft", "KeyR", "KeyE",
    "Digit1", "Digit2", "Digit3",
]


def _bin_index(value: float, bins: np.ndarray) -> int:
    return int(np.argmin(np.abs(bins - value)))


def to_csgo(a: Action) -> Dict[str, Any]:
    return {
        "keys": {k: int(k in a.keys) for k in CSGO_KEYS},
        "fire": int(a.left),
        "scope": int(a.right),
        "mouse_x_bin": _bin_index(a.dx, CSGO_MOUSE_BINS),
        "mouse_y_bin": _bin_index(a.dy, CSGO_MOUSE_BINS),
    }


def csgo_vector(a: Action) -> np.ndarray:
    """길이 = 키 12 + 발사/조준 2 + 마우스 원핫 2×24 = 62."""
    d = to_csgo(a)
    keys = [float(d["keys"][k]) for k in CSGO_KEYS]
    mx = np.zeros(len(CSGO_MOUSE_BINS), dtype=np.float32)
    my = np.zeros(len(CSGO_MOUSE_BINS), dtype=np.float32)
    mx[d["mouse_x_bin"]] = 1.0
    my[d["mouse_y_bin"]] = 1.0
    return np.concatenate(
        [np.asarray(keys + [float(d["fire"]), float(d["scope"])], dtype=np.float32), mx, my]
    )


# ---------------------------------------------------------------------------
# Atari — 단일 이산 액션 (ALE 표준 18개 중 필요한 것)
# ---------------------------------------------------------------------------

# NOOP FIRE UP RIGHT LEFT DOWN UPRIGHT UPLEFT DOWNRIGHT DOWNLEFT
# UPFIRE RIGHTFIRE LEFTFIRE DOWNFIRE UPRIGHTFIRE UPLEFTFIRE DOWNRIGHTFIRE DOWNLEFTFIRE
_ATARI_TABLE = {
    (0, 0, 0, 0, 0): 0,   (0, 0, 0, 0, 1): 1,
    (1, 0, 0, 0, 0): 2,   (0, 0, 0, 1, 0): 3,
    (0, 0, 1, 0, 0): 4,   (0, 1, 0, 0, 0): 5,
    (1, 0, 0, 1, 0): 6,   (1, 0, 1, 0, 0): 7,
    (0, 1, 0, 1, 0): 8,   (0, 1, 1, 0, 0): 9,
    (1, 0, 0, 0, 1): 10,  (0, 0, 0, 1, 1): 11,
    (0, 0, 1, 0, 1): 12,  (0, 1, 0, 0, 1): 13,
    (1, 0, 0, 1, 1): 14,  (1, 0, 1, 0, 1): 15,
    (0, 1, 0, 1, 1): 16,  (0, 1, 1, 0, 1): 17,
}


def to_atari(a: Action) -> int:
    up    = int(a.held("ArrowUp", "KeyW"))
    down  = int(a.held("ArrowDown", "KeyS"))
    left  = int(a.held("ArrowLeft", "KeyA"))
    right = int(a.held("ArrowRight", "KeyD"))
    fire  = int(a.held("Space") or a.left)
    if up and down:
        up = down = 0          # 상충 입력은 상쇄
    if left and right:
        left = right = 0
    return _ATARI_TABLE.get((up, down, left, right, fire), 0)


MAPPERS = {
    "oasis": oasis_vector,          # 25차원 VPT 포맷 (open-oasis 실제 규약)
    "diamond-csgo": csgo_vector,
    "diamond-atari": to_atari,
}