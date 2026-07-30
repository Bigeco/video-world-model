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

# DIAMOND CS:GO는 마우스 이동을 이산 bin으로 다룹니다. 원 리포
# src/csgo/action_processing.py 의 encode_csgo_action() 과 정확히 같은 bin 경계·키
# 순서·차원 수(51)여야 합니다. 하나라도 어긋나면 모델이 엉뚱한 입력으로 해석해
# 화면이 무너집니다 — X/Y 축의 bin 개수가 다르고(23 vs 15), 0을 포함한다는 점에 주의.
CSGO_MOUSE_X_BINS = np.array(
    [-1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2, 0,
     2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000],
    dtype=np.float32,
)
CSGO_MOUSE_Y_BINS = np.array(
    [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200],
    dtype=np.float32,
)

# encode_csgo_action() 의 keys_pressed_onehot 순서 그대로: w,a,s,d,space,ctrl,shift,1,2,3,r
CSGO_KEYS = [
    "KeyW", "KeyA", "KeyS", "KeyD", "Space",
    "ControlLeft", "ShiftLeft", "Digit1", "Digit2", "Digit3", "KeyR",
]


def _bin_index(value: float, bins: np.ndarray) -> int:
    return int(np.argmin(np.abs(bins - value)))


def to_csgo(a: Action) -> Dict[str, Any]:
    return {
        "keys": {k: int(k in a.keys) for k in CSGO_KEYS},
        "fire": int(a.left),
        "scope": int(a.right),
        "mouse_x_bin": _bin_index(a.dx, CSGO_MOUSE_X_BINS),
        "mouse_y_bin": _bin_index(a.dy, CSGO_MOUSE_Y_BINS),
    }


def csgo_vector(a: Action) -> np.ndarray:
    """길이 = 키 11 + 발사(l_click) 1 + 조준(r_click) 1 + 마우스 원핫(23 + 15) = 51.
    encode_csgo_action()과 동일한 순서: keys, l_click, r_click, mouse_x, mouse_y.
    """
    d = to_csgo(a)
    keys = [float(d["keys"][k]) for k in CSGO_KEYS]
    mx = np.zeros(len(CSGO_MOUSE_X_BINS), dtype=np.float32)
    my = np.zeros(len(CSGO_MOUSE_Y_BINS), dtype=np.float32)
    mx[d["mouse_x_bin"]] = 1.0
    my[d["mouse_y_bin"]] = 1.0
    return np.concatenate(
        [np.asarray(keys + [float(d["fire"]), float(d["scope"])], dtype=np.float32), mx, my]
    )


# ---------------------------------------------------------------------------
# Atari — 단일 이산 액션. DIAMOND의 atari_100k 체크포인트 26종은 게임마다 서로 다른
# "축소 액션셋"(ALE의 get_action_meanings(), full_action_space=False)으로 학습돼
# 있다. 예를 들어 Breakout은 [NOOP, FIRE, RIGHT, LEFT] 4개뿐이고, Alien은 ALE 표준
# 18개를 전부 쓴다. 체크포인트가 기대하는 인덱스와 다른 인덱스를 넣으면 엉뚱한
# 버튼을 누른 것으로 해석돼 화면이 무너지므로, 게임별로 정확한 인덱스를 찾아야 한다.
#
# 각 게임의 목록은 축소판이어도 항상 ALE 표준 18개 순서의 부분열이다(값 자체가
# 바뀌는 게 아니라 없는 것만 빠짐 — NOOP은 모든 게임에서 항상 인덱스 0). 그래서
# to_atari()는 먼저 키 입력을 표준 18개 중 하나의 "이름"으로 바꾼 다음, 그 이름을
# 해당 게임의 목록에서 찾아 인덱스로 변환한다. 게임에 없는 이름이면(예: Breakout에서
# 위쪽 방향키) NOOP으로 떨어진다.
#
# 아래 표는 추측이 아니라 실제로 ale-py==0.9.0(gymnasium, full_action_space=False)을
# 설치해 26개 게임 전부 env.unwrapped.get_action_meanings()를 직접 호출해 뽑은 값이다.
# ---------------------------------------------------------------------------

ATARI_GAMES = [
    "Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing",
    "Breakout", "ChopperCommand", "CrazyClimber", "DemonAttack", "Freeway",
    "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster",
    "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown",
]

# ALE 표준 18액션 전체 (모든 게임의 목록은 이 순서를 유지하는 부분열).
_ATARI_FULL_18 = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]

# 게임별 실제(축소) 액션 목록. env.unwrapped.get_action_meanings()와 동일.
ATARI_GAME_ACTIONS = {
    "Alien": _ATARI_FULL_18,
    "Amidar": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE"],
    "Assault": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
    "Asterix": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"],
    "BankHeist": _ATARI_FULL_18,
    "BattleZone": _ATARI_FULL_18,
    "Boxing": _ATARI_FULL_18,
    "Breakout": ["NOOP", "FIRE", "RIGHT", "LEFT"],
    "ChopperCommand": _ATARI_FULL_18,
    "CrazyClimber": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"],
    "DemonAttack": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
    "Freeway": ["NOOP", "UP", "DOWN"],
    "Frostbite": _ATARI_FULL_18,
    "Gopher": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE"],
    "Hero": _ATARI_FULL_18,
    "Jamesbond": _ATARI_FULL_18,
    "Kangaroo": _ATARI_FULL_18,
    "Krull": _ATARI_FULL_18,
    "KungFuMaster": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "DOWNRIGHT", "DOWNLEFT",
                      "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE",
                      "DOWNRIGHTFIRE", "DOWNLEFTFIRE"],
    "MsPacman": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"],
    "Pong": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
    "PrivateEye": _ATARI_FULL_18,
    "Qbert": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"],
    "RoadRunner": _ATARI_FULL_18,
    "Seaquest": _ATARI_FULL_18,
    "UpNDown": ["NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE"],
}

# 입력 조합(up,down,left,right,fire) → ALE 표준 액션 이름. 예전 _ATARI_TABLE(콤보→
# 18개 중 인덱스)과 같은 조합이지만, 여기선 이름으로 남겨뒀다가 게임별 목록에서
# 실제 인덱스를 다시 찾는다.
_ATARI_COMBO_TO_NAME = {
    (0, 0, 0, 0, 0): "NOOP",           (0, 0, 0, 0, 1): "FIRE",
    (1, 0, 0, 0, 0): "UP",             (0, 0, 0, 1, 0): "RIGHT",
    (0, 0, 1, 0, 0): "LEFT",           (0, 1, 0, 0, 0): "DOWN",
    (1, 0, 0, 1, 0): "UPRIGHT",        (1, 0, 1, 0, 0): "UPLEFT",
    (0, 1, 0, 1, 0): "DOWNRIGHT",      (0, 1, 1, 0, 0): "DOWNLEFT",
    (1, 0, 0, 0, 1): "UPFIRE",         (0, 0, 0, 1, 1): "RIGHTFIRE",
    (0, 0, 1, 0, 1): "LEFTFIRE",       (0, 1, 0, 0, 1): "DOWNFIRE",
    (1, 0, 0, 1, 1): "UPRIGHTFIRE",    (1, 0, 1, 0, 1): "UPLEFTFIRE",
    (0, 1, 0, 1, 1): "DOWNRIGHTFIRE",  (0, 1, 1, 0, 1): "DOWNLEFTFIRE",
}


def to_atari(a: Action, game: str = "Alien") -> int:
    """`game`(ATARI_GAMES 중 하나, 대소문자 그대로)이 실제로 학습된 축소 액션셋에서의
    인덱스를 돌려준다. 그 게임에 없는 동작이면 NOOP(항상 인덱스 0)으로 떨어진다."""
    up    = int(a.held("ArrowUp", "KeyW"))
    down  = int(a.held("ArrowDown", "KeyS"))
    left  = int(a.held("ArrowLeft", "KeyA"))
    right = int(a.held("ArrowRight", "KeyD"))
    fire  = int(a.held("Space") or a.left)
    if up and down:
        up = down = 0          # 상충 입력은 상쇄
    if left and right:
        left = right = 0
    name = _ATARI_COMBO_TO_NAME.get((up, down, left, right, fire), "NOOP")
    actions = ATARI_GAME_ACTIONS.get(game, _ATARI_FULL_18)
    return actions.index(name) if name in actions else 0


# ---------------------------------------------------------------------------
# LongLive — 게임형 액션이 아니라 "지금 재생 중인 프롬프트" 선택
#
# LongLive는 WASD로 조작하는 월드모델이 아니라 텍스트 프롬프트로 다음 장면을 계속
# 지시하는 롱비디오 생성기다. hotbar 슬롯(1~9)을 프롬프트 목록(WM_LONGLIVE_PROMPTS,
# '|' 구분) 인덱스로 재활용한다.
# ---------------------------------------------------------------------------

def longlive_prompt_index(a: Action, num_prompts: int) -> int:
    """hotbar 1~9 → 프롬프트 인덱스(0-base). 목록보다 큰 슬롯은 마지막 프롬프트로 클램프."""
    return max(1, min(max(1, num_prompts), a.hotbar)) - 1


MAPPERS = {
    "oasis": oasis_vector,          # 25차원 VPT 포맷 (open-oasis 실제 규약)
    "diamond-csgo": csgo_vector,
    "diamond-atari": to_atari,
    "longlive": longlive_prompt_index,
}