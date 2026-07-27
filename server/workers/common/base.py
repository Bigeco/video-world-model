"""워커가 구현해야 하는 인터페이스.

새 world model을 붙이려면 이 두 메서드만 채우면 됩니다.
서버 루프, 인코딩, 프레임 패킹, 페이싱은 전부 공통 런타임이 처리합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .actions import Action


class WorldModel(ABC):
    """
    구현 규약:

      * reset()과 step()은 **블로킹**이어도 됩니다. 공통 런타임이 별도
        스레드에서 호출하므로 이벤트 루프를 막지 않습니다.
      * 반환 프레임은 HWC/CHW, uint8/float 아무거나 괜찮습니다.
        encode.to_uint8_rgb가 정규화합니다.
      * 모델의 프레임 히스토리(context buffer) 관리는 구현체 책임입니다.
        대부분의 world model은 최근 N 프레임 + N 액션을 조건으로 받습니다.
    """

    #: 목표 생성 FPS. 추론이 더 느리면 그냥 나오는 대로 보냅니다.
    fps: int = 20

    #: JPEG 품질 (1-100). 낮출수록 대역폭이 줄고 지연이 조금 좋아집니다.
    quality: int = 80

    @abstractmethod
    def reset(self) -> np.ndarray:
        """에피소드를 초기화하고 첫 프레임을 반환."""

    @abstractmethod
    def step(self, action: Action) -> np.ndarray:
        """액션 한 틱을 적용하고 다음 프레임을 반환."""

    def close(self) -> None:
        """GPU 메모리 해제 등. 선택 구현."""

    def info(self) -> dict[str, Any]:
        return {"fps": self.fps, "quality": self.quality}
