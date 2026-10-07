import numpy as np

from workers.adapters.dummy import make_dummy
from workers.common.actions import Action
from workers.common.frames import to_uint8_rgb


def test_dummy_model_through_public_api():
    model = make_dummy("oasis")
    first = model.reset()
    second = model.step(Action(keys=frozenset({"KeyW"})))
    assert first.shape == second.shape == (180, 320, 3)
    assert not np.array_equal(first, second)


def test_float_chw_frame_normalization():
    frame = np.zeros((3, 4, 5), dtype=np.float32)
    assert to_uint8_rgb(frame).shape == (4, 5, 3)
