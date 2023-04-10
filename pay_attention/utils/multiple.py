from __future__ import annotations

import math


def multiple(x: int, /, value: int) -> int:
    """Rounds up the input integer to the nearest multiple of a given value."""

    return math.ceil(x / value) * value
