from __future__ import annotations

import math


def multiple(x: int, /, value: int) -> int:
    return math.ceil(x / value) * value
