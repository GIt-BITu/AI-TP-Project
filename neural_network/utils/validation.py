"""Shared validation helpers used by the neural-network components."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence


def positive_int(value: int, name: str) -> int:
    """Return *value* if it is a positive integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def finite_float(value: Real, name: str) -> float:
    """Convert a real number to float and reject NaN/inf and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return result


def numeric_vector(values: Sequence[Real], expected_size: int, name: str) -> list[float]:
    """Validate and copy a one-dimensional numeric vector."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence, not text.")
    try:
        actual_size = len(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of numbers.") from exc

    if actual_size != expected_size:
        raise ValueError(
            f"{name} must contain exactly {expected_size} values; got {actual_size}."
        )
    return [finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)]
