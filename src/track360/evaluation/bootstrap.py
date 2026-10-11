"""Bootstrap confidence intervals over sequences.

Benchmark scores are averages over a few dozen sequences whose individual scores vary
a lot, so a difference between two methods can come from which sequences happen to be
in the set.  Resampling the sequences with replacement shows how much the score moves.

Every statistic here is a ratio of sums over sequences, ``sum(numerator) /
sum(denominator)``: with a denominator of one per sequence that is the plain mean
(S_dual, P_angle), with lost frames over frames it is the loss rate.  Two methods are
compared on the same resampled sequences (a paired bootstrap), which removes the
variation both methods share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_SAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class Interval:
    """A statistic with its bootstrap confidence interval."""

    value: float
    low: float
    high: float

    @property
    def excludesZero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def bootstrapRatio(
    numerators: ArrayLike,
    denominators: ArrayLike | None = None,
    *,
    samples: int = DEFAULT_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Interval of ``sum(numerators) / sum(denominators)``; no denominators: the mean."""
    top, bottom = _pair(numerators, denominators)
    draws = _draws(len(top), samples, seed)
    return _interval(
        _ratio(top.sum(), bottom.sum()),
        _ratios(top, bottom, draws),
        confidence,
    )


def bootstrapDifference(
    baselineNumerators: ArrayLike,
    candidateNumerators: ArrayLike,
    baselineDenominators: ArrayLike | None = None,
    candidateDenominators: ArrayLike | None = None,
    *,
    samples: int = DEFAULT_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Interval of candidate minus baseline, both given per sequence in the same order."""
    baseTop, baseBottom = _pair(baselineNumerators, baselineDenominators)
    candTop, candBottom = _pair(candidateNumerators, candidateDenominators)
    if len(baseTop) != len(candTop):
        raise ValueError("paired bootstrap needs the same sequences for both methods")
    draws = _draws(len(baseTop), samples, seed)
    return _interval(
        _ratio(candTop.sum(), candBottom.sum()) - _ratio(baseTop.sum(), baseBottom.sum()),
        _ratios(candTop, candBottom, draws) - _ratios(baseTop, baseBottom, draws),
        confidence,
    )


def _pair(
    numerators: ArrayLike, denominators: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    top = np.asarray(numerators, dtype=np.float64)
    if top.ndim != 1 or top.size == 0:
        raise ValueError("bootstrap needs a non-empty list of per-sequence values")
    bottom = np.ones_like(top) if denominators is None else np.asarray(denominators, np.float64)
    if bottom.shape != top.shape:
        raise ValueError("bootstrap numerators and denominators must align")
    return top, bottom


def _draws(count: int, samples: int, seed: int) -> NDArray[np.int64]:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    return np.random.default_rng(seed).integers(0, count, size=(samples, count))


def _ratios(
    top: NDArray[np.float64], bottom: NDArray[np.float64], draws: NDArray[np.int64]
) -> NDArray[np.float64]:
    totals = bottom[draws].sum(axis=1)
    return np.divide(top[draws].sum(axis=1), totals, out=np.zeros(len(draws)), where=totals > 0)


def _ratio(top: float, bottom: float) -> float:
    return float(top / bottom) if bottom > 0 else 0.0


def _interval(value: float, resampled: NDArray[np.float64], confidence: float) -> Interval:
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(resampled, [tail, 1.0 - tail])
    return Interval(value=float(value), low=float(low), high=float(high))


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_SAMPLES",
    "Interval",
    "bootstrapDifference",
    "bootstrapRatio",
]
