"""Compare two trackers' BBox scores on the sequences both have results for."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from track360.core.errors import ProtocolError
from track360.evaluation.bootstrap import (
    DEFAULT_SAMPLES,
    Interval,
    bootstrapDifference,
    bootstrapRatio,
)
from track360.evaluation.vot360_metrics import Vot360Scores

# A hard-regression sequence fails when its S_dual drops by more than this.  The
# sequences are chosen to move less than this between unrelated configurations.
HARD_REGRESSION_TOLERANCE = 0.03
# Per-sequence changes smaller than this are not counted as up or down.
CHANGE_THRESHOLD = 0.02


@dataclass(frozen=True, slots=True)
class MetricComparison:
    name: str
    baseline: Interval
    candidate: Interval
    difference: Interval


@dataclass(frozen=True, slots=True)
class SequenceChange:
    sequence: str
    baseline: float
    candidate: float

    @property
    def difference(self) -> float:
        return self.candidate - self.baseline


@dataclass(frozen=True, slots=True)
class Comparison:
    sequences: tuple[str, ...]
    metrics: tuple[MetricComparison, ...]
    changes: tuple[SequenceChange, ...]  # S_dual per sequence, largest drop first

    def metric(self, name: str) -> MetricComparison:
        return next(item for item in self.metrics if item.name == name)


def compareScores(
    baseline: Vot360Scores,
    candidate: Vot360Scores,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> Comparison:
    """Paired bootstrap of S_dual, P_angle and loss rate over the common sequences."""
    if baseline.representation != "bbox" or candidate.representation != "bbox":
        raise ProtocolError("comparison needs BBox scores of both trackers")
    names = tuple(sorted(set(baseline.perSequence) & set(candidate.perSequence)))
    if not names:
        raise ProtocolError("the two trackers share no sequence")
    base = [baseline.perSequence[name] for name in names]
    cand = [candidate.perSequence[name] for name in names]

    def compare(name: str, baseTop, candTop, baseBottom=None, candBottom=None):
        return MetricComparison(
            name=name,
            baseline=bootstrapRatio(baseTop, baseBottom, samples=samples, seed=seed),
            candidate=bootstrapRatio(candTop, candBottom, samples=samples, seed=seed),
            difference=bootstrapDifference(
                baseTop, candTop, baseBottom, candBottom, samples=samples, seed=seed
            ),
        )

    metrics = (
        compare("S_dual", [item.success for item in base], [item.success for item in cand]),
        compare(
            "P_angle",
            [item.anglePrecision for item in base],
            [item.anglePrecision for item in cand],
        ),
        compare(
            "loss_rate",
            [item.lostFrames or 0 for item in base],
            [item.lostFrames or 0 for item in cand],
            [item.frames for item in base],
            [item.frames for item in cand],
        ),
    )
    changes = sorted(
        (
            SequenceChange(name, first.success, second.success)
            for name, first, second in zip(names, base, cand, strict=True)
        ),
        key=lambda change: change.difference,
    )
    return Comparison(sequences=names, metrics=metrics, changes=tuple(changes))


def hardRegressions(
    comparison: Comparison,
    hardSequences: Sequence[str],
    tolerance: float = HARD_REGRESSION_TOLERANCE,
) -> tuple[SequenceChange, ...]:
    """Return the hard-regression sequences whose S_dual dropped by more than ``tolerance``.

    Every hard sequence must have results from both trackers: a missing one would
    otherwise pass silently.
    """
    byName = {change.sequence: change for change in comparison.changes}
    missing = [name for name in hardSequences if name not in byName]
    if missing:
        raise ProtocolError(
            f"hard-regression sequences without results from both trackers: {', '.join(missing)}"
        )
    return tuple(
        byName[name] for name in hardSequences if byName[name].difference < -tolerance
    )


@dataclass(frozen=True, slots=True)
class GroupChange:
    """Mean S_dual of a group of sequences in both runs, and the paired difference."""

    sequences: tuple[str, ...]
    baseline: float
    candidate: float
    difference: Interval

    @property
    def regressed(self) -> bool:
        """The group got worse beyond what resampling its sequences explains."""
        return self.difference.high < 0.0


def groupChange(
    comparison: Comparison,
    sequences: Sequence[str],
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> GroupChange:
    """Judge sequences that are too unstable to be judged one by one.

    The score of a fragile sequence swings with changes that have nothing to do with
    it, so only the group mean and the bootstrap interval of its paired difference
    carry information.  Every sequence must have results from both trackers.
    """
    byName = {change.sequence: change for change in comparison.changes}
    missing = [name for name in sequences if name not in byName]
    if missing:
        raise ProtocolError(
            f"fragile sequences without results from both trackers: {', '.join(missing)}"
        )
    if not sequences:
        raise ProtocolError("the fragile group is empty")
    base = [byName[name].baseline for name in sequences]
    cand = [byName[name].candidate for name in sequences]
    return GroupChange(
        sequences=tuple(sequences),
        baseline=sum(base) / len(base),
        candidate=sum(cand) / len(cand),
        difference=bootstrapDifference(base, cand, samples=samples, seed=seed),
    )


__all__ = [
    "CHANGE_THRESHOLD",
    "GroupChange",
    "groupChange",
    "HARD_REGRESSION_TOLERANCE",
    "Comparison",
    "MetricComparison",
    "SequenceChange",
    "compareScores",
    "hardRegressions",
]
