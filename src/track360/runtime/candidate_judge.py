"""A learned judge of search candidates (experimental, V2.1, evaluation log E049-E051).

The judge scores a candidate from one stateless pass of the backend on a normal-size
view centred on it: the product of P1 (gradient-boosted trees: is the target inside
that view) and Q (ridge regression: the IoU of the pass's box), each clipped to [0, 1].
Its inputs are nine scalars and the 768-value token the backend's score head reads,
in the order of ``tools/head_study_r4.py`` (``candidateData``).

The file is a NumPy archive without pickled objects, written by
``tools/candidate_judge_export.py``.  It holds one judge per cross-fitting fold with
the sequences it must be used for (it was trained without them), and one trained on
every sequence (``all``) for all other sequences.  The judges were trained on 360VOTS
annotations, which are licensed for non-commercial research only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from track360.core.errors import ModelError

TOKEN_SIZE = 768
SCALAR_COUNT = 9


@dataclass(frozen=True, slots=True)
class _Trees:
    """A binary gradient-boosted tree ensemble on numerical features."""

    baseline: float
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaf: np.ndarray
    value: np.ndarray
    roots: np.ndarray

    def probability(self, x: np.ndarray) -> np.ndarray:
        raw = np.full(len(x), self.baseline)
        for row in range(len(x)):
            total = 0.0
            for root in self.roots:
                node = int(root)
                while not self.leaf[node]:
                    goLeft = x[row, self.feature[node]] <= self.threshold[node]
                    node = int(self.left[node] if goLeft else self.right[node])
                total += self.value[node]
            raw[row] += total
        return 1.0 / (1.0 + np.exp(-raw))


@dataclass(frozen=True, slots=True)
class _Judge:
    sequences: frozenset[str]
    # Q: clip, standardise, ridge.
    qLow: np.ndarray
    qHigh: np.ndarray
    qMean: np.ndarray
    qScale: np.ndarray
    qCoef: np.ndarray
    qIntercept: float
    # P1: token standardised and reduced by PCA, then trees on scalars + components.
    pTokenMean: np.ndarray
    pTokenScale: np.ndarray
    pPcaMean: np.ndarray
    pPcaComponents: np.ndarray
    trees: _Trees

    def score(self, x: np.ndarray) -> np.ndarray:
        q = (np.clip(x, self.qLow, self.qHigh) - self.qMean) / self.qScale @ self.qCoef
        q = q + self.qIntercept
        token = (x[:, SCALAR_COUNT:] - self.pTokenMean) / self.pTokenScale
        components = (token - self.pPcaMean) @ self.pPcaComponents.T
        p = self.trees.probability(np.column_stack([x[:, :SCALAR_COUNT], components]))
        return np.clip(p, 0.0, 1.0) * np.clip(q, 0.0, 1.0)


class CandidateJudge:
    """Score search candidates; ``select`` picks the judge trained without a sequence."""

    def __init__(self, path: str | Path, threshold: float, prefilter: float = 0.0) -> None:
        self.path = Path(path)
        self.threshold = float(threshold)
        # Candidates whose search score is below this get no pass and are not taken.
        self.prefilter = float(prefilter)
        self._judges = _load(self.path)
        if "all" not in self._judges:
            raise ModelError(f"candidate judge {path} has no judge named 'all'")
        self._active = self._judges["all"]

    def select(self, sequence: str | None) -> str:
        label = next(
            (
                key
                for key, judge in self._judges.items()
                if key != "all" and sequence in judge.sequences
            ),
            "all",
        )
        self._active = self._judges[label]
        return label

    def score(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if x.shape[1] != SCALAR_COUNT + TOKEN_SIZE:
            raise ModelError(f"candidate judge expects {SCALAR_COUNT + TOKEN_SIZE} features")
        return self._active.score(x)

    def describe(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "threshold": self.threshold,
            "prefilter": self.prefilter,
            "judges": sorted(self._judges),
        }


def judgeFeatures(
    *,
    searchScore: float,
    recentredScore: float,
    similarity: float | None,
    scan: str,
    diagnostics: Mapping[str, Any],
) -> np.ndarray:
    """The judge's inputs for one candidate, as the training data stored them.

    The training archives kept the scalars in float32 and the token in float16; the
    same rounding is applied here so that a pass gets the score it had offline.
    """
    entropy = np.asarray(diagnostics["coordEntropy"], dtype=np.float32)
    peak = np.asarray(diagnostics["coordPeak"], dtype=np.float32)
    gap = np.asarray(diagnostics["hardSoftGap"], dtype=np.float32)
    token = np.asarray(diagnostics["scoreFeature"]).astype(np.float16).astype(np.float64)
    scalars = np.asarray(
        [
            np.float32(round(float(searchScore), 4)),
            np.float32(recentredScore),
            np.float32(round(float(similarity or 0.0), 4)),
            float(scan == "1x"),
            float(scan == "4x"),
            entropy.mean(),
            entropy.max(),
            peak.min(),
            gap.max(),
        ],
        dtype=np.float64,
    )
    return np.concatenate([scalars, token.reshape(-1)])


def writeJudges(path: str | Path, judges: Mapping[str, Mapping[str, Any]]) -> None:
    """Write judges given as plain arrays (see ``tools/candidate_judge_export.py``)."""
    arrays: dict[str, np.ndarray] = {}
    for label, parts in judges.items():
        for key, value in parts.items():
            arrays[f"{label}__{key}"] = np.asarray(value)
    np.savez_compressed(Path(path), **arrays)


def _load(path: Path) -> dict[str, _Judge]:
    data = np.load(path, allow_pickle=False)
    labels = sorted({key.split("__", 1)[0] for key in data.files})
    judges = {}
    for label in labels:

        def get(key: str, label: str = label) -> np.ndarray:
            return data[f"{label}__{key}"]

        trees = _Trees(
            baseline=float(get("treeBaseline")),
            feature=get("treeFeature").astype(np.int64),
            threshold=get("treeThreshold").astype(np.float64),
            left=get("treeLeft").astype(np.int64),
            right=get("treeRight").astype(np.int64),
            leaf=get("treeLeaf").astype(bool),
            value=get("treeValue").astype(np.float64),
            roots=get("treeRoots").astype(np.int64),
        )
        judges[label] = _Judge(
            sequences=frozenset(str(name) for name in get("sequences")),
            qLow=get("qLow"),
            qHigh=get("qHigh"),
            qMean=get("qMean"),
            qScale=get("qScale"),
            qCoef=get("qCoef"),
            qIntercept=float(get("qIntercept")),
            pTokenMean=get("pTokenMean"),
            pTokenScale=get("pTokenScale"),
            pPcaMean=get("pPcaMean"),
            pPcaComponents=get("pPcaComponents"),
            trees=trees,
        )
    return judges


def treeArrays(predictors: Sequence[Any], baseline: float) -> dict[str, np.ndarray]:
    """Flatten scikit-learn's tree predictors (one per boosting round) into arrays."""
    feature, threshold, left, right, leaf, value, roots = [], [], [], [], [], [], []
    offset = 0
    for predictor in predictors:
        nodes = predictor.nodes
        roots.append(offset)
        feature.append(nodes["feature_idx"])
        threshold.append(nodes["num_threshold"])
        left.append(nodes["left"].astype(np.int64) + offset)
        right.append(nodes["right"].astype(np.int64) + offset)
        leaf.append(nodes["is_leaf"])
        value.append(nodes["value"])
        offset += len(nodes)
    return {
        "treeBaseline": np.asarray(baseline),
        "treeFeature": np.concatenate(feature),
        "treeThreshold": np.concatenate(threshold),
        "treeLeft": np.concatenate(left),
        "treeRight": np.concatenate(right),
        "treeLeaf": np.concatenate(leaf),
        "treeValue": np.concatenate(value),
        "treeRoots": np.asarray(roots),
    }
