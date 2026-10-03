"""Candidate scoring, clustering and single-frame aggregation for DTC."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, log, pi
from typing import TYPE_CHECKING

import numpy as np

from track360.core.config import DecisionGateConfig, TrackingConfig
from track360.core.errors import ProtocolError
from track360.core.protocols import SphericalGeometry
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    ProjectedObservation,
    SphericalPoint,
)
from track360.geometry.projection_math import makeSphericalPoint

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ScoredObservation:
    observation: ProjectedObservation
    decisionScore: float


@dataclass(frozen=True, slots=True)
class FrameAggregate:
    """The best geometrically consistent cluster in one frame."""

    bfov: BFoV
    bbox: BBoxXYWH
    confidence: float
    decisionScore: float
    sourceViewIds: tuple[int, ...]
    representativeViewId: int
    localBox: BBoxXYWH | None
    supported: bool
    clusterCount: int = 1
    agreementScore: float = 1.0


class DecisionGate:
    """Apply lightweight control weights without reimplementing backend fusion."""

    def __init__(self, gateConfig: DecisionGateConfig, trackingConfig: TrackingConfig) -> None:
        self._gateConfig = gateConfig
        self._trackingConfig = trackingConfig

    def score(self, observation: ProjectedObservation) -> ScoredObservation:
        weights = [
            max(
                0.0,
                1.0
                - self._gateConfig.motionScoreWeight
                - self._gateConfig.scaleScoreWeight,
            ),
            self._gateConfig.motionScoreWeight,
            self._gateConfig.scaleScoreWeight,
        ]
        values = [observation.fusedScore, observation.motionScore, observation.scaleScore]
        totalWeight = sum(weights)
        if totalWeight <= 0.0:
            raise ProtocolError("decision gate must have a positive effective weight")
        decisionScore = float(np.clip(np.dot(weights, values) / totalWeight, 0.0, 1.0))
        return ScoredObservation(
            observation=observation,
            decisionScore=decisionScore,
        )

    def aggregate(
        self,
        observations: Sequence[ProjectedObservation],
        geometry: SphericalGeometry,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> FrameAggregate | None:
        scored = [self.score(observation) for observation in observations]
        eligible = [
            item for item in scored if item.decisionScore >= self._trackingConfig.candidateMinScore
        ]
        if not eligible:
            return None

        clusters = _connectedClusters(eligible, self._trackingConfig.scaleClusterTolerance)

        best = max(clusters, key=lambda cluster: _clusterRank(cluster))
        return _aggregateCluster(
            best,
            geometry,
            frameWidthPx,
            frameHeightPx,
            self._trackingConfig.minViewsForCommit,
            len(clusters),
        )


def _compatible(
    candidate: ScoredObservation,
    reference: ScoredObservation,
    scaleTolerance: float,
) -> bool:
    angle = _centerAngle(candidate.observation.bfov.center, reference.observation.bfov.center)
    angleLimit = max(
        0.25,
        min(
            pi * 0.95,
            max(
                candidate.observation.bfov.horizontalFovRad,
                reference.observation.bfov.horizontalFovRad,
            )
            * 0.55,
        ),
    )
    candidateArea = candidate.observation.bbox.widthPx * candidate.observation.bbox.heightPx
    referenceArea = reference.observation.bbox.widthPx * reference.observation.bbox.heightPx
    scaleDelta = abs(log(max(candidateArea, 1e-6) / max(referenceArea, 1e-6)))
    return angle <= angleLimit and scaleDelta <= scaleTolerance


def _clusterRank(cluster: list[ScoredObservation]) -> tuple[float, int, float]:
    weight = sum(item.decisionScore for item in cluster)
    peak = max(item.decisionScore for item in cluster)
    return weight, len({item.observation.viewId for item in cluster}), peak


def _connectedClusters(
    candidates: list[ScoredObservation],
    scaleTolerance: float,
) -> list[list[ScoredObservation]]:
    """Build order-independent connected components in candidate compatibility space."""
    remaining = set(range(len(candidates)))
    clusters: list[list[ScoredObservation]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            linked = {
                index
                for index in remaining
                if _compatible(candidates[current], candidates[index], scaleTolerance)
            }
            remaining.difference_update(linked)
            component.update(linked)
            frontier.extend(sorted(linked))
        clusters.append([candidates[index] for index in sorted(component)])
    return clusters


def _aggregateCluster(
    cluster: list[ScoredObservation],
    geometry: SphericalGeometry,
    frameWidthPx: int,
    frameHeightPx: int,
    minViewsForCommit: int,
    clusterCount: int,
) -> FrameAggregate:
    weights = np.asarray([max(item.decisionScore, 1e-6) for item in cluster], dtype=np.float64)
    weights /= float(np.sum(weights))
    vectors = np.asarray(
        [
            (
                item.observation.bfov.center.x,
                item.observation.bfov.center.y,
                item.observation.bfov.center.z,
            )
            for item in cluster
        ],
        dtype=np.float64,
    )
    centerVector = _normalize(np.sum(vectors * weights[:, np.newaxis], axis=0))
    yawRad, pitchRad = _vectorToYawPitch(centerVector)
    horizontalFov = _weightedMedian(
        np.asarray([item.observation.bfov.horizontalFovRad for item in cluster]), weights
    )
    verticalFov = _weightedMedian(
        np.asarray([item.observation.bfov.verticalFovRad for item in cluster]), weights
    )
    fusedBfov = BFoV(
        center=makeSphericalPoint(yawRad, pitchRad),
        horizontalFovRad=min(pi - 1e-5, max(1e-4, horizontalFov)),
        verticalFovRad=min(pi - 1e-5, max(1e-4, verticalFov)),
    )
    bbox = geometry.bfovToBbox(fusedBfov, frameWidthPx, frameHeightPx)
    confidence = float(np.clip(np.dot(weights, [item.decisionScore for item in cluster]), 0.0, 1.0))
    representative = max(cluster, key=lambda item: item.decisionScore).observation
    viewIds = tuple(sorted({item.observation.viewId for item in cluster}))
    agreementScore = _clusterAgreement(cluster, weights, fusedBfov)
    return FrameAggregate(
        bfov=fusedBfov,
        bbox=bbox,
        confidence=confidence,
        decisionScore=confidence,
        sourceViewIds=viewIds,
        representativeViewId=representative.viewId,
        localBox=representative.localBox,
        supported=len(viewIds) >= minViewsForCommit,
        clusterCount=clusterCount,
        agreementScore=agreementScore,
    )


def _clusterAgreement(
    cluster: list[ScoredObservation],
    weights: np.ndarray,
    fusedBfov: BFoV,
) -> float:
    if len(cluster) <= 1:
        return 1.0
    angularResidual = sum(
        float(weight) * _centerAngle(item.observation.bfov.center, fusedBfov.center)
        for item, weight in zip(cluster, weights, strict=True)
    )
    angleScale = max(
        0.05,
        0.5 * max(fusedBfov.horizontalFovRad, fusedBfov.verticalFovRad),
    )
    scaleResidual = sum(
        float(weight)
        * 0.5
        * (
            abs(log(item.observation.bfov.horizontalFovRad / fusedBfov.horizontalFovRad))
            + abs(log(item.observation.bfov.verticalFovRad / fusedBfov.verticalFovRad))
        )
        for item, weight in zip(cluster, weights, strict=True)
    )
    return float(np.clip(np.exp(-angularResidual / angleScale - scaleResidual), 0.0, 1.0))


def _weightedMedian(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sortedValues = values[order]
    sortedWeights = weights[order]
    index = int(np.searchsorted(np.cumsum(sortedWeights), 0.5, side="left"))
    return float(sortedValues[min(index, len(sortedValues) - 1)])


def _centerAngle(first: SphericalPoint, second: SphericalPoint) -> float:
    dot = np.clip(first.x * second.x + first.y * second.y + first.z * second.z, -1.0, 1.0)
    return float(acos(float(dot)))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ProtocolError("candidate center vectors must not cancel out")
    return vector / norm


def _vectorToYawPitch(vector: np.ndarray) -> tuple[float, float]:
    return float(np.arctan2(vector[0], vector[2])), float(np.arcsin(np.clip(vector[1], -1.0, 1.0)))


__all__ = ["DecisionGate", "FrameAggregate", "ScoredObservation"]
