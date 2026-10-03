"""Generate a semantic-grouped instance ID document from segmentation frames."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from track360.core.errors import ProtocolError
from track360.core.types import FramePacket


@dataclass(frozen=True, slots=True)
class InstanceIdGroup:
    """All unique instance IDs assigned to one semantic class."""

    semanticId: int | None
    semanticName: str
    instanceIds: tuple[int, ...]


def collectInstanceIdGroups(frame: FramePacket) -> tuple[InstanceIdGroup, ...]:
    """Group the first frame's non-background instance IDs by semantic class."""
    if int(frame.frameIndex) != 0:
        raise ProtocolError(
            f"instance ID document requires frameIndex 0, actual={frame.frameIndex}"
        )
    segmentation = frame.segmentation
    if segmentation is None or segmentation.instance is None:
        raise ProtocolError("frame 0 contains no instance segmentation mask")

    classNames = dict(segmentation.classNames)
    countsByInstance: dict[int, dict[int | None, int]] = {}
    instance = segmentation.instance
    semantic = segmentation.semantic
    if semantic is None:
        instanceIds, pixelCounts = np.unique(instance, return_counts=True)
        pairs = zip(instanceIds, (None for _ in instanceIds), pixelCounts, strict=True)
    else:
        pairValues, pixelCounts = np.unique(
            np.stack((instance.ravel(), semantic.ravel()), axis=1),
            axis=0,
            return_counts=True,
        )
        pairs = zip(pairValues[:, 0], pairValues[:, 1], pixelCounts, strict=True)

    for rawInstanceId, rawSemanticId, rawCount in pairs:
        instanceId = int(rawInstanceId)
        if instanceId == 0:
            continue
        semanticId = int(rawSemanticId) if rawSemanticId is not None else None
        semanticCounts = countsByInstance.setdefault(instanceId, {})
        semanticCounts[semanticId] = semanticCounts.get(semanticId, 0) + int(rawCount)

    grouped: dict[int | None, list[int]] = {}
    for instanceId, semanticCounts in countsByInstance.items():
        semanticId = max(
            semanticCounts,
            key=lambda value: (
                semanticCounts[value],
                value is not None,
                -(value or 0),
            ),
        )
        grouped.setdefault(semanticId, []).append(instanceId)

    classOrder = {semanticId: index for index, semanticId in enumerate(classNames)}
    orderedSemanticIds = sorted(
        grouped,
        key=lambda semanticId: (
            semanticId not in classOrder,
            classOrder.get(semanticId, 0),
            semanticId is None,
            semanticId if semanticId is not None else 0,
        ),
    )
    return tuple(
        InstanceIdGroup(
            semanticId=semanticId,
            semanticName=(
                classNames.get(semanticId, f"semantic_{semanticId}")
                if semanticId is not None
                else "unknown"
            ),
            instanceIds=tuple(sorted(grouped[semanticId])),
        )
        for semanticId in orderedSemanticIds
    )


def formatInstanceIdDocument(groups: Iterable[InstanceIdGroup]) -> str:
    """Format groups as numbered class sections separated by one blank line."""
    sections = [
        "\n".join(
            f"{group.semanticName} {index} {instanceId}"
            for index, instanceId in enumerate(group.instanceIds, start=1)
        )
        for group in groups
        if group.instanceIds
    ]
    return "\n\n".join(sections) + ("\n" if sections else "")


def writeInstanceIdDocument(
    path: str | Path,
    groups: Iterable[InstanceIdGroup],
) -> Path:
    """Write one UTF-8 InstanceID.txt file and return its resolved path."""
    outputPath = Path(path).expanduser().resolve()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(formatInstanceIdDocument(groups), encoding="utf-8")
    return outputPath


__all__ = [
    "InstanceIdGroup",
    "collectInstanceIdGroups",
    "formatInstanceIdDocument",
    "writeInstanceIdDocument",
]
