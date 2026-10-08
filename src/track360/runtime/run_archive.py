"""Keep a small, versioned record of every benchmark run.

Result files live under ``outputs/``, which is local and not committed.  A record
holds what is needed to look a run up later without repeating it: the scores per
sequence, the cost per sequence, and the configuration and commit that produced
them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from track360.core.errors import DecodeError
from track360.runtime.benchmark import REPORT_DIRECTORY, evaluateResults, loadEfficiency

RECORD_FORMAT = "track360.run-record.v1"
INDEX_NAME = "README.md"
# How far the timings of a run can be trusted.
TIMING_STATES = ("solo", "parallel", "unknown")


def archiveRun(
    *,
    datasetRoot: str | Path,
    outputRoot: str | Path,
    method: str,
    name: str,
    archiveRoot: str | Path,
    labelRoot: str | Path | None = None,
    sequences: Sequence[str] | None = None,
    split: str = "",
    experiment: str = "",
    note: str = "",
    timing: str = "unknown",
) -> Path:
    """Score one run and write its record; returns the record's path.

    ``timing`` says whether the run had the machine to itself: ``solo``, ``parallel``
    or ``unknown``.  Latencies of anything but a solo run are not comparable.
    """
    if timing not in TIMING_STATES:
        raise ValueError(f"timing must be one of {', '.join(TIMING_STATES)}")
    if not name or any(character in name for character in '/\\:*?"<>| '):
        raise ValueError(f"record name must be a plain file name without spaces: '{name}'")
    scores = evaluateResults(
        datasetRoot=datasetRoot,
        outputRoot=outputRoot,
        labelRoot=labelRoot,
        methods=[method],
        only=sequences,
    ).get(method)
    if scores is None or "bbox" not in scores:
        raise DecodeError(f"no BBox results of '{method}' under {outputRoot}")
    bbox = scores["bbox"]
    reportRoot = Path(outputRoot) / REPORT_DIRECTORY / method
    run = _readJson(reportRoot / "run.json") or {}
    perSequence: dict[str, dict[str, Any]] = {}
    for sequence, item in sorted(bbox.perSequence.items()):
        report = _readJson(reportRoot / f"{sequence}.json") or {}
        perSequence[sequence] = {
            "S_dual": item.success,
            "P_angle": item.anglePrecision,
            "P_dual": item.precision,
            "frames": item.frames,
            "lostFrames": item.lostFrames,
            "firstLostFrame": item.firstLostFrame,
            "fps": report.get("fps"),
            "p50LatencyMs": report.get("p50LatencyMs"),
            "p95LatencyMs": report.get("p95LatencyMs"),
            "forwards": report.get("forwards"),
            "invalidFrames": report.get("invalidFrames"),
        }
    efficiency = loadEfficiency(outputRoot, method, list(perSequence))
    record = {
        "format": RECORD_FORMAT,
        "name": name,
        "experiment": experiment,
        "split": split,
        "method": method,
        "note": note,
        "outputRoot": str(outputRoot).replace("\\", "/"),
        "createdAt": run.get("createdAt"),
        "git": run.get("git"),
        "configHash": run.get("configHash"),
        "weightsSha256": run.get("weightsSha256"),
        "config": run.get("config"),
        "environment": {
            key: run.get(key) for key in ("python", "platform", "packages", "device")
        },
        "scores": {
            "bbox": bbox.summary(),
            "bfov": scores["bfov"].summary() if "bfov" in scores else None,
        },
        "efficiency": {
            "timing": timing,
            "sequences": efficiency.sequences if efficiency else 0,
            "forwardsPerFrame": efficiency.forwardsPerFrame if efficiency else None,
            "p50LatencyMs": efficiency.p50LatencyMs if efficiency else None,
            "p95LatencyMs": efficiency.p95LatencyMs if efficiency else None,
            "fps": efficiency.fps if efficiency else None,
        },
        "perSequence": perSequence,
    }
    root = Path(archiveRoot)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    writeIndex(root)
    return path


def loadRecords(archiveRoot: str | Path) -> list[dict[str, Any]]:
    """Every record under ``archiveRoot``, oldest run first."""
    records = []
    for path in sorted(Path(archiveRoot).glob("*.json")):
        record = _readJson(path)
        if record and record.get("format") == RECORD_FORMAT:
            records.append(record)
    return sorted(records, key=lambda record: (record.get("createdAt") or "", record["name"]))


def writeIndex(archiveRoot: str | Path) -> Path:
    """Regenerate the table of all records."""
    root = Path(archiveRoot)
    lines = [
        "# 评测运行记录",
        "",
        "本目录由 `tools/benchmark.py archive` 生成，每个 JSON 文件对应一次运行："
        "逐条序列的分数和耗时、生效的配置、commit 和环境。"
        "结果文件本身在本地的 `outputs/` 下，不随仓库提交；"
        "需要某次运行的数字时先查这里，不要重跑。",
        "",
        "延迟和 FPS 只有“计时”一列为 `solo`（运行时机器上没有别的评测任务）的才可以相互比较；"
        "每帧前向次数不受机器影响。实验的结论和上下文见 [评测记录](../evaluation-log.md)。",
        "",
        "| 记录 | 实验 | 划分 | 方法 | 运行时间 (UTC) | Commit | 配置哈希 | 序列 "
        "| S<sub>dual</sub> "
        "| P<sub>angle</sub> | 丢失率 | 前向 / 帧 | P50 ms | P95 ms | FPS | 计时 | 说明 |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in loadRecords(root):
        bbox = record["scores"]["bbox"]
        efficiency = record["efficiency"]
        git = record.get("git") or {}
        commit = str(git.get("commit") or "")[:7] + ("*" if git.get("dirty") else "")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{record['name']}]({record['name']}.json)",
                    record.get("experiment") or "",
                    record.get("split") or "",
                    f"`{record['method']}`",
                    str(record.get("createdAt") or "")[:16].replace("T", " "),
                    f"`{commit}`" if commit else "",
                    f"`{str(record.get('configHash') or '')[:8]}`",
                    str(bbox["sequences"]),
                    _number(bbox["S_dual"], 3),
                    _number(bbox["P_angle"], 3),
                    _number(bbox["loss_rate"], 3),
                    _number(efficiency["forwardsPerFrame"], 2),
                    _number(efficiency["p50LatencyMs"], 0),
                    _number(efficiency["p95LatencyMs"], 0),
                    _number(efficiency["fps"], 1),
                    efficiency["timing"],
                    record.get("note") or "",
                ]
            )
            + " |"
        )
    lines += ["", "Commit 后的 `*` 表示运行时工作区有未提交的改动。", ""]
    path = root / INDEX_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _number(value: float | None, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _readJson(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__ = ["RECORD_FORMAT", "TIMING_STATES", "archiveRun", "loadRecords", "writeIndex"]
