"""Summarise ``refine_study.py``: second look by distance, zoom scale for small targets.

python tools/refine_report.py outputs/E056/replay/rows.json --out outputs/E056/report.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TAKE = 0.70
BINS = ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path)
    parser.add_argument("--small-deg", type=float, default=4.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    looks = []
    for r in rows:
        e = r.get(f"{r['recordedScale']:g}x")
        if e and "iou2" in e:
            looks.append(e)
    lines = [f"{len(looks)} searches with both looks (target in the picture)"]
    lines.append(
        "first box's distance from its view centre (share of half width): n | first look: "
        "boxed, mean IoU (all / boxed), caught | second look: boxed, mean IoU, caught"
    )
    for low, high in BINS:
        part = [e for e in looks if low <= e["offset1"] < high]
        if not part:
            continue
        i1 = np.array([e["iou1"] for e in part])
        i2 = np.array([e["iou2"] for e in part])
        c1 = np.mean([(e["iou1"] >= 0.5) and e["score1"] >= TAKE for e in part])
        c2 = np.mean([(e["iou2"] >= 0.5) and e["score2"] >= TAKE for e in part])
        boxedIou = i1[i1 >= 0.5].mean() if (i1 >= 0.5).any() else float("nan")
        lines.append(
            f"  {low:.2f}-{high:<4} n={len(part):>5} | {np.mean(i1 >= 0.5):6.1%}  {i1.mean():.3f} / {boxedIou:.3f}"  # noqa: E501
            f"  {c1:6.1%} | {np.mean(i2 >= 0.5):6.1%}  {i2.mean():.3f}  {c2:6.1%}"
        )
    lines.append(
        "second look only when the first box is farther than d: caught, boxed, extra forwards per search"  # noqa: E501
    )
    base = np.mean([(e["iou1"] >= 0.5) and e["score1"] >= TAKE for e in looks])
    lines.append(
        f"  never      caught {base:6.1%}  boxed {np.mean([e['iou1'] >= 0.5 for e in looks]):6.1%}"
    )
    for d in (0.0, 0.1, 0.2, 0.3, 0.5):
        caught, boxed, extra = [], [], 0
        for e in looks:
            if e["offset1"] > d:
                extra += 1
                iou, score = e["iou2"], e["score2"]
            else:
                iou, score = e["iou1"], e["score1"]
            caught.append(iou >= 0.5 and score >= TAKE)
            boxed.append(iou >= 0.5)
        lines.append(
            f"  d > {d:.1f}    caught {np.mean(caught):6.1%}  boxed {np.mean(boxed):6.1%}  extra {extra / len(looks):.2f}"  # noqa: E501
        )
    # Zoom scale for small targets.
    small = [
        r for r in rows if r["trustedSizeDeg"] < args.small_deg and r.get("2x") and r.get("4x")
    ]
    lines.append(f"targets below {args.small_deg:g} deg, both zoom scales replayed: n={len(small)}")
    for recorded in (2.0, 4.0):
        part = [r for r in small if r["recordedScale"] == recorded]
        if not part:
            continue
        cells = []
        for key in ("2x", "4x"):
            boxed = np.mean([r[key]["iou1"] >= 0.5 for r in part])
            caught = np.mean([(r[key]["iou1"] >= 0.5) and r[key]["score1"] >= TAKE for r in part])
            cells.append(f"{key}: boxed {boxed:6.1%} caught {caught:6.1%}")
        lines.append(f"  recorded {recorded:g}x (n={len(part)}): " + "  ".join(cells))
    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
