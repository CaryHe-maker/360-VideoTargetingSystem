"""When is a scan candidate the target?  Rules for accepting a re-acquisition, compared.

    python tools/candidate_study.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E022_c_tune --run outputs/E022_a2_tune --run outputs/E022_a3_tune \\
        --out outputs/E026

Reads the scan candidates recorded in the state traces of runs with loss handling
(every box found in a scan view, with its similarity to the template and its stateless
tracker score) and labels each with the ground truth: on the target when its centre is
within half a target size of the true centre.  The candidate table is written to
``<out>/candidates.csv``.

Rules are compared by how many of the on-target candidates they accept (recall) at a
given share of right acceptances (precision).  The fitted rule is a logistic regression
scored on sequences left out of the fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss_detector_study import Logistic  # noqa: E402

from track360.datasets.vot360 import Vot360Dataset  # noqa: E402

FOLDS = 5
PRECISIONS = (0.5, 0.7, 0.8, 0.9)
COLUMNS = (
    "run",
    "sequence",
    "frame",
    "onTarget",
    "present",
    "similarity",
    "score",
    "current",
    "margin",
    "sizeRatio",
    "verdict",
)
FEATURE_SETS = {
    "similarity": ("similarity",),
    "score": ("score",),
    "similarity + score": ("similarity", "score"),
    "+ margin": ("similarity", "score", "margin"),
    "+ margin + size": ("similarity", "score", "margin", "sizeRatio"),
}


def _direction(yawDeg: float, pitchDeg: float) -> np.ndarray:
    yaw, pitch = np.radians(yawDeg), np.radians(pitchDeg)
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def collect(args: argparse.Namespace) -> list[dict[str, object]]:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    rows = []
    for run in args.run:
        for path in sorted((run / "trace" / args.method).glob("*.csv")):
            name = path.stem
            sequence = dataset.sequence(name)
            truth, present = sequence.groundTruth("bfov")
            sequence.close()
            present = np.asarray(present, dtype=bool)
            with path.open(encoding="utf-8") as stream:
                frames = list(csv.DictReader(stream))
            lastSize = None
            for row in frames:
                index = int(row["frame"])
                candidates = json.loads(row["candidates"] or "[]")
                current = float(row["appearance"]) if row["appearance"] else 0.0
                for candidate in candidates:
                    if candidate["similarity"] is None:
                        continue
                    onTarget = False
                    if present[index]:
                        angle = np.degrees(
                            np.arccos(
                                np.clip(
                                    float(
                                        _direction(truth[index, 0], truth[index, 1])
                                        @ _direction(candidate["yawDeg"], candidate["pitchDeg"])
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                        size = float(np.sqrt(max(truth[index, 2] * truth[index, 3], 1e-6)))
                        onTarget = angle < 0.5 * max(size, float(candidate["sizeDeg"]))
                    reference = lastSize or float(candidate["sizeDeg"])
                    rows.append(
                        {
                            "run": run.name,
                            "sequence": name,
                            "frame": index,
                            "onTarget": int(onTarget),
                            "present": int(present[index]),
                            "similarity": float(candidate["similarity"]),
                            "score": float(candidate["score"]),
                            "current": current,
                            "margin": float(candidate["similarity"]) - current,
                            # Against the size of the box the track had when it was trusted.
                            "sizeRatio": abs(
                                float(np.log(max(float(candidate["sizeDeg"]), 1e-3) / reference))
                            ),
                            "verdict": candidate["verdict"],
                        }
                    )
                if row["modeAfter"] == "TRACKING" and row["sizeDeg"]:
                    lastSize = max(float(row["sizeDeg"]), 1e-3)
    return rows


def recallAt(scores: np.ndarray, label: np.ndarray, precision: float) -> tuple[float, float]:
    """The largest recall whose precision is at least ``precision``, and its threshold."""
    order = np.argsort(-scores)
    hits = np.cumsum(label[order])
    accepted = np.arange(1, len(order) + 1)
    valid = hits / accepted >= precision
    if not valid.any():
        return 0.0, float("inf")
    last = int(np.flatnonzero(valid)[-1])
    return float(hits[last] / max(1, label.sum())), float(scores[order][last])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    rows = collect(args)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    label = np.asarray([row["onTarget"] for row in rows], dtype=bool)
    sequences = np.asarray([row["sequence"] for row in rows])
    print(
        f"{len(rows)} candidates, {int(label.sum())} on the target, "
        f"from {len(set(sequences))} sequences ({len(set(sequences[label]))} with a hit)"
    )
    payload: dict[str, object] = {"candidates": len(rows), "onTarget": int(label.sum())}

    print("\nmedian (25%-75%) of each measure")
    for key in ("similarity", "score", "margin", "sizeRatio"):
        values = np.asarray([row[key] for row in rows])
        cells = []
        for mask in (label, ~label):
            low, mid, high = np.quantile(values[mask], [0.25, 0.5, 0.75])
            cells.append(f"{mid:.2f} ({low:.2f}-{high:.2f})")
        print(f"  {key:<12} on target {cells[0]:<22} elsewhere {cells[1]}")

    def fixedRule(similarity: float, margin: float, score: float) -> np.ndarray:
        return np.asarray(
            [
                row["similarity"] >= similarity
                and row["margin"] >= margin
                and row["score"] >= score
                for row in rows
            ]
        )

    print("\nfixed rules: accepted, of them on target (precision), share of the hits (recall)")
    payload["fixed"] = {}
    for similarity, margin, score in (
        (0.45, 0.15, 0.70),
        (0.45, 0.15, 0.50),
        (0.35, 0.15, 0.70),
        (0.35, 0.10, 0.70),
        (0.25, 0.10, 0.70),
        (0.25, 0.10, 0.80),
        (0.00, 0.10, 0.80),
    ):
        accepted = fixedRule(similarity, margin, score)
        precision = float(label[accepted].mean()) if accepted.any() else 0.0
        recall = float((accepted & label).sum() / max(1, label.sum()))
        key = f"similarity>={similarity:g} margin>={margin:g} score>={score:g}"
        payload["fixed"][key] = {
            "accepted": int(accepted.sum()), "precision": precision, "recall": recall
        }
        print(f"  {key:<46}{int(accepted.sum()):>7}{precision:>9.1%}{recall:>9.1%}")

    names = sorted(set(sequences))
    rng = np.random.default_rng(0)
    rng.shuffle(names)
    folds = [set(names[index::FOLDS]) for index in range(FOLDS)]
    print("\nfitted rules, scored on sequences left out of the fit: recall at a precision")
    print(f"  {'features':<22}" + "".join(f"{f'P>={p:g}':>9}" for p in PRECISIONS))
    payload["fitted"] = {}
    for title, members in FEATURE_SETS.items():
        matrix = np.asarray([[row[key] for key in members] for row in rows], dtype=float)
        scores = np.zeros(len(rows))
        for held in folds:
            test = np.asarray([name in held for name in sequences])
            if test.all() or not label[~test].any():
                continue
            scores[test] = Logistic().fit(matrix[~test], label[~test]).score(matrix[test])
        recalls = [recallAt(scores, label, precision)[0] for precision in PRECISIONS]
        final = Logistic().fit(matrix, label)
        payload["fitted"][title] = {
            "recallAtPrecision": dict(zip(map(str, PRECISIONS), recalls, strict=True)),
            "weights": dict(zip(["bias", *members], map(float, final.weights), strict=True)),
            "mean": dict(zip(members, map(float, final.mean), strict=True)),
            "spread": dict(zip(members, map(float, final.spread), strict=True)),
        }
        print(f"  {title:<22}" + "".join(f"{recall:>9.1%}" for recall in recalls))
    text = json.dumps(payload, indent=1) + "\n"
    (args.out / "candidates.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
