"""Print a LIBERO-style result row from a lerobot eval_info.json.

Matches the table in docs/source/libero.mdx, e.g.:

    | Model       | LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO 10 | Average |
    | ----------- | -------------- | ------------- | ----------- | --------- | ------- |
    | SmolVLA     | 64.0           | 75.0          | 76.0        | 42.0      | 64.2    |

Usage:
    python eval_table.py path/to/eval_info.json [--label "SmolVLA (50k, bf16)"] [--latex]
    python eval_table.py path/to/run_dir            # auto-resolves run_dir/eval_info.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HEADERS = ("LIBERO Spatial", "LIBERO Object", "LIBERO Goal", "LIBERO 10", "Average")


def load_info(path: Path) -> dict:
    if path.is_dir():
        path = path / "eval_info.json"
    with path.open() as f:
        return json.load(f)


def extract_row(info: dict) -> tuple[float, ...]:
    per = info["per_group"]
    suite_pc = tuple(per[s]["pc_success"] for s in SUITES)
    overall = info["overall"]["pc_success"]
    return (*suite_pc, overall)


def fmt_md(label: str, row: tuple[float, ...]) -> str:
    header = "| Model | " + " | ".join(HEADERS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in HEADERS) + " |"
    cells = " | ".join(f"{x:.1f}" for x in row)
    return f"{header}\n{sep}\n| {label} | {cells} |"


def fmt_latex(label: str, row: tuple[float, ...]) -> str:
    cells = " & ".join(f"{x:.1f}" for x in row)
    return f"{label} & {cells} \\\\"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="eval_info.json file or its parent run dir")
    ap.add_argument("--label", default="Policy", help="Row label / model name")
    ap.add_argument("--latex", action="store_true", help="Emit LaTeX row instead of markdown")
    args = ap.parse_args()

    info = load_info(args.path)
    row = extract_row(info)
    print(fmt_latex(args.label, row) if args.latex else fmt_md(args.label, row))


if __name__ == "__main__":
    main()
