"""Label every training report with an offline open-weight LLM (Kaggle GPU, internet off).

    python scripts/label_reports_llm.py --data /kaggle/input/rsna-knee-abnormality-detection \
        --model /kaggle/input/<your-llm-weights> --out labels/llm_states.csv [--backend vllm]

Writes soft labels, per-cell weights (``__w``), explicitness (``__e``) and the raw state
(``__state``), then prints gold-58 agreement and runs the gold-leak check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneemri.io import read_table  # noqa: E402
from kneemri.constants import ID_COL, TARGETS  # noqa: E402
from kneemri.labels.llm_labeler import label_reports, transformers_generate_fn, vllm_generate_fn  # noqa: E402
from kneemri.labels.teachers import detect_gold_leak, gold_agreement, gold_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    train = read_table(Path(a.data) / "train.csv")
    if a.limit:
        train = train.head(a.limit)
    gen = vllm_generate_fn(a.model) if a.backend == "vllm" else transformers_generate_fn(a.model)
    out = label_reports(train, gen, batch_size=a.batch_size)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    gold = gold_rows(train)
    teacher = out.set_index(ID_COL)[TARGETS]
    print("gold agreement:\n", gold_agreement(teacher, gold).round(3).to_string())
    print("gold agreement macro:", round(float(gold_agreement(teacher, gold).mean()), 4))
    print("leak check:", detect_gold_leak(teacher, gold))


if __name__ == "__main__":
    main()
