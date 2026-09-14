#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "Experiment" / "core_code"
sys.path.insert(0, str(CORE))

from emotionsketch.prompt_router import evaluate_router, evaluate_rule, iter_jsonl, train_router


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight Prompt-to-Q router.")
    parser.add_argument("--train", default=str(ROOT / "Experiment/datasets/prompt_q_router/train.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "Experiment/datasets/prompt_q_router/val.jsonl"))
    parser.add_argument("--stress", default=str(ROOT / "Experiment/datasets/prompt_q_router/stress_test_gold.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "Experiment/core_code/checkpoints/prompt_q_router.pkl"))
    parser.add_argument("--summary", default=str(ROOT / "Experiment/analysis/PROMPT_Q_ROUTER_RESULTS_2026-05-10.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    train_rows = list(iter_jsonl(args.train))
    val_rows = list(iter_jsonl(args.val))
    stress_rows = list(iter_jsonl(args.stress))
    router = train_router(train_rows, epochs=args.epochs, lr=args.lr)
    router.save(args.out)

    result = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "PROMPT_Q_ROUTER_TRAINED",
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "stress_count": len(stress_rows),
        "model": str(Path(args.out).relative_to(ROOT)),
        "router_val": evaluate_router(router, val_rows),
        "router_stress": evaluate_router(router, stress_rows),
        "rule_val": evaluate_rule(val_rows),
        "rule_stress": evaluate_rule(stress_rows),
        "boundary": (
            "The stress-test set is manually curated and should be user-audited before being described "
            "as a human gold set in the final report."
        ),
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = out.with_suffix(".md")
    md.write_text(
        "# Prompt-Q Router Results\n\n"
        f"Created: {result['created_at']}\n\n"
        "## Dataset Counts\n\n"
        f"- train: {len(train_rows)}\n"
        f"- val: {len(val_rows)}\n"
        f"- stress_test_gold: {len(stress_rows)}\n\n"
        "## Metrics\n\n"
        "| Split | Model | Accuracy | Macro-F1 |\n"
        "|---|---|---:|---:|\n"
        f"| val | lightweight router | {result['router_val']['accuracy']:.4f} | {result['router_val']['macro_f1']:.4f} |\n"
        f"| val | rule baseline | {result['rule_val']['accuracy']:.4f} | {result['rule_val']['macro_f1']:.4f} |\n"
        f"| stress | lightweight router | {result['router_stress']['accuracy']:.4f} | {result['router_stress']['macro_f1']:.4f} |\n"
        f"| stress | rule baseline | {result['rule_stress']['accuracy']:.4f} | {result['rule_stress']['macro_f1']:.4f} |\n\n"
        "## Boundary\n\n"
        f"{result['boundary']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "router_val_macro_f1": result["router_val"]["macro_f1"],
        "router_stress_macro_f1": result["router_stress"]["macro_f1"],
        "summary": str(out.relative_to(ROOT)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
