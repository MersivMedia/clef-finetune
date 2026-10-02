"""Evaluation and base-vs-tuned comparison reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .data import load_jsonl, validate_record
from .metrics import summarize


@torch.inference_mode()
def predict_rows(model, tokenizer, processor, jsm, records, base_dir: Path, max_length: int = 8192) -> list[dict[str, Any]]:
    from .train import encode_example

    device = next(model.parameters()).device
    rows = []
    for record in records:
        encoded, targets = encode_example(jsm, tokenizer, processor, record, base_dir, max_length)
        logits = model(jsm.collate_records([encoded], tokenizer.pad_token_id, device))[0]
        for q, ql, t in zip(encoded.questions, logits, targets):
            if t is None:
                continue
            rows.append(
                {
                    "record_id": record.get("id"),
                    "question_id": q.question_id,
                    "type": record["questions"][q.question_id]["type"],
                    "option_ids": list(q.option_ids),
                    "probs": ql.float().softmax(-1).tolist(),
                    "target": t,
                }
            )
    return rows


def evaluate(model_dir, eval_path, checkpoint=None, device="cpu", dtype="bfloat16", target_precisions=(0.99, 0.95), max_length=8192):
    from .model import build_eval_model

    records = load_jsonl(eval_path)
    for i, r in enumerate(records):
        errs = validate_record(r)
        if errs:
            raise ValueError(f"eval record {i}: {errs[0]}")
    model, tok, proc, jsm = build_eval_model(model_dir, device, getattr(torch, dtype), checkpoint)
    rows = predict_rows(model, tok, proc, jsm, records, Path(eval_path).parent, max_length)
    return {"model_dir": str(model_dir), "checkpoint": str(checkpoint) if checkpoint else None, "eval": str(eval_path),
            "metrics": summarize(rows, target_precisions), "rows": rows}


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def compare_markdown(base: dict, tuned: dict) -> str:
    lines = ["# Base vs tuned", "", f"eval set: `{tuned['eval']}`", "",
             "| question | n | metric | base | tuned | delta |", "|---|---|---|---|---|---|"]
    bq, tq = base["metrics"]["per_question"], tuned["metrics"]["per_question"]
    keys = ["overall"] + sorted(tq)
    for q in keys:
        b = base["metrics"]["overall"] if q == "overall" else bq.get(q, {})
        t = tuned["metrics"]["overall"] if q == "overall" else tq[q]
        for m in ("accuracy", "macro_f1", "nll", "brier", "ece"):
            if m in t:
                bv, tv = b.get(m), t[m]
                d = tv - bv if isinstance(bv, float) and isinstance(tv, float) else None
                lines.append(f"| {q} | {t['n']} | {m} | {_fmt(bv)} | {_fmt(tv)} | {_fmt(d)} |")
        for tp, cov in t["auto_decide"].items():
            bc = (b.get("auto_decide") or {}).get(tp, {})
            lines.append(
                f"| {q} | {t['n']} | auto-decide coverage @ {tp} precision (threshold) | "
                f"{_fmt(bc.get('coverage'))} ({_fmt(bc.get('threshold'))}) | {_fmt(cov['coverage'])} ({_fmt(cov['threshold'])}) | "
                f"{_fmt(cov['coverage'] - bc['coverage'] if bc else None)} |"
            )
    lines += ["", "Lower is better for nll / brier / ece. Auto-decide coverage = fraction of cases whose top "
              "confidence clears the threshold while precision on those cases stays at the target; the rest go to a human. "
              "Small eval sets make the high-precision numbers very noisy."]
    return "\n".join(lines) + "\n"


def write_reports(out_dir: Path, base: dict | None, tuned: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    strip = lambda r: {k: v for k, v in r.items() if k != "rows"}
    (out_dir / "tuned_metrics.json").write_text(json.dumps(strip(tuned), indent=2))
    with (out_dir / "tuned_predictions.jsonl").open("w") as fh:
        for row in tuned["rows"]:
            fh.write(json.dumps(row) + "\n")
    if base is not None:
        (out_dir / "base_metrics.json").write_text(json.dumps(strip(base), indent=2))
        (out_dir / "comparison.json").write_text(json.dumps({"base": strip(base)["metrics"], "tuned": strip(tuned)["metrics"]}, indent=2))
        (out_dir / "comparison.md").write_text(compare_markdown(base, tuned))
