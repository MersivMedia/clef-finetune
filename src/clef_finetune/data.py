"""Training data format and validation.

One JSON object per line -- a System One request plus labels::

    {"id": "c1", "state": {...}, "questions": {...}, "images": ["a.jpg"],
     "labels": {"disposition": "approve", "fraud": false, "severity": 2}}

Label conventions
- noul:   true / false (bool) or "true" / "false"
- choice: an option id (a key of ``criteria``)
- score:  integer level 0..n-1 (index into the ``criteria`` list) or its string
A question without a label is still encoded (it contributes context) but adds no loss.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

QUESTION_TYPES = ("noul", "choice", "score")


def option_ids(question: dict[str, Any]) -> list[str]:
    """Option ids in the order the model sees them (mirrors ``question_options``).

    NOTE: Cloudflare's encoder SORTS choice option ids alphabetically.
    """
    qtype = question["type"]
    if qtype == "noul":
        return ["true", "false"]
    if qtype == "choice":
        return sorted(str(k) for k in question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def normalize_label(question: dict[str, Any], label: Any) -> str:
    """Map a raw label to an option id string; raise ValueError if illegal."""
    qtype = question["type"]
    if qtype == "noul":
        if isinstance(label, bool):
            return "true" if label else "false"
        if isinstance(label, str) and label.lower() in ("true", "false"):
            return label.lower()
        raise ValueError(f"noul label must be true/false, got {label!r}")
    if qtype == "choice":
        if not isinstance(label, str):
            raise ValueError(f"choice label must be a string option id, got {label!r}")
        if label not in {str(k) for k in question["criteria"]}:
            raise ValueError(f"choice label {label!r} not in options {sorted(question['criteria'])}")
        return label
    if qtype == "score":
        if isinstance(label, int) and not isinstance(label, bool):
            level = label
        elif isinstance(label, str) and label.strip().isdigit():
            level = int(label)
        else:
            raise ValueError(f"score label must be an integer level, got {label!r}")
        n = len(question["criteria"])
        if not 0 <= level < n:
            raise ValueError(f"score label {level} outside 0..{n - 1}")
        return str(level)
    raise ValueError(f"unknown question type {qtype!r}")


def label_index(question: dict[str, Any], label: Any) -> int:
    return option_ids(question).index(normalize_label(question, label))


def validate_question(qid: str, question: Any) -> list[str]:
    errors = []
    if not isinstance(question, dict):
        return [f"{qid}: question must be an object"]
    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        return [f"{qid}: type must be one of {QUESTION_TYPES}, got {qtype!r}"]
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            errors.append(f"{qid}: choice criteria must be an object with >=2 options")
    elif qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            errors.append(f"{qid}: score criteria must be a list with >=2 levels")
    elif criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
        errors.append(f"{qid}: noul criteria may only override 'true'/'false' descriptions")
    return errors


def validate_record(record: Any, require_labels: bool = True) -> list[str]:
    if not isinstance(record, dict):
        return ["record must be a JSON object"]
    errors = []
    if "state" not in record:
        errors.append("missing 'state'")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        return errors + ["'questions' must be a non-empty object"]
    for qid, q in questions.items():
        errors.extend(validate_question(qid, q))
    images = record.get("images")
    if images is not None and (not isinstance(images, list) or not all(isinstance(i, str) for i in images)):
        errors.append("'images' must be a list of file paths")
    labels = record.get("labels")
    if labels is None:
        if require_labels:
            errors.append("missing 'labels'")
        return errors
    if not isinstance(labels, dict):
        return errors + ["'labels' must be an object"]
    if require_labels and not labels:
        errors.append("'labels' is empty")
    for qid, label in labels.items():
        if qid not in questions:
            errors.append(f"label for unknown question {qid!r}")
            continue
        if validate_question(qid, questions[qid]):
            continue
        try:
            normalize_label(questions[qid], label)
        except ValueError as error:
            errors.append(f"{qid}: {error}")
    return errors


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {error}") from None
    return records


def resolve_images(record: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Make relative image paths relative to the JSONL file; load them as PIL images."""
    if not record.get("images"):
        return record
    from PIL import Image

    out = dict(record)
    out["images"] = [Image.open(p if Path(p).is_absolute() else base_dir / p).convert("RGB") for p in record["images"]]
    return out


def validate_file(path: str | Path, require_labels: bool = True) -> dict[str, Any]:
    """Validate a JSONL file; return a report with errors and class balance."""
    report: dict[str, Any] = {"file": str(path), "records": 0, "errors": [], "balance": {}}
    balance: dict[str, Counter] = defaultdict(Counter)
    types: dict[str, str] = {}
    base = Path(path).parent
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        report["records"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            report["errors"].append(f"line {lineno}: invalid JSON: {error}")
            continue
        for err in validate_record(record, require_labels=require_labels):
            report["errors"].append(f"line {lineno} ({record.get('id', '?') if isinstance(record, dict) else '?'}): {err}")
        if not isinstance(record, dict):
            continue
        for img in record.get("images") or []:
            if isinstance(img, str) and not (Path(img) if Path(img).is_absolute() else base / img).exists():
                report["errors"].append(f"line {lineno}: image not found: {img}")
        questions = record.get("questions") or {}
        for qid, label in (record.get("labels") or {}).items():
            q = questions.get(qid)
            if not isinstance(q, dict):
                continue
            try:
                balance[qid][normalize_label(q, label)] += 1
                types[qid] = q["type"]
            except (ValueError, KeyError):
                pass
    for qid, counts in balance.items():
        total = sum(counts.values())
        report["balance"][qid] = {
            "type": types[qid],
            "n": total,
            "counts": dict(sorted(counts.items())),
            "majority_fraction": round(max(counts.values()) / total, 4),
        }
    report["ok"] = not report["errors"]
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = [f"{report['file']}: {report['records']} records, {len(report['errors'])} errors"]
    for err in report["errors"][:50]:
        lines.append(f"  ERROR {err}")
    if len(report["errors"]) > 50:
        lines.append(f"  ... {len(report['errors']) - 50} more")
    lines.append("class balance:")
    for qid, b in report["balance"].items():
        counts = ", ".join(f"{k}={v}" for k, v in b["counts"].items())
        warn = "  <-- imbalanced (>90% one class)" if b["majority_fraction"] > 0.9 else ""
        lines.append(f"  {qid} [{b['type']}] n={b['n']}: {counts}{warn}")
    return "\n".join(lines)


def iter_labelled(record: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], int]]:
    questions = record["questions"]
    for qid, label in (record.get("labels") or {}).items():
        yield qid, questions[qid], label_index(questions[qid], label)
