"""Schema augmentation -- the main lever for generalising to unseen schema wording.

All transforms return a NEW record with labels remapped consistently.

Important facts about Cloudflare's encoder (``encode_record``):
- question order is preserved (dict insertion order) -> shuffling it matters;
- ``choice`` option ids are SORTED alphabetically before encoding, so shuffling
  the ``criteria`` dict is a no-op. Only *renaming* ids changes what the model sees
  (and can change their sorted position);
- ``score`` levels are an ordered list whose index IS the label; order is semantic
  and is never shuffled here;
- ``noul`` options are always "true"/"false"; only their descriptions can change.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _case_variants(option_id: str) -> list[str]:
    base = option_id.replace("-", "_").replace(" ", "_")
    words = [w for w in base.split("_") if w]
    out = {
        option_id.upper(),
        option_id.lower(),
        "_".join(words).lower(),
        "-".join(words).lower(),
        "".join(w.capitalize() for w in words),
        " ".join(words).lower(),
    }
    out.discard(option_id)
    return sorted(out)


@dataclass
class AugmentConfig:
    shuffle_questions: bool = True
    question_subset_prob: float = 0.3  # chance of dropping to a random non-empty subset
    min_questions: int = 1
    paraphrase_prob: float = 0.5
    rename_option_prob: float = 0.3  # per choice question
    criteria_dropout_prob: float = 0.0  # per choice option description / score level text
    # {question_id: [instruction template, ...]}; "{instructions}" is the original text
    paraphrases: dict[str, list[str]] = field(default_factory=dict)
    # {question_id: {option_id: [alias, ...]}}; falls back to case variants when absent
    option_aliases: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    generic_paraphrases: list[str] = field(default_factory=list)  # applied to any question

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AugmentConfig":
        data = dict(data or {})
        templates = data.pop("templates_file", None)
        cfg = cls(**data)
        if templates:
            cfg.merge_templates(load_templates(templates))
        return cfg

    def merge_templates(self, t: dict[str, Any]) -> None:
        for qid, items in (t.get("paraphrases") or {}).items():
            self.paraphrases.setdefault(qid, []).extend(items)
        for qid, mapping in (t.get("option_aliases") or {}).items():
            self.option_aliases.setdefault(qid, {}).update(mapping)
        self.generic_paraphrases.extend(t.get("generic_paraphrases") or [])


def load_templates(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def rename_choice_options(question: dict[str, Any], label: Any, mapping: dict[str, str]) -> tuple[dict[str, Any], Any]:
    """Rename choice option ids; returns (new_question, new_label). Mapping must be injective."""
    if len(set(mapping.values())) != len(mapping) or len({mapping.get(k, k) for k in question["criteria"]}) != len(question["criteria"]):
        raise ValueError("option renaming must be injective")
    q = copy.deepcopy(question)
    q["criteria"] = {mapping.get(k, k): v for k, v in question["criteria"].items()}
    new_label = mapping.get(label, label) if label is not None else None
    return q, new_label


def augment_record(record: dict[str, Any], cfg: AugmentConfig, rng: random.Random) -> dict[str, Any]:
    out = copy.deepcopy(record)
    questions: dict[str, Any] = out["questions"]
    labels: dict[str, Any] = dict(out.get("labels") or {})
    qids = list(questions)

    # 1. random question subset (keep >=1 labelled question when possible)
    if len(qids) > cfg.min_questions and rng.random() < cfg.question_subset_prob:
        k = rng.randint(cfg.min_questions, len(qids))
        keep = set(rng.sample(qids, k))
        if labels and not keep & set(labels):
            keep.add(rng.choice([q for q in qids if q in labels]))
        qids = [q for q in qids if q in keep]

    # 2. shuffle question order (encoder preserves insertion order)
    if cfg.shuffle_questions:
        rng.shuffle(qids)

    new_questions: dict[str, Any] = {}
    for qid in qids:
        q = questions[qid]
        # 3. instruction paraphrase
        templates = list(cfg.paraphrases.get(qid, [])) + list(cfg.generic_paraphrases)
        if templates and rng.random() < cfg.paraphrase_prob:
            original = q.get("instructions") or qid
            q["instructions"] = rng.choice(templates).replace("{instructions}", str(original))
        # 4. option-id renaming (choice only) with label remap
        if q["type"] == "choice" and rng.random() < cfg.rename_option_prob:
            aliases = cfg.option_aliases.get(qid, {})
            mapping: dict[str, str] = {}
            used: set[str] = set()
            for oid in q["criteria"]:
                pool = list(aliases.get(oid, [])) or _case_variants(oid)
                pool = [p for p in pool if p not in used and (p == oid or p not in q["criteria"])]
                new = rng.choice(pool) if pool else oid
                if new in used:
                    new = oid
                mapping[oid] = new
                used.add(new)
            if len(set(mapping.values())) == len(mapping):
                q, new_label = rename_choice_options(q, labels.get(qid), mapping)
                if qid in labels:
                    labels[qid] = new_label
        # 5. criteria-description dropout (ids/levels stay; description -> None/short)
        if cfg.criteria_dropout_prob > 0:
            if q["type"] == "choice":
                q["criteria"] = {
                    k: (None if rng.random() < cfg.criteria_dropout_prob else v) for k, v in q["criteria"].items()
                }
            elif q["type"] == "score":
                q["criteria"] = [
                    (f"level {i}" if rng.random() < cfg.criteria_dropout_prob else v)
                    for i, v in enumerate(q["criteria"])
                ]
        new_questions[qid] = q

    out["questions"] = new_questions
    out["labels"] = {k: v for k, v in labels.items() if k in new_questions}
    return out
