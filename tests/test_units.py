import json
import random

import pytest
import torch

from clef_finetune.augment import AugmentConfig, augment_record, rename_choice_options
from clef_finetune.data import label_index, normalize_label, option_ids, validate_file, validate_record
from clef_finetune.jsm import VENDORED_PATH, VENDORED_SHA256, load_jsm, sha256
from clef_finetune.losses import brier, emd_ordinal, question_loss
from clef_finetune.metrics import coverage_at_precision, ece, macro_f1, summarize

Q = {
    "disp": {"type": "choice", "instructions": "d", "criteria": {"deny": "x", "approve": "y", "escalate": "z"}},
    "fraud": {"type": "noul", "instructions": "f"},
    "sev": {"type": "score", "instructions": "s", "criteria": ["lo", "mid", "hi"]},
}
REC = {"id": "r", "state": {"a": 1}, "questions": Q, "labels": {"disp": "escalate", "fraud": True, "sev": 2}}


class CharTok:
    def __call__(self, text, add_special_tokens=False):
        class R: ...
        r = R()
        r.input_ids = [ord(c) % 256 for c in text]
        return r


# ---------- vendored code ----------
def test_vendored_file_unmodified():
    assert sha256(VENDORED_PATH) == VENDORED_SHA256


def test_choice_ids_sorted_by_encoder():
    jsm = load_jsm()
    enc = jsm.encode_record(CharTok(), REC)
    by_id = {q.question_id: q.option_ids for q in enc.questions}
    assert by_id["disp"] == ("approve", "deny", "escalate")
    assert list(by_id["disp"]) == option_ids(Q["disp"])
    assert by_id["sev"] == ("0", "1", "2") and by_id["fraud"] == ("true", "false")


# ---------- validation ----------
def test_label_normalisation():
    assert normalize_label(Q["fraud"], False) == "false"
    assert normalize_label(Q["fraud"], "TRUE") == "true"
    assert normalize_label(Q["sev"], 1) == "1" and normalize_label(Q["sev"], "2") == "2"
    assert label_index(Q["disp"], "escalate") == 2  # sorted position, not insertion
    for q, bad in [("fraud", 1), ("sev", 3), ("sev", True), ("sev", 1.5), ("disp", "pay"), ("disp", 0)]:
        with pytest.raises(ValueError):
            normalize_label(Q[q], bad)


def test_validate_record_errors():
    assert validate_record(REC) == []
    bad = json.loads(json.dumps(REC))
    bad["labels"] = {"disp": "pay", "nope": 1, "sev": -1}
    errs = validate_record(bad)
    assert len(errs) == 3
    assert validate_record({"state": 1, "questions": {"q": {"type": "choice", "criteria": {"a": 1}}}, "labels": {}})
    assert validate_record({"state": 1, "questions": {"q": {"type": "bogus"}}, "labels": {"q": 1}})


def test_validate_file_balance(tmp_path):
    p = tmp_path / "d.jsonl"
    r2 = json.loads(json.dumps(REC)); r2["labels"] = {"disp": "deny", "fraud": False, "sev": 0}
    bad = json.loads(json.dumps(REC)); bad["labels"] = {"disp": "nope"}
    p.write_text("\n".join(json.dumps(x) for x in (REC, r2, bad)) + "\nnot json\n")
    rep = validate_file(p)
    assert rep["records"] == 4 and not rep["ok"] and len(rep["errors"]) == 2
    assert rep["balance"]["disp"]["counts"] == {"deny": 1, "escalate": 1}
    assert rep["balance"]["fraud"]["counts"] == {"false": 1, "true": 1}


# ---------- augmentation ----------
def test_rename_remaps_label():
    q, lab = rename_choice_options(Q["disp"], "escalate", {"escalate": "ESCALATE", "deny": "denied"})
    assert set(q["criteria"]) == {"approve", "denied", "ESCALATE"} and lab == "ESCALATE"
    assert q["criteria"]["ESCALATE"] == "z"
    with pytest.raises(ValueError):
        rename_choice_options(Q["disp"], "deny", {"deny": "approve"})


def test_augment_preserves_semantics_over_many_seeds():
    cfg = AugmentConfig(question_subset_prob=0.5, paraphrase_prob=1.0, rename_option_prob=1.0, criteria_dropout_prob=0.3,
                        paraphrases={"disp": ["What next? {instructions}"]},
                        option_aliases={"disp": {"approve": ["APPROVE", "approved"], "deny": ["DENY"], "escalate": ["ESCALATE", "refer"]}})
    rng = random.Random(0)
    orders = set()
    for _ in range(300):
        out = augment_record(REC, cfg, rng)
        assert validate_record(out) == []
        orders.add(tuple(out["questions"]))
        if "disp" in out["labels"]:
            crit = out["questions"]["disp"]["criteria"]
            lab = out["labels"]["disp"]
            # the renamed label must carry the original 'escalate' description (or dropped None)
            assert crit[lab] in ("z", None)
            assert out["questions"]["disp"]["instructions"].startswith("What next?")
        if "sev" in out["questions"]:
            # score order is semantic: never shuffled, label unchanged
            assert len(out["questions"]["sev"]["criteria"]) == 3 and out["labels"].get("sev", 2) == 2
            assert out["questions"]["sev"]["criteria"][2] in ("hi", "level 2")
        if "fraud" in out["labels"]:
            assert out["labels"]["fraud"] is True
        assert out["labels"]  # at least one labelled question kept
    assert len(orders) > 5
    assert REC["labels"]["disp"] == "escalate"  # input untouched


def test_augment_is_seeded():
    cfg = AugmentConfig(rename_option_prob=1.0)
    a = augment_record(REC, cfg, random.Random(7))
    b = augment_record(REC, cfg, random.Random(7))
    assert a == b


# ---------- losses ----------
def test_losses():
    good = torch.tensor([10.0, -10.0, -10.0])
    assert brier(good, 0).item() < 1e-6
    assert abs(brier(torch.zeros(2), 0).item() - 0.5) < 1e-6
    near = torch.tensor([-10.0, 10.0, -10.0, -10.0])
    far = torch.tensor([-10.0, -10.0, -10.0, 10.0])
    assert emd_ordinal(near, 0) < emd_ordinal(far, 0)
    assert emd_ordinal(good, 0).item() < 1e-6
    l_choice, p1 = question_loss(near, 0, "choice", ordinal_weight=1.0)
    l_score, p2 = question_loss(near, 0, "score", ordinal_weight=1.0)
    assert "ordinal" not in p1 and "ordinal" in p2 and l_score > l_choice
    x = torch.zeros(3, requires_grad=True)
    question_loss(x, 1, "score", ordinal_weight=0.5)[0].backward()
    assert x.grad[1] < 0


# ---------- metrics ----------
def test_metrics():
    assert macro_f1([0, 0, 1, 1], [0, 1, 1, 1], 2) == pytest.approx((2 / 3 + 0.8) / 2)
    assert ece([[1.0, 0.0]] * 4, [0, 0, 0, 0]) == pytest.approx(0.0)
    assert ece([[0.9, 0.1]] * 10, [0] * 5 + [1] * 5) == pytest.approx(0.4)


def test_coverage_at_precision():
    conf = [0.99, 0.98, 0.97, 0.9, 0.8, 0.6]
    ok = [True, True, True, False, True, False]
    r = coverage_at_precision(conf, ok, 0.99)
    assert r["coverage"] == pytest.approx(0.5) and r["threshold"] == 0.97 and r["precision"] == 1.0
    r = coverage_at_precision(conf, ok, 0.8)
    assert r["coverage"] == pytest.approx(5 / 6) and r["threshold"] == 0.8
    assert coverage_at_precision([0.9, 0.8], [False, True], 0.99)["coverage"] == 0.0
    # ties are never split
    r = coverage_at_precision([0.9, 0.9, 0.5], [True, False, True], 0.99)
    assert r["coverage"] == 0.0


def test_summarize():
    rows = [{"question_id": "a", "probs": [0.9, 0.1], "target": 0}, {"question_id": "a", "probs": [0.2, 0.8], "target": 1},
            {"question_id": "b", "probs": [0.3, 0.3, 0.4], "target": 0}]
    s = summarize(rows)
    assert s["per_question"]["a"]["accuracy"] == 1.0 and s["per_question"]["b"]["accuracy"] == 0.0
    assert "macro_f1" not in s["overall"]  # mixed option counts
    assert set(s["per_question"]["a"]["auto_decide"]) == {"0.99", "0.95"}
