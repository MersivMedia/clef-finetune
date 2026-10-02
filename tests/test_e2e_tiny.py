"""End-to-end CPU test with a tiny RANDOM Qwen3_5ForConditionalGeneration.

data -> augment -> train (loss must drop on a repeated record) -> eval -> save
adapter -> merge -> load_release_model (Cloudflare's loader, unchanged) ->
same logits as the pre-merge PEFT model.
"""

import json
from pathlib import Path

import pytest
import torch

from clef_finetune.data import load_jsonl
from clef_finetune.evaluate import compare_markdown, evaluate, write_reports
from clef_finetune.export import merge
from clef_finetune.jsm import load_jsm
from clef_finetune.model import build_eval_model
from clef_finetune.tiny import build_tiny_release
from clef_finetune.train import encode_example, load_config, train

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "insurance-claims"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return build_tiny_release(tmp_path_factory.mktemp("release") / "clef-tiny")


@pytest.fixture(scope="module")
def run(tiny, tmp_path_factory):
    work = tmp_path_factory.mktemp("run")
    one = work / "one.jsonl"
    rec = load_jsonl(EX / "train.synthetic.jsonl")[0]
    one.write_text(json.dumps(rec) + "\n")
    cfg = load_config(ROOT / "configs" / "tiny-cpu.yaml", {
        "model": {"path": str(tiny), "dtype": "float32"},
        "data": {"train": str(one)},
        "train": {"max_steps": 5, "grad_accum": 1, "augment_train": False, "save_every": 5, "lr": 3e-3, "head_lr": 3e-3},
        "output_dir": str(work / "out"),
    })
    result = train(cfg, device="cpu")
    (work / "losses.json").write_text(json.dumps([h["loss"] for h in result["history"]]))
    return cfg, result, work


def test_loss_decreases_on_repeated_record(run):
    _, result, _ = run
    losses = [h["loss"] for h in result["history"]]
    print("tiny training losses:", [round(x, 5) for x in losses])
    assert len(losses) == 5
    assert losses[-1] < losses[0]


def test_lora_targets_cover_linear_attention(run, tiny):
    """Reload the saved adapter and check what PEFT actually wraps, not what the config says."""
    from peft import PeftModel

    from clef_finetune.model import load_backbone

    _, result, _ = run
    ckpt = Path(result["checkpoint"]).parent / "checkpoint-5"
    cfg = json.loads((ckpt / "adapter" / "adapter_config.json").read_text())
    assert isinstance(cfg["target_modules"], str)  # full-name regex, not bare suffixes
    reloaded = PeftModel.from_pretrained(load_backbone(tiny, torch.float32, "cpu"), ckpt / "adapter")
    wrapped = [n[: -len(".lora_A")] for n, _ in reloaded.named_modules() if n.endswith(".lora_A")]
    names = " ".join(wrapped)
    for suffix in ("linear_attn.in_proj_qkv", "linear_attn.out_proj", "self_attn.q_proj", "mlp.down_proj"):
        assert suffix in names
    assert not any(".visual." in n or n.endswith("lm_head") for n in wrapped)
    assert len(wrapped) == result["lora_targets"]


def test_resume_continues_from_checkpoint(run, tiny):
    cfg, result, work = run
    cfg = json.loads(json.dumps(cfg))
    cfg["train"]["max_steps"] = 6
    cfg["train"]["save_every"] = 6
    res2 = train(cfg, device="cpu", resume=True)
    assert [h["step"] for h in res2["history"]] == [6]


def test_eval_and_compare(run, tiny, tmp_path):
    _, result, _ = run
    ckpt = Path(result["checkpoint"].replace("checkpoint-6", "checkpoint-5"))
    data = EX / "eval.synthetic.jsonl"
    tuned = evaluate(tiny, data, ckpt, "cpu", "float32")
    base = evaluate(tiny, data, None, "cpu", "float32")
    write_reports(tmp_path, base, tuned)
    for f in ("comparison.md", "comparison.json", "tuned_metrics.json", "base_metrics.json"):
        assert (tmp_path / f).exists()
    m = tuned["metrics"]["overall"]
    assert 0 <= m["accuracy"] <= 1 and m["nll"] > 0 and "0.99" in m["auto_decide"]
    assert "auto-decide coverage" in compare_markdown(base, tuned)


def test_merge_roundtrip_with_cloudflare_loader(run, tiny, tmp_path):
    _, result, _ = run
    ckpt = Path(result["checkpoint"]).parent / "checkpoint-5"
    out = merge(tiny, ckpt, tmp_path / "merged", dtype="float32", head_dtype="float32")
    for f in ("config.json", "joint_head.safetensors", "joint_head_config.json", "joint_schema_model.py",
              "tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        assert (out / f).exists(), f
    assert list(out.glob("model*.safetensors"))
    # Cloudflare's original loader, from the merged dir, untouched
    jsm = load_jsm(release_dir=out)
    merged_model, processor = jsm.load_release_model(out, device="cpu", dtype=torch.float32)
    pre_model, tok, proc, jsm2 = build_eval_model(tiny, "cpu", torch.float32, ckpt)
    base_model, _, _, _ = build_eval_model(tiny, "cpu", torch.float32, None)
    max_diff, base_diff = 0.0, 0.0
    for rec in load_jsonl(EX / "eval.synthetic.jsonl")[:2]:
        enc, _ = encode_example(jsm, processor.tokenizer, None, rec, EX, 4096)
        batch = jsm.collate_records([enc], processor.tokenizer.pad_token_id, torch.device("cpu"))
        with torch.inference_mode():
            a = merged_model(batch)[0]
            b = pre_model(batch)[0]
            c = base_model(batch)[0]
        for x, y, z in zip(a, b, c):
            max_diff = max(max_diff, (x.float() - y.float()).abs().max().item())
            base_diff = max(base_diff, (z.float() - y.float()).abs().max().item())
    print(f"merge round-trip max |logit diff| = {max_diff:.3e} (tuned vs base diff = {base_diff:.3e})")
    (tmp_path.parent / "merge_diff.txt").write_text(f"{max_diff} {base_diff}")
    assert max_diff < 1e-3
    assert base_diff > 10 * max_diff  # training actually changed something
