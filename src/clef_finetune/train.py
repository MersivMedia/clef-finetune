"""Training loop: batch size 1 + gradient accumulation, seeded, checkpoint/resume."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .augment import AugmentConfig, augment_record
from .data import load_jsonl, normalize_label, resolve_images, validate_record
from .losses import question_loss

DEFAULTS: dict[str, Any] = {
    "model": {"path": "Cloudflare/clef-flash", "revision": None, "dtype": "bfloat16"},
    "data": {"train": None, "eval": None, "max_length": 8192, "max_state_tokens": None},
    "augment": {},
    "lora": {"r": 64, "alpha": 128, "dropout": 0.0},
    "train": {
        "seed": 0,
        "max_steps": 1000,  # optimizer steps
        "grad_accum": 16,
        "lr": 1e-4,
        "head_lr": 5e-5,
        "weight_decay": 0.0,
        "warmup_steps": 20,
        "max_grad_norm": 1.0,
        "label_smoothing": 0.05,
        "brier_weight": 1.0,
        "ordinal_weight": 0.0,
        "train_head": True,
        "head_dtype": "float32",
        "gradient_checkpointing": True,
        "save_every": 100,
        "log_every": 1,
        "augment_train": True,
    },
    "output_dir": "runs/clef-finetune",
}


def deep_update(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in (new or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None, overrides: dict | None = None) -> dict[str, Any]:
    cfg = DEFAULTS
    if path:
        cfg = deep_update(cfg, yaml.safe_load(Path(path).read_text()) or {})
        base = Path(path).parent
        for key in ("train", "eval"):
            p = cfg["data"].get(key)
            if p and not Path(p).is_absolute() and not Path(p).exists() and (base / p).exists():
                cfg["data"][key] = str(base / p)
        tf = cfg["augment"].get("templates_file")
        if tf and not Path(tf).is_absolute() and not Path(tf).exists() and (base / tf).exists():
            cfg["augment"]["templates_file"] = str(base / tf)
    return deep_update(cfg, overrides or {})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def encode_example(jsm, tokenizer, processor, record, base_dir: Path, max_length: int, max_state_tokens=None):
    """Encode a labelled record; returns (encoded, targets list aligned to encoded.questions)."""
    rec = resolve_images(record, base_dir)
    encoded = jsm.encode_record(
        tokenizer,
        rec,
        max_length=max_length,
        max_state_tokens=max_state_tokens,
        processor=processor if rec.get("images") else None,
    )
    labels = record.get("labels") or {}
    targets = []
    for q in encoded.questions:
        if q.question_id in labels:
            option = normalize_label(record["questions"][q.question_id], labels[q.question_id])
            # option ids come from encode_record itself, so choice sorting is handled
            targets.append(q.option_ids.index(option))
        else:
            targets.append(None)
    return encoded, targets


def record_loss(logits, encoded, targets, record, tcfg) -> tuple[torch.Tensor | None, dict[str, float]]:
    losses, parts_sum = [], {}
    for q, ql, t in zip(encoded.questions, logits, targets):
        if t is None:
            continue
        qtype = record["questions"][q.question_id]["type"]
        loss, parts = question_loss(
            ql, t, qtype, tcfg["label_smoothing"], tcfg["brier_weight"], tcfg["ordinal_weight"]
        )
        losses.append(loss)
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v / max(1, sum(x is not None for x in targets))
    if not losses:
        return None, {}
    return torch.stack(losses).mean(), parts_sum


def save_checkpoint(out: Path, peft_model, head, optimizer, scheduler, step: int, rng_state: dict, cfg: dict):
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out / "adapter")
    save_file({k: v.detach().float().contiguous().cpu() for k, v in head.state_dict().items()}, str(out / "joint_head.safetensors"))
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state},
        out / "training_state.pt",
    )
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2, default=str))


def latest_checkpoint(output_dir: Path) -> Path | None:
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return ckpts[-1] if ckpts else None


def train(cfg: dict[str, Any], device: str | None = None, resume: bool | str = False) -> dict[str, Any]:
    from .model import build_training_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["train"]
    seed_everything(tcfg["seed"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(cfg["data"]["train"])
    records = load_jsonl(train_path)
    bad = [(i, e) for i, r in enumerate(records) for e in validate_record(r)]
    if bad:
        raise ValueError(f"training data has {len(bad)} errors, first: {bad[0]} -- run `clef-finetune validate`")
    aug_cfg = AugmentConfig.from_dict(cfg.get("augment"))

    model, peft_model, head, tokenizer, processor, jsm, model_dir, lora_targets = build_training_model(cfg, device)
    lora_params = [p for p in peft_model.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": tcfg["lr"]}]
    if head_params:
        groups.append({"params": head_params, "lr": tcfg["head_lr"]})
    n_trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"[clef-finetune] trainable params: lora={sum(p.numel() for p in lora_params):,} head={sum(p.numel() for p in head_params):,}")
    optimizer = torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])
    max_steps, warmup = tcfg["max_steps"], tcfg["warmup_steps"]

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rng = random.Random(tcfg["seed"])
    step = 0
    if resume:
        ckpt = Path(resume) if isinstance(resume, str) else latest_checkpoint(output_dir)
        if ckpt is not None:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            adapter_file = ckpt / "adapter" / "adapter_model.safetensors"
            set_peft_model_state_dict(peft_model, load_file(str(adapter_file)))
            head.load_state_dict({k: v.to(next(head.parameters()).dtype) for k, v in load_file(str(ckpt / "joint_head.safetensors")).items()})
            state = torch.load(ckpt / "training_state.pt", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = state["step"]
            rng.setstate(state["rng"]["python"])
            torch.set_rng_state(state["rng"]["torch"])
            print(f"[clef-finetune] resumed from {ckpt} at step {step}")

    model.train()
    history = []
    order: list[int] = []
    t0 = time.time()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss, accum_parts, n_used = 0.0, {}, 0
        for _ in range(tcfg["grad_accum"]):
            if not order:
                order = list(range(len(records)))
                rng.shuffle(order)
            record = records[order.pop()]
            if tcfg.get("augment_train", True):
                record = augment_record(record, aug_cfg, rng)
            encoded, targets = encode_example(
                jsm, tokenizer, processor, record, train_path.parent, cfg["data"]["max_length"], cfg["data"].get("max_state_tokens")
            )
            batch = jsm.collate_records([encoded], tokenizer.pad_token_id, torch.device(device))
            logits = model(batch)[0]
            loss, parts = record_loss(logits, encoded, targets, record, tcfg)
            if loss is None:
                continue
            (loss / tcfg["grad_accum"]).backward()
            accum_loss += loss.item()
            n_used += 1
            for k, v in parts.items():
                accum_parts[k] = accum_parts.get(k, 0.0) + v
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], tcfg["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        step += 1
        entry = {
            "step": step,
            "loss": accum_loss / max(1, n_used),
            **{k: v / max(1, n_used) for k, v in accum_parts.items()},
            "grad_norm": float(grad_norm),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(entry)
        if step % tcfg["log_every"] == 0:
            print("[train] " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in entry.items()}))
        with (output_dir / "train_log.jsonl").open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        if step % tcfg["save_every"] == 0 or step == max_steps:
            rng_state = {"python": rng.getstate(), "torch": torch.get_rng_state()}
            save_checkpoint(output_dir / f"checkpoint-{step}", peft_model, head, optimizer, scheduler, step, rng_state, cfg)
    final = latest_checkpoint(output_dir)
    return {"history": history, "checkpoint": str(final) if final else None, "model_dir": str(model_dir),
            "lora_targets": len(lora_targets), "trainable_params": n_trainable}
