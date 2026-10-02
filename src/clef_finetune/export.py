"""Export: adapter+head checkpoint -> full release-format directory.

Output layout matches Cloudflare's release so ``load_release_model(path)`` loads it
unchanged (interop contract with clef-rag):

    config.json, generation_config.json (if present), model*.safetensors (+ index)
    joint_head.safetensors, joint_head_config.json
    tokenizer.json, tokenizer_config.json, processor_config.json, chat_template.jinja
    joint_schema_model.py (Cloudflare's original, copied from the base release)
    LICENSE / NOTICE (if present in base), CLEF_FINETUNE.json (provenance)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

COPY_FILES = (
    "joint_head_config.json", "joint_schema_model.py", "tokenizer.json", "tokenizer_config.json",
    "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
    "chat_template.jinja", "chat_template.json", "special_tokens_map.json", "vocab.json", "merges.txt",
    "generation_config.json", "LICENSE", "NOTICE",
)


def merge(base_dir: str | Path, checkpoint: str | Path, out_dir: str | Path, dtype: str = "bfloat16",
          head_dtype: str = "bfloat16", max_shard_size: str = "5GB") -> Path:
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    from .jsm import VENDORED_PATH, sha256
    from .model import load_backbone, resolve_model_dir

    base = resolve_model_dir(str(base_dir))
    ckpt, out = Path(checkpoint), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    backbone = load_backbone(base, getattr(torch, dtype), "cpu")
    merged = PeftModel.from_pretrained(backbone, ckpt / "adapter").merge_and_unload()
    merged.save_pretrained(out, max_shard_size=max_shard_size)
    for name in COPY_FILES:
        if (base / name).is_file():
            shutil.copy(base / name, out / name)
    if not (out / "joint_schema_model.py").exists():
        shutil.copy(VENDORED_PATH, out / "joint_schema_model.py")
    head = load_file(str(ckpt / "joint_head.safetensors"))
    save_file({k: v.to(getattr(torch, head_dtype)).contiguous() for k, v in head.items()}, str(out / "joint_head.safetensors"))
    info = {
        "tool": "clef-finetune",
        "base_model": str(base_dir),
        "checkpoint": str(ckpt),
        "dtype": dtype,
        "head_dtype": head_dtype,
        "joint_schema_model_sha256": sha256(out / "joint_schema_model.py"),
    }
    cfg = ckpt / "train_config.json"
    if cfg.exists():
        info["train_config"] = json.loads(cfg.read_text())
    (out / "CLEF_FINETUNE.json").write_text(json.dumps(info, indent=2, default=str))
    return out
