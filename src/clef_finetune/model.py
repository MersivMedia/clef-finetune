"""Model assembly: backbone + LoRA + JointSchemaHead with an explicit dtype boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from .jsm import PINNED_REVISIONS, load_jsm

# Linear-layer name suffixes inside Qwen3.5 text decoder layers.
# full_attention: q_proj k_proj v_proj o_proj ; linear_attention (Gated DeltaNet):
# in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj ; MLP: gate_proj up_proj down_proj
DEFAULT_EXCLUDE = (r"(^|\.)visual\.", r"(^|\.)lm_head$", r"(^|\.)mtp\.")


def resolve_model_dir(path_or_repo: str, revision: str | None = None) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, revision=revision or PINNED_REVISIONS.get(path_or_repo)))


def discover_lora_targets(
    model: torch.nn.Module,
    include_suffixes: list[str] | None = None,
    exclude_patterns: list[str] | tuple[str, ...] = DEFAULT_EXCLUDE,
    scope: str = "language_model",
) -> list[str]:
    """Return full module names of every nn.Linear in the text decoder.

    Discovered from ``named_modules`` rather than a hard-coded q/v list, because
    Qwen3.5's linear-attention layers use different names (in_proj_qkv, out_proj...).
    """
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if scope and f"{scope}." not in f"{name}." and not name.startswith(scope):
            continue
        if any(re.search(p, name) for p in exclude_patterns):
            continue
        if include_suffixes and name.split(".")[-1] not in include_suffixes:
            continue
        names.append(name)
    if not names:
        raise RuntimeError("no LoRA target modules found -- check lora.include_suffixes / scope")
    return names


def targets_regex(names: list[str]) -> str:
    """PEFT treats a str target_modules as a full-match regex on module names."""
    return r"^(?:" + "|".join(re.escape(n) for n in names) + r")$"


def summarize_targets(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in names:
        kind = ("linear_attn." if ".linear_attn." in n else "self_attn." if ".self_attn." in n else "mlp." if ".mlp." in n else "") + n.split(".")[-1]
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


class Fp32Head(torch.nn.Module):
    """Wrap Cloudflare's JointSchemaHead so it can stay fp32 while the backbone runs bf16.

    ``load_release_model`` casts the head to bf16 -- fine for inference, but training
    the head in bf16 loses small updates. We cast the incoming hidden states and the
    output-embedding weight to the head's dtype at the boundary.
    """

    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.inner = head

    @property
    def dtype(self) -> torch.dtype:
        return next(self.inner.parameters()).dtype

    def forward(self, hidden_states, input_ids, attention_mask, records, output_embedding_weight):
        dtype = self.dtype
        return self.inner(
            hidden_states.to(dtype),
            input_ids,
            attention_mask,
            records,
            output_embedding_weight.detach().to(dtype),
        )


def load_backbone(model_dir: Path, dtype: torch.dtype, device: str):
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device_map={"": device})
    model.config.use_cache = False
    return model


def load_head(jsm, model_dir: Path, dtype: torch.dtype = torch.float32, state_path: Path | None = None):
    from safetensors.torch import load_file

    cfg = json.loads((model_dir / "joint_head_config.json").read_text())
    head = jsm.JointSchemaHead(**cfg)
    state = load_file(str(state_path or model_dir / "joint_head.safetensors"))
    head.load_state_dict(state, strict=True)
    return head.to(dtype), cfg


def load_tokenizer_and_processor(model_dir: Path):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    return processor.tokenizer, processor


def build_training_model(cfg: dict[str, Any], device: str = "cpu"):
    """Return (clef_model, peft_backbone, head, tokenizer, processor, jsm, model_dir, targets)."""
    from peft import LoraConfig, get_peft_model

    mcfg = cfg["model"]
    model_dir = resolve_model_dir(mcfg["path"], mcfg.get("revision"))
    jsm = load_jsm(release_dir=model_dir)
    dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
    backbone = load_backbone(model_dir, dtype, device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    lcfg = cfg.get("lora", {})
    targets = discover_lora_targets(
        backbone,
        include_suffixes=lcfg.get("include_suffixes"),
        exclude_patterns=tuple(lcfg.get("exclude_patterns", DEFAULT_EXCLUDE)),
    )
    print(f"[clef-finetune] LoRA targets: {len(targets)} modules -> {summarize_targets(targets)}")
    lora = LoraConfig(
        r=lcfg.get("r", 64),
        lora_alpha=lcfg.get("alpha", 128),
        lora_dropout=lcfg.get("dropout", 0.0),
        # A list would be saved by PEFT as bare suffixes ("q_proj", "out_proj"...) and
        # re-matched against EVERY module on reload, vision tower included. A full-name
        # regex keeps reload/merge wrapping exactly the modules that were trained.
        target_modules=targets_regex(targets),
        bias="none",
    )
    peft_model = get_peft_model(backbone, lora)
    # LoRA weights in fp32 for stable optimisation even when the base is bf16
    for n, p in peft_model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if cfg.get("train", {}).get("gradient_checkpointing", True):
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        peft_model.enable_input_require_grads()

    head_dtype = getattr(torch, cfg.get("train", {}).get("head_dtype", "float32"))
    head, _ = load_head(jsm, model_dir, head_dtype)
    head = head.to(device)
    train_head = cfg.get("train", {}).get("train_head", True)
    for p in head.parameters():
        p.requires_grad_(train_head)
    tokenizer, processor = load_tokenizer_and_processor(model_dir)
    model = jsm.ClefModel(peft_model, Fp32Head(head))
    return model, peft_model, head, tokenizer, processor, jsm, model_dir, targets


def build_eval_model(model_dir: str | Path, device: str, dtype: torch.dtype, checkpoint: str | Path | None = None):
    """Base or tuned model for evaluation. ``checkpoint`` = adapter+head dir from training."""
    model_dir = resolve_model_dir(str(model_dir))
    jsm = load_jsm(release_dir=model_dir)
    backbone = load_backbone(model_dir, dtype, device)
    head_state = None
    if checkpoint is not None:
        from peft import PeftModel

        backbone = PeftModel.from_pretrained(backbone, Path(checkpoint) / "adapter")
        head_state = Path(checkpoint) / "joint_head.safetensors"
    head, _ = load_head(jsm, model_dir, torch.float32, head_state)
    tokenizer, processor = load_tokenizer_and_processor(model_dir)
    model = jsm.ClefModel(backbone, Fp32Head(head.to(device))).eval()
    return model, tokenizer, processor, jsm
