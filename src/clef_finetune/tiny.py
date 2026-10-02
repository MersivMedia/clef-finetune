"""Tiny randomly-initialised Clef-shaped release for CPU tests.

Builds a real ``Qwen3_5ForConditionalGeneration`` (hybrid linear/full attention,
tiny widths), a byte-level tokenizer with the Qwen special tokens, a
Qwen3VLProcessor, and a randomly-initialised JointSchemaHead -- saved in exactly
the release layout so ``load_release_model`` can load it. Nothing is downloaded.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from .jsm import VENDORED_PATH, load_jsm

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|image_pad|>", "<|video_pad|>", "<think>", "</think>",
]
TINY_HEAD = {"hidden_size": 64, "width": 32, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 64}


def _tokenizer(vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIAL_TOKENS)
    assert tok.get_vocab_size() <= vocab_size
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        extra_special_tokens={
            "image_token": "<|image_pad|>", "video_token": "<|video_pad|>",
            "vision_bos_token": "<|vision_start|>", "vision_eos_token": "<|vision_end|>",
        },
    )


def tiny_config(tokenizer, vocab_size: int = 512):
    from transformers import Qwen3_5Config

    ids = tokenizer.convert_tokens_to_ids
    text = dict(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        vocab_size=vocab_size, max_position_embeddings=4096, tie_word_embeddings=False,
        eos_token_id=ids("<|im_end|>"), pad_token_id=ids("<|endoftext|>"),
    )
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                  patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)
    return Qwen3_5Config(
        text_config=text, vision_config=vision, tie_word_embeddings=False,
        image_token_id=ids("<|image_pad|>"), video_token_id=ids("<|video_pad|>"),
        vision_start_token_id=ids("<|vision_start|>"), vision_end_token_id=ids("<|vision_end|>"),
    )


def build_tiny_release(out_dir: str | Path, seed: int = 0, vocab_size: int = 512) -> Path:
    from safetensors.torch import save_file
    from transformers import (Qwen2VLImageProcessor, Qwen3_5ForConditionalGeneration, Qwen3VLProcessor,
                              Qwen3VLVideoProcessor)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = _tokenizer(vocab_size)
    model = Qwen3_5ForConditionalGeneration(tiny_config(tokenizer, vocab_size)).to(torch.bfloat16)
    model.save_pretrained(out)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tokenizer,
        chat_template="{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}",
    )
    processor.save_pretrained(out)
    jsm = load_jsm()
    head = jsm.JointSchemaHead(**TINY_HEAD)
    # The released head's scalar gates are trained; zeros make the joint path tiny,
    # so give the random head non-trivial values to make tests meaningful.
    with torch.no_grad():
        head.residual_gate.fill_(0.5)
        head.joint_logit_scale.fill_(1.0)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items()}, str(out / "joint_head.safetensors"))
    (out / "joint_head_config.json").write_text(json.dumps(TINY_HEAD, indent=2))
    shutil.copy(VENDORED_PATH, out / "joint_schema_model.py")
    (out / "TINY_RANDOM_MODEL.txt").write_text("Randomly initialised test fixture. Not a real Clef model.\n")
    return out
