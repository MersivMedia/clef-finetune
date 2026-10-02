# Known issues and status

## No GPU training run yet

**Nothing in this repo has been trained on a real Clef or Clef-flash model.** There are no accuracy, calibration or speed results for real weights. Every config under `configs/` other than `tiny-cpu.yaml` is a starting point, not a tested recipe. Results will be added here after the first GPU run.

## What has been verified (CPU, tiny random model)

All of this runs in CI and in `scripts/verify_cpu.sh` against a **randomly initialised, tiny `Qwen3_5ForConditionalGeneration`**. It's the real transformers class with hybrid linear and full attention, at about 4 layers and hidden size 64, plus a tiny `JointSchemaHead`. No real weights are downloaded.

| Check | Result on 2026-10-02 |
|---|---|
| Unit tests (validation, augmentation label remapping, losses, metrics, vendored-file hash) | pass |
| Loss falls on a repeated record (5 steps) | 2.008 → 1.861 → 1.628 → 1.316 → 1.091 |
| LoRA covers linear-attention, full-attention and MLP layers, and nothing in the vision tower or lm_head, *after reloading the saved adapter* | 31 modules wrapped, 0 visual |
| Resume from checkpoint continues at the next step | pass |
| Eval writes base-vs-tuned JSON + markdown with auto-decide coverage | pass |
| `merge` → release dir → Cloudflare's own `load_release_model` + `systemone`, imported from the merged dir | loads and answers |
| Merged vs pre-merge PEFT logits | max abs diff 5.1e-07 (base vs tuned differ by 2.16) |

Versions used: torch 2.11.0+cpu, transformers 5.18.0, peft 0.21.2, Python 3.12. Cloudflare tested Clef with transformers 5.10.2.

A tiny random model proves the plumbing works. It says nothing about how much fine-tuning improves real decisions.

## Not verified

- **Real-scale memory and speed.** The numbers in [GPU_AND_RUNPOD.md](GPU_AND_RUNPOD.md) are estimates.
- **QLoRA / 4-bit** with Qwen3.5's Gated DeltaNet (linear-attention) layers.
- **CUDA kernels.** On CPU, transformers falls back to the reference PyTorch versions of `causal_conv1d` and `chunk_gated_delta_rule`. On GPU, install `flash-linear-attention` and `causal-conv1d` for speed. Results should match, but this hasn't been checked.
- **Image and video records in training.** The data loader resolves image paths and passes them to Cloudflare's encoder, but no test trains on images yet.
- **Multi-GPU** (Clef 27B on 2× 80 GB). Training currently uses `device_map={"": device}` (one device). Sharding is not wired up.
- **Hub loading of `joint_schema_model.py`** at the pinned revision. CI forces the vendored copy (`CLEF_FINETUNE_JSM=vendored`). Both pinned revisions exist on the Hub, but downloading from them hasn't been exercised in a test.

## Things that will bite you (found while building this)

1. **Choice option ids are sorted.** Cloudflare's `encode_record` sorts choice option ids alphabetically, so logit index ≠ position in your `criteria` dict. This package always takes indices from the encoder's own `option_ids`. Do the same if you write your own tooling.
2. **PEFT saves target modules as bare suffixes.** A list of full module names gets saved in `adapter_config.json` as suffixes (`q_proj`, `out_proj`...). On reload these match the **vision tower** too (`visual.blocks.*.attn.proj`...). We save a full-name regex instead, and a test reloads the adapter to check exactly what gets wrapped.
3. **The head is bf16 in the release.** `load_release_model` casts the head to bf16, which is fine for inference but loses small updates in training. Training keeps the head and LoRA weights in fp32 and casts hidden states at the boundary (`Fp32Head`). `merge` writes the head back in bf16 by default (`--head-dtype`).
4. **Scalar gates.** The head's `residual_gate` / `joint_logit_scale` / `prior_logit_scale` start at zero in a fresh `JointSchemaHead`. That doesn't matter when you fine-tune the released head, which is trained already, but it makes a randomly initialised head almost all prior. The tiny test fixture sets them to non-zero values for that reason.
5. **The ordinal loss is not RLCD.** `ordinal_weight` adds an earth-mover's-distance term for score questions: a supervised take on the "partial credit for adjacent levels" idea in Cloudflare's RLCD stage. There are no rewards, no exact-record bonus and no reference penalty. Cloudflare hasn't released RLCD.
