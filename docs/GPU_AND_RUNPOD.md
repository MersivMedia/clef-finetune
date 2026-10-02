# GPU sizing and RunPod how-to

**Status:** none of this has been run on a GPU yet. The sizes below are estimates from parameter counts, not measurements. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Sizing

| Model | Backbone | bf16 weights | Starting point for LoRA training |
|---|---|---|---|
| Clef-flash | Qwen3.5-9B | ~18 GB | one 80 GB H100 or A100 |
| Clef | Qwen3.8-27B | ~54 GB | one H200 (141 GB), or 2× 80 GB |

Cloudflare tested inference with torch 2.11 and transformers 5.10.2 on one H200 (from the model card).

How much memory training takes on top of the weights depends on:
- **Sequence length.** The biggest lever. `data.max_length` defaults to 8192. Insurance claim files with long notes push it up, so set `max_state_tokens` to cap the state.
- **Gradient checkpointing.** On by default; it trades compute for activation memory.
- **LoRA rank.** `r: 64` adapters on every text Linear layer are small next to the backbone. Cloudflare used rank 256 for their own post-training.
- **The head.** The joint schema head (about 125M parameters for the released config) trains in fp32 with AdamW, about 2 GB including optimizer state.

**QLoRA (4-bit) is UNTESTED.** Qwen3.5's linear-attention (Gated DeltaNet) layers have not been checked with bitsandbytes 4-bit. Don't plan around it.

## Start with Clef-flash

Prove the pipeline and the gain on Clef-flash first. Move to Clef only if the eval shows Clef-flash isn't good enough. A run on Clef costs roughly three times as much per step.

## RunPod how-to

1. **Network volume.** Create one (100 GB is enough for Clef-flash plus checkpoints, 200 GB for Clef) in the same datacenter as the GPU you want.
2. **CPU pod first, for setup.** Download weights and install packages on a cheap CPU pod attached to the volume, so you don't pay GPU rates for downloads:
   ```bash
   cd /workspace && git clone https://github.com/MersivMedia/clef-finetune && cd clef-finetune
   python -m venv /workspace/venv && . /workspace/venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -e .
   huggingface-cli download Cloudflare/clef-flash --revision 17f0b0ad64efb65d273590632833508766b2aae6 --local-dir /workspace/models/clef-flash
   ```
3. **GPU pod for the run.** Stop the CPU pod, then start an H100 80 GB pod (Secure Cloud) on the same volume:
   ```bash
   . /workspace/venv/bin/activate && cd /workspace/clef-finetune
   clef-finetune validate /workspace/data/train.jsonl /workspace/data/eval.jsonl
   clef-finetune train configs/clef-flash-insurance.yaml --model /workspace/models/clef-flash --output-dir /workspace/runs/ins-v1
   ```
   Checkpoints land on the volume every `save_every` steps. After an interruption, `--resume` continues from the latest one.
4. **Evaluate against the base model** (writes `comparison.md` and `comparison.json`):
   ```bash
   clef-finetune eval --model /workspace/models/clef-flash --data /workspace/data/eval.jsonl \
     --checkpoint /workspace/runs/ins-v1/checkpoint-500 --compare-base --output-dir /workspace/runs/ins-v1/eval
   ```
5. **Merge into a release directory**, which loads with Cloudflare's own `load_release_model`. Merging can run on the CPU pod, since it needs about 2× the bf16 weights in RAM:
   ```bash
   clef-finetune merge --model /workspace/models/clef-flash --checkpoint /workspace/runs/ins-v1/checkpoint-500 \
     --output /workspace/releases/clef-flash-insurance-v1
   ```
6. **Stop the GPU pod** as soon as the run finishes. A network volume bills while it exists, even with no pod attached.

## Cost control

- Do a 20-step dry run (`--max-steps 20`) and read the step time and peak memory before launching a full run.
- Run the base-model eval first. If zero-shot is already good enough, skip training.
