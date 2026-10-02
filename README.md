# clef-finetune

**Fine-tune Cloudflare's open-source Clef and Clef-flash decision models on your own labelled decisions, then ship them as a standard Clef release folder.**

Clef ([Cloudflare/clef](https://huggingface.co/Cloudflare/clef), 27B) and Clef-flash ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), 9B) are Apache-2.0 decision models. You send them a state and typed questions, and they return a probability for every allowed answer in one forward pass. Cloudflare released inference code only. This repo adds the training side, following the recipe they describe in their [launch post](https://blog.cloudflare.com/clef-decision-models): LoRA on the backbone, the joint schema head trained alongside it, and a label-smoothed cross-entropy + Brier loss.

> **Status:** the full pipeline is tested end to end on CPU against a tiny random model. **No GPU training run on real Clef weights has been done yet**, so there are no accuracy claims. See [docs/KNOWN_ISSUES.md](https://github.com/MersivMedia/clef-finetune/blob/main/docs/KNOWN_ISSUES.md).

## Features

- **LoRA that actually covers Qwen3.5.** Targets are found from `named_modules`, so the linear-attention (Gated DeltaNet) projections get adapters, not just `q_proj`/`v_proj`. The vision tower and `lm_head` are excluded, and that holds when the adapter is reloaded.
- **Joint schema head trained alongside**, in fp32 with its own learning rate, with an explicit dtype boundary to the bf16 backbone. It can also be frozen.
- **Cloudflare's loss recipe**: label-smoothed cross-entropy + λ·Brier for calibration, plus an optional ordinal (earth-mover's) term for score questions. The ordinal term is a supervised take on the adjacent-credit idea in RLCD, not RLCD itself.
- **Schema augmentation that remaps labels**: question shuffling, random question subsets, instruction paraphrases from YAML, choice option-id renaming (`approve` → `APPROVE` / `pay`), and description dropout. It accounts for Cloudflare's encoder sorting choice ids, and never shuffles score levels.
- **Data validator**: every label is checked against the legal options, with class balance reported and a warning on more than 90% majority.
- **Business-grade eval**: accuracy, macro-F1, NLL, Brier, ECE, and **auto-decide coverage at a target precision**: what share of cases the model can decide alone at, say, 99% precision, and at what confidence threshold. Reports base vs tuned as JSON + markdown.
- **Release-format export**: `merge` writes merged safetensors + `joint_head.safetensors` + tokenizer/processor + Cloudflare's original `joint_schema_model.py`. The result loads with Cloudflare's own `load_release_model`, unchanged.
- **Vertical starter kits**: [insurance claims](https://github.com/MersivMedia/clef-finetune/tree/main/examples/insurance-claims) (disposition, fraud red flags, severity, coverage line) and [compliance](https://github.com/MersivMedia/clef-finetune/tree/main/examples/compliance) (AML alert disposition, sanctions hit, policy violation, risk score). Both come with schemas, paraphrase templates and clearly labelled **synthetic** records.
- **Practical training**: batch size 1 + gradient accumulation, gradient checkpointing, bf16, seeded runs, checkpoint/resume, YAML configs.
- **Cloudflare's code is never modified.** `joint_schema_model.py` loads from your release folder or the Hub at a pinned revision. A byte-identical vendored copy (hash-checked in CI) is the offline fallback.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or /whl/cpu
pip install clef-finetune
```

The starter kits and example configs live in the repo, so for those (or to develop):

```bash
git clone https://github.com/MersivMedia/clef-finetune && cd clef-finetune
pip install -e ".[dev]"
```

Needs transformers ≥ 5.10.2 (for `Qwen3_5ForConditionalGeneration`).

## Usage

```bash
# 1. check your data (one System One request + labels per line, see docs/DATA_GUIDE.md)
clef-finetune validate data/train.jsonl data/eval.jsonl

# 2. preview augmentation
clef-finetune augment data/train.jsonl --config configs/clef-flash-insurance.yaml --copies 3 -o /tmp/aug.jsonl

# 3. measure the base model first: you may not need to train at all
clef-finetune eval --model Cloudflare/clef-flash --data data/eval.jsonl --output-dir reports/base

# 4. train (GPU), resume after interruption with --resume
clef-finetune train configs/clef-flash-insurance.yaml --output-dir runs/ins-v1

# 5. base vs tuned report
clef-finetune eval --model Cloudflare/clef-flash --data data/eval.jsonl \
  --checkpoint runs/ins-v1/checkpoint-500 --compare-base --output-dir reports/ins-v1

# 6. export a release folder that Cloudflare's loader (and a System One server) can load
clef-finetune merge --model Cloudflare/clef-flash --checkpoint runs/ins-v1/checkpoint-500 --output releases/ins-v1
```

Try the whole chain on a laptop CPU, no downloads, using a tiny random model:

```bash
bash scripts/verify_cpu.sh     # pytest + make-tiny -> validate -> augment -> train -> resume -> eval -> merge -> reload
```

### Data format

```json
{"id": "c1", "state": {"claim": {"description": "...", "estimate_usd": 6400}},
 "questions": {"disposition": {"type": "choice", "instructions": "...", "criteria": {"approve": "...", "deny": "...", "escalate": "..."}},
               "fraud_indicators": {"type": "noul", "instructions": "..."},
               "severity": {"type": "score", "instructions": "...", "criteria": ["Minor", "Moderate", "Serious"]}},
 "labels": {"disposition": "escalate", "fraud_indicators": true, "severity": 2}}
```

### Config

Every setting, with defaults, is in [`configs/clef-flash-insurance.yaml`](https://github.com/MersivMedia/clef-finetune/blob/main/configs/clef-flash-insurance.yaml): model path and pinned revision, augmentation probabilities, LoRA rank/alpha/target suffixes, learning rates, loss weights, checkpointing.

## Docs

- [Data and labelling guide](https://github.com/MersivMedia/clef-finetune/blob/main/docs/DATA_GUIDE.md): what to collect, how many examples, consensus labels, hold-out sets
- [GPU sizing and RunPod how-to](https://github.com/MersivMedia/clef-finetune/blob/main/docs/GPU_AND_RUNPOD.md)
- [Known issues and verification status](https://github.com/MersivMedia/clef-finetune/blob/main/docs/KNOWN_ISSUES.md)

## License

Apache-2.0. Clef and Clef-flash weights and `joint_schema_model.py` are © Cloudflare, Inc., also Apache-2.0 (see [`src/clef_finetune/_vendor/NOTICE`](https://github.com/MersivMedia/clef-finetune/tree/main/src/clef_finetune/_vendor/NOTICE)). This project is not affiliated with Cloudflare.
