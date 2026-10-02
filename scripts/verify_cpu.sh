#!/usr/bin/env bash
# Full CPU verification: pytest + CLI smoke (make-tiny -> validate -> augment -> train -> resume -> eval -> merge -> reload)
set -u
cd "$(dirname "$0")/.."
export CLEF_FINETUNE_JSM=vendored
LOG=${LOG:-verify_cpu.log}
W=$(mktemp -d)
{
echo "== pytest"; timeout 900 .venv/bin/python -m pytest -q -s -p no:cacheprovider 2>&1 | grep -vE "Loading weights|Writing model|it/s\]"; echo "pytest exit=${PIPESTATUS[0]}"
echo "== cli make-tiny"; .venv/bin/clef-finetune make-tiny "$W/tiny" || echo FAIL
echo "== cli validate"; .venv/bin/clef-finetune validate examples/insurance-claims/train.synthetic.jsonl examples/compliance/train.synthetic.jsonl; echo "validate exit=$?"
echo "== cli augment"; .venv/bin/clef-finetune augment examples/insurance-claims/train.synthetic.jsonl --config configs/clef-flash-insurance.yaml --copies 3 -o "$W/aug.jsonl" && .venv/bin/clef-finetune validate "$W/aug.jsonl" | head -3
echo "== cli train"; .venv/bin/clef-finetune train configs/tiny-cpu.yaml --model "$W/tiny" --output-dir "$W/run" --device cpu 2>&1 | grep -E "LoRA targets|trainable|\[train\]|lora_targets|checkpoint\"|Error|Traceback"
echo "== cli resume"; .venv/bin/clef-finetune train configs/tiny-cpu.yaml --model "$W/tiny" --output-dir "$W/run" --device cpu --max-steps 6 --resume 2>&1 | grep -E "resumed|\[train\]|Error|Traceback"
echo "== cli eval"; .venv/bin/clef-finetune eval --model "$W/tiny" --data examples/insurance-claims/eval.synthetic.jsonl --checkpoint "$W/run/checkpoint-6" --compare-base --dtype float32 --device cpu --output-dir "$W/report" 2>&1 | grep -vE "Loading weights|it/s\]" | tail -8; ls "$W/report"; head -12 "$W/report/comparison.md"
echo "== cli merge"; .venv/bin/clef-finetune merge --model "$W/tiny" --checkpoint "$W/run/checkpoint-6" --output "$W/merged" 2>&1 | tail -1; ls "$W/merged"
echo "== reload merged with Cloudflare's own file from the merged dir (no clef_finetune import)"
cat > "$W/reload.py" <<EOF
import sys, torch
sys.path.insert(0, "$W/merged")
from joint_schema_model import load_release_model, systemone
m, p = load_release_model("$W/merged", device="cpu", dtype=torch.float32)
r = systemone(m, p, {"model": "clef", "state": "Kitchen fire, \$6,400 estimate.", "questions": {"fraud": {"type": "noul", "instructions": "Fraud red flags?"}, "d": {"type": "choice", "instructions": "Next step?", "criteria": {"approve": "pay", "deny": "decline"}}}})
print("systemone answers:", r["answers"])
EOF
.venv/bin/python "$W/reload.py" 2>&1 | grep -vE "Loading weights|it/s\]"
} > "$LOG" 2>&1
echo "workdir=$W" >> "$LOG"
echo done >> "$LOG"
