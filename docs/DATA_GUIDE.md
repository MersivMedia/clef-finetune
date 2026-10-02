# Data and labelling guide

This is the part clients get wrong most often, and no amount of training can fix bad labels. Read it before you collect any data.

## 1. One record = one decision you already made

Each line of the JSONL file is a System One request (the exact body you would send to Clef in production) with the answers your team gave, added under `labels`:

```json
{"id": "CLM-2024-00417",
 "state": {"policy": {...}, "claim": {...}, "notes": "..."},
 "questions": {"disposition": {"type": "choice", "instructions": "...", "criteria": {"approve": "...", "deny": "...", "escalate": "...", "request_info": "..."}},
               "fraud_indicators": {"type": "noul", "instructions": "..."},
               "severity": {"type": "score", "instructions": "...", "criteria": ["Minor", "Moderate", "Serious", "Severe", "Catastrophic"]}},
 "labels": {"disposition": "escalate", "fraud_indicators": true, "severity": 2}}
```

| Question type | Label |
|---|---|
| `noul` | `true` / `false` |
| `choice` | the option id, exactly as written in `criteria` |
| `score` | the level index, `0` to `n-1` (position in the `criteria` list) |

Questions without labels are allowed. They still give the model context but add nothing to the loss.

Run `clef-finetune validate data.jsonl` before every training run. It rejects illegal labels and prints class balance.

## 2. Make `state` look like production

Train on exactly what the model will see at inference: the same fields, the same redaction, the same truncation. If production sends the adjuster's notes, training must too. If production strips PII, strip it from training data in the same way.

## 3. How many examples

These are planning numbers, not measured results. No GPU run has been done yet (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)).

| Goal | Rough starting point |
|---|---|
| First signal on whether tuning beats zero-shot | 200–500 labelled records |
| A pilot you would show a client | 1,000–3,000 |
| Rare classes (fraud, sanctions hits) | at least 50–100 positives each, or the model will learn to never predict them |

Measure first, though. Run `clef-finetune eval` on the **base** model with well-written `criteria`. Plenty of schemas work zero-shot, and then you don't need to tune at all.

## 4. Label quality beats label quantity

- **Get consensus labels for the eval set.** Have at least two reviewers label each eval record independently, and settle disagreements with a senior reviewer. Report how often reviewers agree: the model can't be expected to beat that ceiling.
- **Write the labelling rubric down.** It should be the same text that goes into `criteria`. If reviewers need a rule to decide, the model needs it too.
- **Keep labels from the time of the decision.** Later outcomes (such as "this claim later turned out to be fraud") are a different question. Label them as one if you want them.

## 5. Hold-outs that catch overfitting

Keep three separate eval sets:

1. **In-distribution:** same schema wording, new records.
2. **Held-out wording:** the same decisions, but with instructions and option ids rephrased in ways that never appeared in training or in `paraphrases.yaml`. This tells you whether the model learned the decision or memorised the prompt.
3. **General ability:** a slice of a public benchmark, to catch forgetting.

Split by **entity and time**, not by random row. All claims from one claimant, and everything after a cut-off date, go on the same side of the split.

## 6. Augmentation settings

`augment:` in the training config controls schema augmentation. Every transform remaps labels consistently.

| Setting | What it does |
|---|---|
| `shuffle_questions` | Randomises question order. The encoder keeps insertion order, so this matters. |
| `question_subset_prob` | Trains on random subsets of questions, so the model doesn't rely on one question always being present. |
| `paraphrase_prob` + `templates_file` | Swaps in instruction paraphrases from your YAML. `{instructions}` expands to the original text. |
| `rename_option_prob` | Renames choice option ids (`approve` → `APPROVE` / `pay`) and remaps the label. |
| `criteria_dropout_prob` | Blanks option descriptions so the model can't lean on them alone. |

Two facts about Cloudflare's encoder shape these options:
- **Choice option ids are sorted alphabetically before encoding.** Shuffling the `criteria` dict does nothing, and only renaming changes what the model sees.
- **Score levels are never shuffled.** Their order *is* the meaning.

## 7. Calibration is the product

The client's real question is usually: "what share of cases can we let the model decide alone?" `clef-finetune eval` reports **auto-decide coverage at a target precision**. That is the share of cases whose top confidence clears a threshold, chosen so precision on those cases stays at or above the target (for example 99%). Everything below the threshold goes to a human.

Measuring 99% precision honestly takes hundreds of eval cases per question. On small eval sets, treat the number as noise.
