"""clef-finetune command line."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def cmd_validate(args) -> int:
    from .data import format_report, validate_file

    ok = True
    for path in args.files:
        report = validate_file(path, require_labels=not args.unlabelled)
        print(json.dumps(report, indent=2) if args.json else format_report(report))
        ok &= report["ok"]
    return 0 if ok else 1


def cmd_augment(args) -> int:
    from .augment import AugmentConfig, augment_record, load_templates
    from .data import load_jsonl, validate_record

    cfg = AugmentConfig()
    if args.config:
        from .train import load_config

        cfg = AugmentConfig.from_dict(load_config(args.config)["augment"])
    if args.templates:
        cfg.merge_templates(load_templates(args.templates))
    rng = random.Random(args.seed)
    out = sys.stdout if args.output == "-" else open(args.output, "w")
    n = 0
    for record in load_jsonl(args.input):
        for _ in range(args.copies):
            new = augment_record(record, cfg, rng)
            errs = validate_record(new)
            assert not errs, errs
            out.write(json.dumps(new, ensure_ascii=False) + "\n")
            n += 1
    if out is not sys.stdout:
        out.close()
        print(f"wrote {n} augmented records to {args.output}")
    return 0


def cmd_train(args) -> int:
    from .train import load_config, train

    overrides = {}
    if args.max_steps is not None:
        overrides.setdefault("train", {})["max_steps"] = args.max_steps
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.model:
        overrides.setdefault("model", {})["path"] = args.model
    cfg = load_config(args.config, overrides)
    result = train(cfg, device=args.device, resume=args.resume if args.resume != "latest" else True)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


def cmd_eval(args) -> int:
    from .evaluate import evaluate, write_reports

    tps = tuple(float(x) for x in args.target_precision.split(","))
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    tuned = evaluate(args.model, args.data, args.checkpoint, device, args.dtype, tps, args.max_length)
    base = None
    if args.checkpoint and args.compare_base:
        base = evaluate(args.model, args.data, None, device, args.dtype, tps, args.max_length)
    write_reports(Path(args.output_dir), base, tuned)
    print(json.dumps(tuned["metrics"]["overall"], indent=2))
    print(f"reports written to {args.output_dir}")
    return 0


def cmd_merge(args) -> int:
    from .export import merge

    out = merge(args.model, args.checkpoint, args.output, args.dtype, args.head_dtype)
    print(f"merged release written to {out}")
    return 0


def cmd_tiny(args) -> int:
    from .tiny import build_tiny_release

    print(build_tiny_release(args.output, seed=args.seed))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="clef-finetune", description="LoRA fine-tuning toolkit for Cloudflare Clef / Clef-flash")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="check labels are legal options; report class balance")
    s.add_argument("files", nargs="+")
    s.add_argument("--unlabelled", action="store_true", help="allow records without labels")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("augment", help="write schema-augmented copies of a JSONL file (preview / offline aug)")
    s.add_argument("input")
    s.add_argument("-o", "--output", default="-")
    s.add_argument("--config")
    s.add_argument("--templates")
    s.add_argument("--copies", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_augment)

    s = sub.add_parser("train", help="LoRA + head training from a YAML config")
    s.add_argument("config")
    s.add_argument("--device")
    s.add_argument("--model", help="override model.path")
    s.add_argument("--max-steps", type=int)
    s.add_argument("--output-dir")
    s.add_argument("--resume", nargs="?", const="latest", default=False, help="resume from latest (or given) checkpoint")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="metrics + auto-decide coverage; optional base-vs-tuned report")
    s.add_argument("--model", required=True, help="base release dir or HF repo id")
    s.add_argument("--data", required=True)
    s.add_argument("--checkpoint", help="adapter+head checkpoint dir (omit to eval the base)")
    s.add_argument("--compare-base", action="store_true")
    s.add_argument("--target-precision", default="0.99,0.95")
    s.add_argument("--output-dir", default="eval_report")
    s.add_argument("--device")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--max-length", type=int, default=8192)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("merge", help="merge LoRA into the backbone; write a release-format dir")
    s.add_argument("--model", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--head-dtype", default="bfloat16")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("make-tiny", help="build a tiny random Clef-shaped release for CPU smoke tests")
    s.add_argument("output")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_tiny)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
