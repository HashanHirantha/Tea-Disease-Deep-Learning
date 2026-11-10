"""Run with python -m tea_research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import (
    PUBLIC_METADATA, audit_rows, load_manifest, make_demo,
    prepare_field, prepare_public, public_records, write_json,
)
from .learning import ARCHITECTURES, train
from .workflow import (
    CONDITIONS, benchmark_host, calibration_stage, evaluate_stage,
    export_stage, predict_one, summarize,
)


def add_training(p):
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", choices=ARCHITECTURES, default="mobilenetv3small")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=25)
    p.add_argument("--finetune-layers", type=int, default=20)
    p.add_argument("--head-lr", type=float, default=0.001)
    p.add_argument("--finetune-lr", type=float, default=0.0001)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)


def parser():
    p = argparse.ArgumentParser(description="Tea leaf classification: public baseline and held-out estate research.")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("audit-public-metadata", help="Verify the provided CSVs without images.")
    a = sub.add_parser("prepare-public", help="Resolve public images into the provided base-ID splits.")
    for q in (m, a):
        q.add_argument("--metadata-dir", type=Path, default=PUBLIC_METADATA)
        q.add_argument("--packaged-augmentations", action="store_true")
        q.add_argument("--seed", type=int, default=42)
    a.add_argument("--image-root", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    f = sub.add_parser("prepare-field", help="Make audited leave-one-estate-out manifests.")
    f.add_argument("--annotations", required=True, type=Path)
    f.add_argument("--image-root", required=True, type=Path)
    f.add_argument("--output", required=True, type=Path)
    f.add_argument("--classes", required=True, help="Comma-separated expert-confirmed classes.")
    f.add_argument("--holdout", help="Prepare one estate first; omit for all estates.")
    f.add_argument("--seed", type=int, default=42)
    au = sub.add_parser("audit", help="Validate a prepared manifest and image files.")
    au.add_argument("--manifest", required=True, type=Path)
    au.add_argument("--image-root", type=Path)
    au.add_argument("--output", type=Path)
    d = sub.add_parser("make-demo", help="Make 72 synthetic pattern images; no real tea data.")
    d.add_argument("--output", required=True, type=Path)
    d.add_argument("--seed", type=int, default=42)
    t = sub.add_parser("train", help="Train and select a checkpoint on source data.")
    add_training(t)
    r = sub.add_parser("run", help="Train, export, calibrate, and evaluate a frozen protocol.")
    add_training(r)
    e = sub.add_parser("export", help="Export matched FP32 and full INT8 models.")
    c = sub.add_parser("calibrate", help="Fit confidence using the dedicated source split.")
    ev = sub.add_parser("evaluate", help="Evaluate the four fixed conditions on test data.")
    for q in (e, c, ev):
        q.add_argument("--run", required=True, type=Path)
        q.add_argument("--image-root", type=Path, help="Use if the image root moved after training.")
    for q in (e, r):
        q.add_argument("--representative", type=int, default=200)
    for q in (c, r):
        q.add_argument("--coverage", type=float, default=0.8)
    for q in (ev, r):
        q.add_argument("--bootstrap", type=int, default=500)
    pr = sub.add_parser("predict", help="Predict one image using an exported condition.")
    pr.add_argument("--condition", choices=CONDITIONS, default="int8_cal")
    b = sub.add_parser("benchmark-host", help="Compare runtime latency on this computer.")
    b.add_argument("--repeats", type=int, default=100)
    b.add_argument("--warmup", type=int, default=30)
    for q in (pr, b):
        q.add_argument("--run", required=True, type=Path)
        q.add_argument("--image", required=True, type=Path)
    for q in (c, ev, pr, b, r):
        q.add_argument("--threads", type=int, default=2)
    s = sub.add_parser("summarize", help="Summarize repeated seeds and estate-wise results.")
    s.add_argument("--runs", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    return p


def evaluate_summary(report):
    keys = ("n", "macro_f1", "brier", "ece", "coverage", "retained_error", "model_bytes")
    return {**{key: report[key] for key in ("protocol", "architecture", "test_estate", "synthetic_notice")},
            "conditions": {k: {m: v[m] for m in keys} for k, v in report["conditions"].items()}}


def main(argv=None):
    args = parser().parse_args(argv)
    cmd = args.command
    try:
        if getattr(args, "threads", 1) < 1 or getattr(args, "bootstrap", 0) < 0:
            raise ValueError("Use positive runtime threads and nonnegative bootstrap resamples.")
        if cmd == "audit-public-metadata":
            result = audit_rows(public_records(args.metadata_dir, args.packaged_augmentations, args.seed))
        elif cmd == "prepare-public":
            result = prepare_public(args.image_root, args.output, args.metadata_dir,
                                    args.packaged_augmentations, args.seed)
        elif cmd == "prepare-field":
            result = prepare_field(args.annotations, args.image_root, args.output,
                                   [s.strip() for s in args.classes.split(",") if s.strip()],
                                   args.seed, args.holdout)
            result = {"protocol": result["protocol"], "classes": result["classes"],
                      "skipped": result["skipped"],
                      "fold_sizes": {k: v["split_sizes"] for k, v in result["folds"].items()}}
        elif cmd == "audit":
            _, result = load_manifest(args.manifest, args.image_root, verify_images=True)
            if args.output:
                write_json(args.output, result)
        elif cmd == "make-demo":
            result = make_demo(args.output, args.seed)
            result = {"protocol": result["protocol"], "folds": sorted(result["folds"]),
                      "note": "Synthetic pattern images; do not report these scores as research."}
        elif cmd == "train":
            result = train(args)
        elif cmd == "run":
            train(args)
            export_stage(args.output, representative=args.representative)
            calibration_stage(args.output, coverage=args.coverage, threads=args.threads)
            result = evaluate_summary(evaluate_stage(args.output, bootstrap=args.bootstrap,
                                                      threads=args.threads))
        elif cmd == "export":
            ex = export_stage(args.run, args.image_root, args.representative)
            result = {"models": ex["models"], "representative_images": len(ex["representative_images"])}
        elif cmd == "calibrate":
            c = calibration_stage(args.run, args.image_root, args.coverage, args.threads)
            result = {k: c[k] for k in ("source_split", "source_count", "coverage_target", "conditions")}
        elif cmd == "evaluate":
            result = evaluate_summary(evaluate_stage(args.run, args.image_root, args.bootstrap, args.threads))
        elif cmd == "predict":
            result = predict_one(args.run, args.image, args.condition, args.threads)
        elif cmd == "benchmark-host":
            result = benchmark_host(args.run, args.image, args.repeats, args.warmup, args.threads)
        else:
            result = summarize(args.runs, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
