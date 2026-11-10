"""Separate training, export, calibration, and evaluation stages."""

from __future__ import annotations

import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import image_path, load_manifest, new_directory, sha256, write_csv, write_json
from .learning import LiteRunner, export_models, preprocess_image
from .metrics import (
    coverage_threshold, evaluate_probs, fit_temperature, nll,
    paired_intervals, probabilities, risk_coverage,
)

CONDITIONS = ("fp32_raw", "fp32_cal", "int8_raw", "int8_cal")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_config(run):
    run = Path(run).resolve()
    return run, read_json(run / "training.json")


def data_context(run, root_override=None):
    run, config = run_config(run)
    root = Path(root_override or config["image_root"]).resolve()
    manifest = run / "manifest.csv"
    if sha256(manifest) != config["manifest_sha256"]:
        raise ValueError("The run manifest changed after training. Use the original manifest.")
    rows, audit = load_manifest(manifest, root)
    if audit["classes"] != config["classes"] or audit["protocol"] != config["protocol"]:
        raise ValueError("Manifest class order or protocol does not match training.")
    return run, config, root, rows


def checked_exports(run, config):
    export = read_json(run / "export.json")
    for key in ("manifest_sha256", "checkpoint_sha256", "classes"):
        if export[key] != config[key]:
            raise ValueError("Export does not match this trained checkpoint and split.")
    for kind in ("fp32", "int8"):
        if sha256(run / f"{kind}.tflite") != export["models"][kind]["sha256"]:
            raise ValueError(f"Saved {kind} model changed. Export again.")
    return export


def checked_calibration(run):
    calibration = read_json(run / "calibration.json")
    if calibration["export_sha256"] != sha256(run / "export.json"):
        raise ValueError("Calibration belongs to a different export. Run calibrate again.")
    return calibration


def export_stage(run, root=None, representative=200):
    run, config, root, rows = data_context(run, root)
    return export_models(run, config, rows, root, representative)


def collect_logits(kind, run, rows, root, size, threads):
    runner = LiteRunner(run / f"{kind}.tflite", threads=threads)
    result = np.stack([runner.predict(preprocess_image(str(image_path(root, r["image_path"])), size))
                       for r in rows])
    if not np.isfinite(result).all():
        raise ValueError(f"{kind} produced invalid logits.")
    return result


def calibration_stage(run, root=None, coverage=0.8, threads=2):
    run, config, root, rows = data_context(run, root)
    checked_exports(run, config)
    source = [r for r in rows if r["split"] == "calibration"]
    ids = {name: i for i, name in enumerate(config["classes"])}
    y = np.asarray([ids[r["class_label"]] for r in source], dtype=int)
    conditions = {}
    for kind in ("fp32", "int8"):
        z = collect_logits(kind, run, source, root, config["image_size"], threads)
        fitted = fit_temperature(z, y)
        for mode, temp in (("raw", 1.0), ("cal", fitted)):
            scores = probabilities(z, temp).max(axis=1)
            cutoff = coverage_threshold(scores, coverage)
            conditions[f"{kind}_{mode}"] = {
                "temperature": temp, "threshold": cutoff,
                "source_nll": nll(z, y, temp),
                "source_coverage": float((scores >= cutoff).mean()),
            }
    result = {"source_split": "calibration", "source_count": len(source),
              "source_paths": [r["image_path"] for r in source],
              "coverage_target": coverage, "conditions": conditions,
              "export_sha256": sha256(run / "export.json")}
    write_json(run / "calibration.json", result)
    return result


def evaluate_stage(run, root=None, bootstrap=500, threads=2):
    run, config, root, rows = data_context(run, root)
    export = checked_exports(run, config)
    calibration = checked_calibration(run)
    test = [r for r in rows if r["split"] == "test"]
    classes = config["classes"]
    ids = {label: i for i, label in enumerate(classes)}
    y = np.asarray([ids[r["class_label"]] for r in test], dtype=int)
    z = {kind: collect_logits(kind, run, test, root, config["image_size"], threads)
         for kind in ("fp32", "int8")}
    scores, outputs, prediction_rows, curve_rows = {}, {}, [], []
    for key in CONDITIONS:
        kind = key.split("_")[0]
        rule = calibration["conditions"][key]
        p = probabilities(z[kind], rule["temperature"])
        scores[key] = p
        outputs[key] = evaluate_probs(y, p, classes, rule["threshold"])
        outputs[key]["temperature"] = rule["temperature"]
        outputs[key]["model_bytes"] = export["models"][kind]["bytes"]
        for curve_row in risk_coverage(y, p):
            curve_rows.append({"condition": key, **curve_row})
        outputs[key]["by_acquisition_condition"] = {}
        for condition in sorted({r["acquisition_condition"] for r in test if r["acquisition_condition"]}):
            mask = np.asarray([r["acquisition_condition"] == condition for r in test])
            outputs[key]["by_acquisition_condition"][condition] = evaluate_probs(
                y[mask], p[mask], classes, rule["threshold"])
        for i, row in enumerate(test):
            conf = float(p[i].max())
            prediction_rows.append({
                "condition": key, "image_path": row["image_path"], "estate_id": row["estate_id"],
                "region": row["region"], "group_id": row["group_id"],
                "acquisition_condition": row["acquisition_condition"], "actual": classes[y[i]],
                "predicted": classes[int(p[i].argmax())], "confidence": conf,
                "accepted": int(conf >= rule["threshold"]),
                **{f"p_{c}": float(p[i, j]) for j, c in enumerate(classes)},
            })
    report = {
        "protocol": config["protocol"], "architecture": config["architecture"],
        "seed": config["seed"], "classes": classes, "test_estate": config["test_estate"],
        "manifest_sha256": config["manifest_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "export_sha256": sha256(run / "export.json"),
        "calibration_sha256": sha256(run / "calibration.json"),
        "runtime_threads": threads, "test_images": len(test),
        "bootstrap_resamples_requested": bootstrap,
        "bootstrap_unit": "image family" if config["protocol"] == "public" else "bush within estate",
        "synthetic_notice": "Software smoke test, not tea disease evidence."
        if config["protocol"] == "synthetic" else None,
        "conditions": outputs,
        "paired_differences": paired_intervals(y, [r["group_id"] for r in test], scores,
                                               n_samples=bootstrap, seed=config["seed"]),
        "int8_minus_fp32_recall": {
            c: outputs["int8_raw"]["per_class"][c]["recall"] -
            outputs["fp32_raw"]["per_class"][c]["recall"] for c in classes},
    }
    pred_fields = ("condition", "image_path", "estate_id", "region", "group_id",
                   "acquisition_condition", "actual", "predicted", "confidence", "accepted")
    write_csv(run / "predictions.csv", prediction_rows, pred_fields + tuple(f"p_{c}" for c in classes))
    write_csv(run / "risk_coverage.csv", curve_rows, ("condition", "threshold", "coverage", "retained_error"))
    write_json(run / "evaluation.json", report)
    return report


def predict_one(run, path, condition="int8_cal", threads=2):
    run, config = run_config(run)
    checked_exports(run, config)
    rule = checked_calibration(run)["conditions"][condition]
    kind = condition.split("_")[0]
    runner = LiteRunner(run / f"{kind}.tflite", threads=threads)
    z = runner.predict(preprocess_image(str(Path(path).resolve()), config["image_size"]))
    probs = probabilities(z[None, :], rule["temperature"])[0]
    top = int(probs.argmax())
    return {"protocol": config["protocol"], "condition": condition,
            "predicted_class": config["classes"][top], "confidence": float(probs[top]),
            "needs_expert_review": bool(probs[top] < rule["threshold"]),
            "threshold": rule["threshold"],
            "probabilities": dict(zip(config["classes"], map(float, probs))),
            "note": "Synthetic pattern prediction only." if config["protocol"] == "synthetic"
            else "Image screening prediction; unrepresented conditions may be misclassified."}


def benchmark_host(run, path, repeats=100, warmup=30, threads=2):
    if repeats < 1 or warmup < 0:
        raise ValueError("Use positive repeats and nonnegative warm-up runs.")
    run, config = run_config(run)
    export = checked_exports(run, config)
    x = preprocess_image(str(Path(path).resolve()), config["image_size"])
    out = {"device": "host CPU", "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "threads": threads, "warmup": warmup, "repeats": repeats,
           "measurement": "Set tensor, invoke, retrieve and dequantize logits; excludes image decode.",
           "models": {}}
    for kind in ("fp32", "int8"):
        runner = LiteRunner(run / f"{kind}.tflite", threads=threads)
        inp = runner.tensor(x)
        for _ in range(warmup):
            runner.infer_tensor(inp)
        times = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            runner.infer_tensor(inp)
            times.append((time.perf_counter_ns() - start) / 1e6)
        out["models"][kind] = {"bytes": export["models"][kind]["bytes"],
                               "median_ms": float(np.median(times)),
                               "p95_ms": float(np.percentile(times, 95))}
    write_json(run / "benchmark_host.json", out)
    return out


def summarize(runs_root, output):
    paths = sorted(Path(runs_root).rglob("evaluation.json"))
    if not paths:
        raise ValueError("No evaluation.json files found.")
    reports = [read_json(p) for p in paths]
    protocols = {r["protocol"] for r in reports}
    if len(protocols) != 1 or len({tuple(r["classes"]) for r in reports}) != 1:
        raise ValueError("Summarize only one protocol and one frozen class set at a time.")
    seen, manifests, rows = set(), {}, []
    metric_keys = ("accuracy", "macro_f1", "brier", "ece", "coverage", "retained_error")
    by_site = defaultdict(list)
    for r in reports:
        site = r["test_estate"] or "public_test"
        identity = (r["architecture"], site, r["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate architecture/estate/seed run: {identity}.")
        seen.add(identity)
        if manifests.setdefault(site, r["manifest_sha256"]) != r["manifest_sha256"]:
            raise ValueError(f"Inconsistent frozen manifest for {site}.")
        for c in CONDITIONS:
            record = {"architecture": r["architecture"], "test_estate": site,
                      "seed": r["seed"], "condition": c, "n": r["test_images"],
                      **{key: r["conditions"][c][key] for key in metric_keys}}
            rows.append(record)
            by_site[(r["architecture"], c, site)].append(record)

    def avg(v):
        usable = [x for x in v if x is not None]
        return float(np.mean(usable)) if usable else None

    site_rows = []
    for (arch, cond, site), items in sorted(by_site.items()):
        site_rows.append({"architecture": arch, "condition": cond, "test_estate": site,
                          "seeds": len(items),
                          **{key: avg([r[key] for r in items]) for key in metric_keys}})
    grouped = defaultdict(list)
    for r in site_rows:
        grouped[(r["architecture"], r["condition"])].append(r)
    combined = [{"architecture": arch, "condition": cond, "estates": len(items),
                 **{key: avg([r[key] for r in items]) for key in metric_keys}}
                for (arch, cond), items in sorted(grouped.items())]
    out = new_directory(output)
    write_csv(out / "per_run.csv", rows,
              ("architecture", "test_estate", "seed", "condition", "n") + metric_keys)
    write_csv(out / "per_estate.csv", site_rows,
              ("architecture", "test_estate", "condition", "seeds") + metric_keys)
    write_csv(out / "equal_estate_means.csv", combined,
              ("architecture", "condition", "estates") + metric_keys)
    result = {"protocol": next(iter(protocols)), "runs": len(reports),
              "note": "Average seeds within each estate first, then give each estate equal weight. "
                      "Compare architectures only over matching estates and seed sets.",
              "equal_estate_means": combined}
    write_json(out / "summary.json", result)
    return result
