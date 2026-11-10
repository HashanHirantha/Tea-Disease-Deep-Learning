"""Dataset preparation and validation. This module does not import TensorFlow."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw

SPLITS = ("train", "tune", "calibration", "test")
PUBLIC_CLASSES = (
    "algal_leaf_spot", "black_blight", "blister_blight",
    "gray_blight", "healthy", "spider_mites",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PUBLIC_METADATA = Path(__file__).resolve().parents[1] / "data" / "public_metadata"
FIELD_COLUMNS = (
    "image_path", "class_label", "label_status", "estate_id", "region",
    "plant_id", "leaf_id", "session_id", "device_name", "capture_date",
)
MANIFEST_COLUMNS = FIELD_COLUMNS + (
    "split", "group_id", "base_image_id", "is_augmented",
    "acquisition_condition", "protocol",
)
DEMO_CLASSES = ("pattern_dots", "pattern_rings", "pattern_stripes")


def read_csv(path, required=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate column names.")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}.")
        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"{path}:{reader.line_num}: too many columns.")
            cleaned = {k: (v or "").strip() for k, v in row.items()}
            if any(cleaned.values()):
                rows.append(cleaned)
        if not rows:
            raise ValueError(f"{path}: no records.")
        return rows


def write_csv(path, rows, fields=MANIFEST_COLUMNS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_key(label):
    label = re.sub(r"[\s-]+", "_", label.strip().lower())
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", label):
        raise ValueError(f"Invalid class label: {label!r}. Use words and underscores.")
    return label


def group_key(*parts):
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))

def filename_key(value):
    """Raw .jpeg names and expanded .jpg copies can share one image ID."""
    p = PurePosixPath(value)
    suffix = p.suffix.lower()
    return p.stem.casefold() + (".jpg" if suffix == ".jpeg" else suffix)


def relative_path(value):
    value = value.replace("\\", "/")
    p = PurePosixPath(value)
    if not value or p.is_absolute() or not p.parts or ".." in p.parts or ":" in value:
        raise ValueError(f"Image path must be relative to the image root: {value!r}")
    return p.as_posix()


def image_path(root, value):
    root = Path(root).resolve()
    p = (root / relative_path(value)).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"Image path escapes the image root: {value!r}")
    if p.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Expected a JPG or PNG image: {value!r}")
    if not p.is_file():
        raise ValueError(f"Missing image: {p}")
    return p


def new_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output already exists and is not empty: {path}. Choose a new path.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def audit_rows(rows, image_root=None, verify_images=True, cache=None):
    """Validate groups, labels, coverage and, if available, exact image contents."""
    if not rows:
        raise ValueError("The manifest is empty.")
    protocols = {r["protocol"] for r in rows}
    if len(protocols) != 1 or not protocols <= {"public", "field", "synthetic"}:
        raise ValueError("Use one protocol per manifest: public, field, or synthetic.")
    protocol = next(iter(protocols))
    classes = sorted({label_key(r["class_label"]) for r in rows})
    if len(classes) < 2:
        raise ValueError("At least two classes are required.")
    counts = {split: Counter() for split in SPLITS}
    group_split, base_split, base_class = {}, {}, {}
    originals = Counter()
    paths, leaf_ids = set(), set()
    estates = {split: set() for split in SPLITS}
    sessions = {split: set() for split in SPLITS}
    hashes, duplicate_count = {}, 0
    cache = {} if cache is None else cache
    for r in rows:
        split, label = r["split"], r["class_label"]
        if split not in SPLITS or label != label_key(label):
            raise ValueError(f"Invalid split or noncanonical label: {split!r}, {label!r}.")
        path = relative_path(r["image_path"])
        if path.casefold() in paths:
            raise ValueError(f"Duplicate image path: {path}")
        paths.add(path.casefold())
        base, group = r["base_image_id"], r["group_id"]
        if not base or not group:
            raise ValueError(f"Missing image family or grouping ID: {path}")
        if group_split.setdefault(group, split) != split:
            raise ValueError(f"Group leakage: {group} occurs in multiple splits.")
        if base_split.setdefault(base, split) != split:
            raise ValueError(f"Image-family leakage: {base} occurs in multiple splits.")
        if base_class.setdefault(base, label) != label:
            raise ValueError(f"Conflicting labels for image family {base}.")
        if r["is_augmented"] not in {"0", "1"}:
            raise ValueError(f"Invalid is_augmented for {path}.")
        if r["is_augmented"] == "1":
            if split != "train" or protocol != "public":
                raise ValueError("Packaged augmented images may only occur in public training.")
        else:
            originals[base] += 1
        if protocol != "public":
            for key in ("estate_id", "region", "plant_id", "leaf_id", "session_id"):
                if not r[key]:
                    raise ValueError(f"{path}: {key} is required for field grouping.")
            if r["label_status"] != "expert_confirmed":
                raise ValueError(f"{path}: only expert_confirmed records are eligible.")
            plant = group_key(r["estate_id"], r["plant_id"])
            if group != plant:
                raise ValueError(f"Invalid plant group for {path}. Regenerate this manifest.")
            leaf = (r["estate_id"], r["plant_id"], r["leaf_id"])
            if leaf in leaf_ids:
                raise ValueError(f"Use one primary image per physical leaf: {leaf}.")
            leaf_ids.add(leaf)
            estates[split].add(r["estate_id"])
            sessions[split].add((r["estate_id"], r["session_id"]))
        elif group != base or r["estate_id"]:
            raise ValueError("Public records must use image families, without estate identifiers.")
        counts[split][label] += 1
        if image_root is not None:
            p = image_path(image_root, path)
            if verify_images:
                if p not in cache:
                    try:
                        with Image.open(p) as img:
                            if img.format not in {"JPEG", "PNG"}:
                                raise ValueError("Unsupported image format.")
                            img.verify()
                        cache[p] = sha256(p)
                    except (OSError, ValueError) as exc:
                        raise ValueError(f"Unreadable image {p}: {exc}") from exc
                fingerprint = cache[p]
                previous = hashes.get(fingerprint)
                if previous and previous[0] != split:
                    raise ValueError(f"Identical image bytes across splits: {previous[1]} and {path}")
                if previous:
                    if previous[2] != label:
                        raise ValueError(f"Identical image bytes have conflicting labels: {previous[1]} and {path}")
                    duplicate_count += 1
                hashes[fingerprint] = (split, path, label)
    if any(originals[base] != 1 for base in base_split):
        raise ValueError("Each image family must have exactly one original image.")
    for split in SPLITS:
        absent = set(classes) - set(counts[split])
        if absent:
            raise ValueError(f"{split}: missing class support for {', '.join(sorted(absent))}. "
                             "Collect more expert-confirmed leaves or revise the common classes.")
    if protocol != "public":
        source_estates = set.union(*(estates[s] for s in SPLITS[:-1]))
        if len(estates["test"]) != 1 or source_estates & estates["test"]:
            raise ValueError("Field test must contain exactly one estate absent from every source split.")
    if protocol == "public" and classes != list(PUBLIC_CLASSES):
        raise ValueError("The public protocol expects the six SLTeaLeaf classes.")
    return {
        "protocol": protocol, "classes": classes,
        "counts": {s: dict(sorted(counts[s].items())) for s in SPLITS},
        "split_sizes": {s: sum(counts[s].values()) for s in SPLITS},
        "estate_ids": {s: sorted(estates[s]) for s in SPLITS},
        "groups": len(group_split), "image_families": len(base_split),
        "paths_checked": image_root is not None,
        "image_content_checked": image_root is not None and verify_images,
        "extra_identical_files_within_splits": duplicate_count,
        "source_sessions_shared_across_splits": sum(
            len(sessions[a] & sessions[b]) for i, a in enumerate(SPLITS[:-1])
            for b in SPLITS[i + 1:-1]
        ),
    }


def load_manifest(path, root=None, verify_images=False):
    rows = read_csv(path, MANIFEST_COLUMNS)
    return rows, audit_rows(rows, root, verify_images)


def public_records(metadata=PUBLIC_METADATA, packaged_aug=False, seed=42):
    """Reconstruct the supplied base-ID splits; divide only the validation originals."""
    meta = Path(metadata)
    required = ("file_name", "relative_path", "class_folder", "base_image_id", "is_augmented")
    raw = read_csv(meta / "SLTeaLeaf_metadata_raw.csv", required)
    expanded = read_csv(meta / "SLTeaLeaf_metadata_augmented.csv", required) if packaged_aug else []
    index = {}
    for row in raw + [r for r in expanded if r["is_augmented"] == "1"]:
        key = (filename_key(row["file_name"]), label_key(row["class_folder"]))
        if key in index:
            raise ValueError(f"Duplicate public metadata key: {key}.")
        index[key] = row
    inputs = (
        ("train", "train_original_plus_augmented_baseid_split.csv"),
        ("tune", "validation_original_only_baseid_split.csv"),
        ("test", "test_original_only_baseid_split.csv"),
    )
    records = []
    for split, file in inputs:
        for row in read_csv(meta / file, ("image_path", "class_label", "base_id", "is_augmented")):
            aug = row["is_augmented"]
            if aug not in {"0", "1"} or (split != "train" and aug == "1"):
                raise ValueError(f"Invalid augmentation flag in {file}.")
            if aug == "1" and not packaged_aug:
                continue
            name = PurePosixPath(row["image_path"].replace("\\", "/")).name
            label = label_key(row["class_label"])
            source = index.get((filename_key(name), label))
            if not source or source["base_image_id"] != row["base_id"] or source["is_augmented"] != aug:
                raise ValueError(f"Split and metadata disagree for {name} in {file}.")
            rec = dict.fromkeys(MANIFEST_COLUMNS, "")
            rec.update(image_path=relative_path(source["relative_path"]),
                       class_label=label, label_status="published_dataset",
                       split=split, group_id=row["base_id"], base_image_id=row["base_id"],
                       is_augmented=aug, protocol="public",
                       acquisition_condition=source.get("acquisition_condition", ""),
                       device_name=source.get("device_name", ""))
            records.append(rec)
    if {r["base_image_id"] for r in records if r["is_augmented"] == "0"} != {
            r["base_image_id"] for r in raw}:
        raise ValueError("The supplied splits do not cover exactly the public originals.")
    rng = np.random.default_rng(seed)
    for label in PUBLIC_CLASSES:
        val = sorted((r for r in records if r["split"] == "tune" and r["class_label"] == label),
                     key=lambda r: r["base_image_id"])
        if len(val) < 2:
            raise ValueError(f"Not enough validation examples for {label}.")
        for i in rng.permutation(len(val))[: len(val) // 2]:
            val[i]["split"] = "calibration"
    return sorted(records, key=lambda r: (SPLITS.index(r["split"]), r["image_path"]))


def resolve_public_paths(rows, root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Image root does not exist: {root}")
    by_name = defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file() and p.resolve().is_relative_to(root):
            by_name[filename_key(p.name)].append(p)
    missing = []
    for row in rows:
        expected = root / row["image_path"]
        if expected.is_file() and expected.resolve().is_relative_to(root):
            continue
        candidates = sorted(by_name.get(filename_key(expected.name), []))
        if not candidates:
            missing.append(expected.name)
            continue
        if len(candidates) > 1:
            prefer = {"raw", "sltea-raw", "original", "originals"} if row["is_augmented"] == "0" else {"augmented", "sltea-aug"}
            preferred = [p for p in candidates if prefer & {x.lower() for x in p.relative_to(root).parts[:-1]}]
            candidates = preferred or candidates
        if len(candidates) > 1 and len({sha256(p) for p in candidates}) > 1:
            raise ValueError(f"Ambiguous image {expected.name}. Point --image-root to the intended image folders.")
        row["image_path"] = candidates[0].relative_to(root).as_posix()
    if missing:
        raise ValueError(f"{len(missing)} images missing (examples: {', '.join(missing[:5])}). "
                         "Extract the SLTeaLeaf v5 images and pass that folder as --image-root. "
                         "The metadata CSVs do not contain images.")
    return rows


def prepare_public(root, output, metadata=PUBLIC_METADATA, packaged_aug=False, seed=42):
    rows = resolve_public_paths(public_records(metadata, packaged_aug, seed), root)
    report = audit_rows(rows, root)
    out = new_directory(output)
    write_csv(out / "public.csv", rows)
    write_json(out / "audit.json", {**report, "split_seed": seed,
                                   "packaged_augmentations": packaged_aug})
    return report


def field_records(annotations, classes, protocol="field"):
    chosen = sorted(set(map(label_key, classes)))
    if len(chosen) < 2:
        raise ValueError("Specify at least two distinct, expert-confirmed classes.")
    records, skipped = [], Counter()
    for source in read_csv(annotations, FIELD_COLUMNS):
        if source["label_status"] != "expert_confirmed":
            skipped["not_expert_confirmed"] += 1
            continue
        label = label_key(source["class_label"])
        if label not in chosen:
            skipped["outside_selected_classes"] += 1
            continue
        for key in ("estate_id", "plant_id", "leaf_id", "session_id"):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", source[key]):
                raise ValueError(f"{source['image_path']}: use letters, digits, _ or - for {key}.")
        for key in ("region", "device_name", "capture_date"):
            if not source[key]:
                raise ValueError(f"{source['image_path']}: {key} is required.")
        rec = dict.fromkeys(MANIFEST_COLUMNS, "")
        rec.update({key: source[key] for key in FIELD_COLUMNS})
        rec.update(image_path=relative_path(source["image_path"]), class_label=label,
                   group_id=group_key(source["estate_id"], source["plant_id"]),
                   base_image_id=group_key(source["estate_id"], source["plant_id"], source["leaf_id"]),
                   protocol=protocol, is_augmented="0", acquisition_condition="field")
        records.append(rec)
    if {r["class_label"] for r in records} != set(chosen):
        raise ValueError("Some selected classes have no expert-confirmed field images.")
    return sorted(records, key=lambda r: r["image_path"]), dict(skipped)


def split_source(rows, classes, seed, attempts=1600):
    """Select a plant-grouped split using source labels only, never model scores."""
    groups = sorted({r["group_id"] for r in rows})
    if len(groups) < 3:
        raise ValueError("Need at least three source plant groups.")
    g_idx = {g: i for i, g in enumerate(groups)}
    c_idx = {c: i for i, c in enumerate(classes)}
    es = sorted({r["estate_id"] for r in rows})
    e_idx = {e: i for i, e in enumerate(es)}
    per_class = np.zeros((len(groups), len(classes)), dtype=np.int32)
    per_estate = np.zeros((len(groups), len(es)), dtype=np.int32)
    for row in rows:
        idx = g_idx[row["group_id"]]
        per_class[idx, c_idx[row["class_label"]]] += 1
        per_estate[idx, e_idx[row["estate_id"]]] += 1
    if np.any((per_class > 0).sum(axis=0) < 3):
        raise ValueError("Every source class needs leaves from at least three different bushes "
                         "to separate training, tuning and calibration.")
    rng = np.random.default_rng(seed)
    n = max(1, min(round(0.15 * len(groups)), (len(groups) - 1) // 2))
    targets = np.array([0.70, 0.15, 0.15])[:, None]
    best = None
    for _ in range(attempts):
        perm = rng.permutation(len(groups))
        pieces = (perm[2 * n:], perm[:n], perm[n:2 * n])
        counts = np.stack([per_class[g].sum(axis=0) for g in pieces])
        if not np.all(counts):
            continue
        ec = np.stack([per_estate[g].sum(axis=0) for g in pieces])
        cost = (np.abs(counts / counts.sum(axis=0) - targets).mean() +
                0.25 * np.abs(ec / ec.sum(axis=0) - targets).mean())
        if best is None or cost < best[0]:
            best = (float(cost), pieces)
    if best is None:
        raise ValueError("Could not find a grouped 70/15/15 source split with every class. "
                         "Collect more independent bushes per class before modelling.")
    return {groups[i]: split for split, piece in zip(SPLITS[:-1], best[1]) for i in piece}


def prepare_field(annotations, root, output, classes, seed=42, holdout=None, protocol="field"):
    rows, skipped = field_records(annotations, classes, protocol)
    estates = sorted({r["estate_id"] for r in rows})
    if len(estates) < 3:
        raise ValueError("Use at least three documented estates for this grouped workflow.")
    if holdout and holdout not in estates:
        raise ValueError(f"Unknown held-out estate {holdout!r}. Available: {', '.join(estates)}.")
    labels = sorted({r["class_label"] for r in rows})
    folds, reports, cache = {}, {}, {}
    for i, estate in enumerate(estates):
        if holdout and holdout != estate:
            continue
        source = [r for r in rows if r["estate_id"] != estate]
        test_labels = {r["class_label"] for r in rows if r["estate_id"] == estate}
        if test_labels != set(labels):
            raise ValueError(f"{estate}: missing test classes: {', '.join(sorted(set(labels) - test_labels))}. "
                             "Select shared classes based on the pilot; do not invent examples.")
        groups = split_source(source, labels, seed + i)
        fold = [{**r, "split": "test" if r["estate_id"] == estate else groups[r["group_id"]]} for r in rows]
        reports[estate] = audit_rows(fold, root, cache=cache)
        folds[estate] = fold
    out = new_directory(output)
    for estate, fold in folds.items():
        write_csv(out / f"fold_{estate}.csv", fold)
    report = {"protocol": protocol, "classes": labels, "split_seed": seed,
              "skipped": skipped, "annotation_sha256": sha256(annotations), "folds": reports}
    write_json(out / "audit.json", report)
    return report


def make_demo(output, seed=42):
    """Generate clearly synthetic patterns; their scores are software checks only."""
    out = new_directory(output)
    rng = np.random.default_rng(seed)
    annotations = []
    for site in range(1, 4):
        estate = f"S{site:02d}"
        for ci, label in enumerate(DEMO_CLASSES):
            for plant in range(8):
                name = f"{estate}_{ci}_{plant:02d}.png"
                relative = f"{estate}/{name}"
                path = out / "images" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                base = np.clip(rng.normal(32 + site * 17, 8, (80, 80, 3)), 0, 255).astype("uint8")
                pic = Image.fromarray(base)
                draw = ImageDraw.Draw(pic)
                d = int(rng.integers(-8, 9))
                color = (210, 120 + plant * 4, 80)
                if ci == 0:
                    for x, y in [(20, 20), (50, 25), (35, 52)]:
                        draw.ellipse((x + d, y, x + d + 9, y + 9), fill=color)
                elif ci == 1:
                    draw.ellipse((14 + d, 15, 59 + d, 62), outline=color, width=6)
                else:
                    for x in [19, 35, 51]:
                        draw.line((x + d, 13, x + d, 66), fill=color, width=5)
                pic.save(path)
                annotations.append(dict(image_path=relative, class_label=label, label_status="expert_confirmed",
                                        estate_id=estate, region=f"synthetic_region_{site}",
                                        plant_id=f"P{ci}_{plant}", leaf_id="L1", session_id=f"S{plant % 2}",
                                        device_name="synthetic_generator", capture_date="synthetic"))
    write_csv(out / "annotations.csv", annotations, FIELD_COLUMNS)
    return prepare_field(out / "annotations.csv", out / "images", out / "splits", DEMO_CLASSES,
                         seed=seed, protocol="synthetic")
