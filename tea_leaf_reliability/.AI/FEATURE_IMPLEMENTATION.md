# Feature Implementation Guide

> [!NOTE]
> This document describes every implemented feature in the project, how it works internally, and where the relevant code lives. Use it as a map when modifying or extending the system.

---

## 1. CLI Entry Point

**What it does:** Single unified command-line interface for all operations via `python -m tea_research <command>`.

**How it works:**
- Uses `argparse` with subcommands. Each subcommand maps to a function in `data.py`, `learning.py`, or `workflow.py`.
- All outputs are JSON printed to stdout. Errors go to stderr with exit code 2.
- Shared arguments (e.g., `--run`, `--image-root`, `--threads`) are added to multiple subparsers via loops.

**Key code:**
- [\_\_main\_\_.py](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/__main__.py) — CLI parser and dispatcher
- 13 subcommands: `audit-public-metadata`, `prepare-public`, `prepare-field`, `audit`, `make-demo`, `train`, `run`, `export`, `calibrate`, `evaluate`, `predict`, `benchmark-host`, `summarize`

---

## 2. Data Management & Validation

### 2.1 Manifest System

**What it does:** CSV-based manifests track every image with its split assignment, group/family IDs, labels, and provenance metadata.

**How it works:**
- 14 columns defined in `MANIFEST_COLUMNS`: image path, label, estate/plant/leaf IDs, split, group ID, base image ID, augmentation flag, acquisition condition, protocol.
- `audit_rows()` performs comprehensive validation:
  - No group/family leakage across splits
  - Every class present in every split
  - One original per image family
  - Exact image byte deduplication via SHA-256
  - Field protocol enforces estate isolation in test split

**Key code:**
- [data.py § MANIFEST_COLUMNS](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L26-L29)
- [data.py § audit_rows()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L125-L233)

### 2.2 Public Dataset Preparation

**What it does:** Resolves the SLTeaLeaf v5 image archive against supplied metadata CSVs and builds audited train/tune/calibration/test splits.

**How it works:**
- Reads raw and augmented metadata CSVs from `data/public_metadata/`.
- Matches by filename key (handles `.jpg`/`.jpeg` alias transparently).
- Splits the validation set 50/50 into tuning and calibration by class using a seeded RNG.
- Resolves actual file paths via recursive search with disambiguation (prefers `raw`/`original` folders for originals).

**Key code:**
- [data.py § public_records()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L241-L290)
- [data.py § resolve_public_paths()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L293-L321)
- [data.py § prepare_public()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L324-L331)

### 2.3 Field Data Preparation

**What it does:** Creates leave-one-estate-out cross-validation folds from expert-confirmed field annotations.

**How it works:**
- Filters for `expert_confirmed` labels in the selected class list.
- For each fold, one estate is held out entirely for testing.
- Source estates are split by bush (plant group) into ~70/15/15 train/tune/calibration using a randomised search optimised for class and estate balance.
- Attempts 1,600 random permutations and selects the split with lowest imbalance cost.

**Key code:**
- [data.py § field_records()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L334-L362)
- [data.py § split_source()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L365-L401)
- [data.py § prepare_field()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L404-L431)

### 2.4 Demo / Synthetic Data Generator

**What it does:** Creates 72 synthetic pattern images (3 sites × 3 classes × 8 plants) for smoke-testing the full pipeline without real data.

**How it works:**
- Generates 80×80 RGB images with Gaussian noise background.
- Three pattern classes: dots (ellipses), rings (circle outline), stripes (vertical lines).
- Site index shifts the base colour; plant index shifts the pattern colour.
- Outputs annotations CSV, then calls `prepare_field()` with `protocol="synthetic"`.

**Key code:**
- [data.py § make_demo()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/data.py#L434-L467)

---

## 3. Model Training

### 3.1 Architecture & Model Building

**What it does:** Builds a transfer-learning classifier using MobileNetV3-Small or EfficientNet-B0 backbones.

**How it works:**
- Loads the Keras Application with `include_top=False, pooling="avg"`.
- Adds Dropout(0.2) → Dense(n_classes) logits head.
- Input: float32 RGB pixels [0–255], resized with bilinear padding to square.
- Backbone starts frozen; uses built-in input preprocessing (MobileNetV3 has `include_preprocessing=True`; EfficientNet has its own).

**Key code:**
- [learning.py § build_model()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L67-L82)
- [learning.py § ARCHITECTURES](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L16)

### 3.2 Two-Stage Training

**What it does:** Trains in two phases — head-only, then fine-tuning the last N backbone layers.

**How it works:**
- **Stage 1 (head):** Backbone frozen, train only the Dense + Dropout layers with Adam at `head_lr` (default 0.001).
- **Stage 2 (fine-tune):** Reload best checkpoint, unfreeze last `finetune_layers` (default 20) of backbone, keeping BatchNorm frozen. Train with Adam at `finetune_lr` (default 0.0001).
- Both stages use early stopping on tuning macro-F1 with configurable patience.
- Class weights computed as `n_total / (n_classes × n_class)` to handle imbalance.
- Full determinism: `set_random_seed()` + `enable_op_determinism()`.

**Key code:**
- [learning.py § train()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L85-L173)
- [learning.py § TuneF1 callback](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L114-L132)

### 3.3 Data Pipeline & Augmentation

**What it does:** Loads images, applies optional training augmentation, batches for Keras.

**How it works:**
- `tf.data.Dataset` from tensor slices (paths + labels).
- Training data: shuffled, with optional mild augmentation (horizontal flip, ±2° rotation, ±7% zoom, ±8% contrast).
- All augmentation is clipped to [0, 255].
- Deterministic threading and prefetch(1).
- Image preprocessing: decode RGB → bilinear resize-with-pad to square → float32.

**Key code:**
- [learning.py § dataset()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L44-L64)
- [learning.py § preprocess_image()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L36-L41)

---

## 4. Model Export

### 4.1 FP32 & INT8 TFLite Export

**What it does:** Converts the best Keras checkpoint into matched FP32 and full-INT8 TensorFlow Lite models.

**How it works:**
- Saves a `tf.keras.export.ExportArchive` with a `serve` endpoint.
- **FP32:** Standard TFLite conversion.
- **INT8:** Full integer quantization with:
  - Representative dataset from original training images, balanced across classes and estates.
  - `supported_ops = [TFLITE_BUILTINS_INT8]`
  - `inference_input_type = tf.int8`, `inference_output_type = tf.int8`
- Validation: verifies input/output dtypes match, no float tensors in INT8 model, logits are finite.
- Outputs: `fp32.tflite`, `int8.tflite`, `export.json` (tensor contracts, SHA-256, file sizes).

**Key code:**
- [learning.py § export_models()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L250-L313)
- [learning.py § representative_rows()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L176-L201)
- [learning.py § LiteRunner](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/learning.py#L204-L248)

---

## 5. Confidence Calibration

### 5.1 Temperature Scaling

**What it does:** Fits a single positive temperature parameter per precision (FP32/INT8) to calibrate prediction confidence.

**How it works:**
- Uses only the dedicated **calibration split** (never test data).
- Minimises NLL via `scipy.optimize.minimize_scalar` over log-temperature in [-3, 3].
- Falls back to T=1.0 if the fitted temperature doesn't improve NLL.
- Temperature divides logits before softmax: `softmax(logits / T)`.

**Key code:**
- [metrics.py § fit_temperature()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/metrics.py#L23-L30)
- [metrics.py § probabilities()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/metrics.py#L13-L15)
- [workflow.py § calibration_stage()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/workflow.py#L76-L99)

### 5.2 Selective Prediction (Coverage Threshold)

**What it does:** Sets a confidence cutoff so the model can abstain ("refer to expert") on uncertain images.

**How it works:**
- Target coverage (default 80%) means at least 80% of source calibration images should exceed the threshold.
- Threshold = the k-th largest confidence score where k = ⌈target × n⌉.
- At test time, images below the threshold are flagged `needs_expert_review = true`.

**Key code:**
- [metrics.py § coverage_threshold()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/metrics.py#L33-L40)

---

## 6. Evaluation

### 6.1 Four-Condition Evaluation

**What it does:** Evaluates the test set under four fixed conditions: `fp32_raw`, `fp32_cal`, `int8_raw`, `int8_cal`.

**How it works:**
- Collects logits from both TFLite models on all test images.
- Applies the condition's temperature and threshold.
- Reports per condition: accuracy, macro-F1, Brier score, ECE, coverage, retained error, per-class recall/precision/F1, confusion matrix, reliability diagram bins.
- Breaks down by acquisition condition if present.
- Outputs: `evaluation.json`, `predictions.csv`, `risk_coverage.csv`.

**Key code:**
- [workflow.py § evaluate_stage()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/workflow.py#L102-L162)
- [metrics.py § evaluate_probs()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/metrics.py#L61-L97)

### 6.2 Paired Bootstrap Intervals

**What it does:** Computes paired 95% bootstrap confidence intervals for the three primary contrasts.

**How it works:**
- Resamples **groups** (bush/family), not individual images, to respect clustering.
- Three contrasts:
  1. INT8 raw − FP32 raw (macro-F1 difference)
  2. FP32 calibrated − FP32 raw (Brier difference)
  3. INT8 calibrated − INT8 raw (Brier difference)
- Drops bootstrap samples where not all classes are represented.
- Reports observed difference, 95% CI, number of valid draws, and number of groups.

**Key code:**
- [metrics.py § paired_intervals()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/metrics.py#L113-L145)

### 6.3 Risk-Coverage Curve

**What it does:** Tabulates the trade-off between coverage (fraction of images answered) and retained error at every confidence threshold.

**How it works:**
- Sorts images by descending confidence.
- At each unique threshold, reports the fraction answered and cumulative error rate.
- Ties enter together (a threshold cannot split tied scores).
- **Descriptive only** — not used for tuning.

**Key code:**
- [metrics.py § risk_coverage()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/metrics.py#L100-L110)

---

## 7. Inference & Benchmarking

### 7.1 Single-Image Prediction

**What it does:** Classifies one leaf image using a specified condition (default: `int8_cal`).

**How it works:**
- Loads the appropriate TFLite model via `LiteRunner`.
- Preprocesses the image (decode → resize-with-pad → quantize if INT8).
- Applies the condition's temperature, computes softmax probabilities.
- Returns predicted class, confidence, referral flag, and full probability vector.

**Key code:**
- [workflow.py § predict_one()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/workflow.py#L165-L180)

### 7.2 Host CPU Benchmarking

**What it does:** Measures inference latency of both FP32 and INT8 models on the host CPU.

**How it works:**
- Pre-processes one image, then:
  - Warm-up runs (default 30)
  - Timed runs (default 100) using `time.perf_counter_ns()`
- Reports median and p95 latency in milliseconds.
- Excludes image decoding from measurement.
- Records platform, machine, processor, and thread count.

**Key code:**
- [workflow.py § benchmark_host()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/workflow.py#L183-L207)

---

## 8. Multi-Run Summarisation

**What it does:** Aggregates evaluation results across seeds and estates into summary tables.

**How it works:**
- Finds all `evaluation.json` files under a root directory.
- Validates: single protocol, consistent class set, no duplicate (architecture, estate, seed) combinations, consistent manifests per estate.
- Three output levels:
  1. **per_run.csv** — every individual run
  2. **per_estate.csv** — seeds averaged within each estate
  3. **equal_estate_means.csv** — estates given equal weight (the primary summary)

**Key code:**
- [workflow.py § summarize()](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/tea_research/workflow.py#L210-L263)

---

## 9. Field Experiment Runner

**What it does:** Orchestrates the full grid of folds × models × seeds as a batch job.

**How it works:**
- Enumerates all `fold_*.csv` manifests in the given directory.
- Validates consistent class sets across folds.
- Generates a task list: each fold × each model × each seed.
- `--dry-run` previews the plan; without it, runs sequentially via `subprocess`.
- Each task calls `python -m tea_research run` with the full pipeline.

**Key code:**
- [run_field_experiment.py](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/scripts/run_field_experiment.py)

---

## 10. Integrity & Reproducibility

| Feature | Implementation |
|---|---|
| **Seeded randomness** | `tf.keras.utils.set_random_seed()` + `enable_op_determinism()` + NumPy `default_rng(seed)` |
| **SHA-256 chain** | Manifest → checkpoint → export → calibration, each stage verifies the previous hash |
| **Manifest immutability** | Copied into the run directory at training time; subsequent stages verify its hash |
| **Deterministic data pipeline** | `tf.data.Options().deterministic = True`, fixed threadpool size |
| **Checkpoint integrity** | SHA-256 comparison before export to prevent using a modified checkpoint |

---

## 11. Docker Containerisation

**What it does:** Packages the full project into a reproducible container image.

**How it works:**
- Multi-stage build: `python:3.12-slim` → dependency layer → application layer.
- OS dependencies for Pillow/TFLite (libjpeg, libpng, libgl).
- Volumes for `/app/work` and `/app/runs` (bind-mount real data).
- Entrypoint: `python -m tea_research`.
- Health check: verifies CLI loads.

**Key code:**
- [Dockerfile](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/Dockerfile)
- [.dockerignore](file:///e:/Tea-Disease-Deep-Learning/tea_leaf_reliability/.dockerignore)

---

## 12. Full-Stack Diagnostic Web Application

**What it does:** Provides a live, interactive web application pairing a **FastAPI** Python backend with a **Vite + React** frontend for real-time tea leaf disease diagnosis.

**How it works:**
- **Backend (`web/backend/main.py`):**
  - REST endpoints for health check (`/api/health`), run discovery (`/api/runs`), execution conditions (`/api/conditions`), and inference (`/api/predict`).
  - Supports automatic synthetic demo model bootstrapping (`/api/setup-demo-model`).
  - Integrates Tea Research Institute of Sri Lanka (TRI) botanical and disease knowledge base (symptoms, fungal vectors, and agronomic management protocols).
- **Frontend (`web/frontend/`):**
  - Built with Vite, React 18, Tailwind CSS, and Lucide icons.
  - Drag-and-drop leaf image upload with live preview.
  - Interactive condition selector (INT8 Calibrated, INT8 Raw, FP32 Calibrated, FP32 Raw).
  - Diagnostic card showing top classification, confidence gauge, selective referral alert banner, and softmax probability distribution.
  - TRI management guidelines card (symptoms, environmental causes, chemical and cultural control).
- **Unified Launcher (`web/run_web.py`):**
  - Starts both the FastAPI Uvicorn server and Vite development server in one command and opens the browser.

