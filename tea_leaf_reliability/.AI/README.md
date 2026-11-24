# Tea Leaf Reliability Research — AI Context

## Project Overview

This project implements the model and evaluation pipeline for **Reliable Tea Leaf Disease Detection Across Sri Lankan Plantations Using Lightweight Deep Learning**. It classifies a single photograph of a tea leaf into disease categories using two architectures: **MobileNetV3-Small** and **EfficientNet-B0**.

## Tech Stack

- **Language:** Python 3.11 / 3.12
- **Framework:** TensorFlow 2.19.1, Keras 3.11.3
- **Core Libraries:** NumPy 1.26.4, SciPy 1.17.0, Pillow 13.0.0
- **Deployment:** TensorFlow Lite (FP32 + full INT8 quantization)
- **Containerisation:** Docker (CPU; GPU variant available)

## Project Structure

```
tea_leaf_reliability/
├── tea_research/              # Main Python package
│   ├── __init__.py            # Package init, version
│   ├── __main__.py            # CLI entry point (python -m tea_research)
│   ├── data.py                # Data loading, manifest prep, demo generation
│   ├── learning.py            # Model architectures, training loop
│   ├── metrics.py             # Evaluation metrics, calibration
│   └── workflow.py            # Export, calibrate, evaluate, predict, benchmark
├── scripts/
│   └── run_field_experiment.py  # Orchestrates multi-fold field experiments
├── tests/
│   └── test_project.py        # Unit / integration tests
├── data/
│   ├── public_metadata/       # Pre-supplied CSV splits (no images)
│   └── field_annotations_template.csv
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── Tea_Leaf_Training_Colab.ipynb
├── README.md
├── RESEARCH_PLAN.md
├── START_HERE_SI.md           # Sinhala language guide
└── VERIFICATION.md
```

## Key CLI Commands

All commands run via `python -m tea_research <command>`:

| Command | Purpose |
|---|---|
| `make-demo` | Generate 72 synthetic test images |
| `audit-public-metadata` | Validate supplied CSVs without images |
| `prepare-public` | Resolve public dataset images into splits |
| `prepare-field` | Create leave-one-estate-out manifests |
| `train` | Train and select best checkpoint |
| `run` | Full pipeline: train → export → calibrate → evaluate |
| `export` | Export FP32 + INT8 TFLite models |
| `calibrate` | Fit confidence thresholds on source split |
| `evaluate` | Score four test conditions (fp32, int8, fp32_cal, int8_cal) |
| `predict` | Classify a single leaf image |
| `benchmark-host` | Measure CPU inference latency |
| `summarize` | Aggregate results across seeds and estates |

## Architecture Decisions

- **Leave-one-estate-out cross-validation:** Each field fold holds out an entire estate to test generalisation across plantations.
- **Matched FP32/INT8 exports:** Both precisions are exported from the same checkpoint for fair comparison.
- **Temperature scaling:** Post-hoc calibration on a dedicated source split (not test data).
- **Selective prediction (coverage):** Confidence thresholds allow the model to abstain on uncertain images.
- **No EXIF rotation:** Pipeline expects upright pixels in stored files.
- **Family-aware splits:** Original-photo families and identical files never cross train/test boundaries.

## Coding Conventions

- Pure Python standard library + the five pinned packages (no additional deps).
- CLI outputs are JSON to stdout; errors go to stderr.
- All randomness is seeded for reproducibility.
- Paths are handled via `pathlib.Path`.
- Tests run with `python -m unittest discover -s tests -v`.

## Data Notes

- **No real images are included.** The repo contains only metadata CSVs and code to generate synthetic patterns.
- Real tea leaf images must be downloaded separately from [SLTeaLeaf v5](https://data.mendeley.com/datasets/sjmy6k24d6/5) or collected in the field.
- Field data requires expert-confirmed labels with estate/plant/leaf IDs.

## Docker

```bash
# Build
docker build -t tea-leaf-research .

# Demo run (no external data needed)
docker run --rm tea-leaf-research \
    sh -c "python -m tea_research make-demo --output work/demo && \
           python -m tea_research run \
               --manifest work/demo/splits/fold_S01.csv \
               --image-root work/demo/images \
               --output runs/demo/small \
               --model mobilenetv3small --weights none \
               --image-size 64 --batch-size 8 \
               --head-epochs 1 --finetune-epochs 1 \
               --representative 18 --bootstrap 50"

# With real data
docker run --rm \
    -v /path/to/sltealeaf_v5:/app/data/sltealeaf_v5 \
    -v /path/to/output:/app/runs \
    tea-leaf-research train \
        --manifest work/public/public.csv \
        --image-root data/sltealeaf_v5 \
        --output runs/public/mobile_seed11 \
        --model mobilenetv3small --seed 11
```
