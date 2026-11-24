# AI Assistant Rules for This Project

## General

- This is a **research-grade** deep learning project. Reproducibility matters — always preserve seeds, manifest integrity, and class ordering.
- Python version: **3.11 or 3.12 only**. Do not introduce Python 3.10 or 3.13+ features.
- Dependencies are strictly pinned in `requirements.txt`. Do not add new packages without discussion.

## Code Style

- Use `pathlib.Path` for all file system operations, not `os.path`.
- All CLI output must be valid JSON printed to stdout. Use `stderr` for errors and progress.
- Keep functions pure where possible; side effects (file I/O) should be explicit.
- Type hints with `from __future__ import annotations` at the top of each module.
- No wildcard imports. No relative imports outside the `tea_research` package.

## Data Handling

- **Never commit real images** or trained model weights (`.keras`, `.tflite`) to the repository.
- Manifests (CSV) are the source of truth for splits. Do not modify them after the experiment begins.
- Original-photo families must never cross split boundaries — always validate with `audit`.
- Handle the `.jpg` / `.jpeg` alias transparently (the resolver already does this).

## Testing

- Run tests with: `python -m unittest discover -s tests -v`
- All new features must include tests in `tests/`.
- The demo pipeline (`make-demo` → `run`) serves as the integration test — it must always pass on synthetic data without network access.

## Model & Experiment Integrity

- Do not use test data for model selection, hyperparameter tuning, or threshold calibration.
- The four evaluation conditions (`fp32`, `int8`, `fp32_cal`, `int8_cal`) are fixed — do not add or rename them.
- INT8 quantization must be full integer (no float fallback); the converter is configured to fail otherwise.
- Temperature scaling uses only the dedicated source/calibration split.

## Docker

- The Dockerfile uses a multi-stage build. Keep the dependency layer cached by modifying `requirements.txt` only when needed.
- Large data and outputs should be bind-mounted, not baked into the image.
