# Troubleshooting Guide

## Common Issues

### 1. TensorFlow Installation Fails
- **Cause:** Incompatible Python version or OS.
- **Fix:** Use Python 3.11 or 3.12. On Windows, CPU-only TensorFlow is recommended. For GPU, use the Docker image with `tensorflow/tensorflow:2.19.1-gpu` base.

### 2. Missing Images Error
- **Cause:** The CSVs reference images that aren't downloaded or extracted.
- **Fix:** Download [SLTeaLeaf v5](https://data.mendeley.com/datasets/sjmy6k24d6/5), extract all inner archives, and point `--image-root` to the folder containing the class directories.

### 3. INT8 Export Fails
- **Cause:** An operator cannot be fully quantized.
- **Fix:** This is by design — the pipeline rejects float fallback. Check the TensorFlow version and representative dataset size (`--representative`).

### 4. Out of Memory During Training
- **Fix:** Reduce `--batch-size` (try 8 or 16). On CPU, use `--image-size 64` for testing. For full experiments, use a GPU or Colab.

### 5. "No fold_*.csv manifests found"
- **Cause:** Wrong `--manifests` path for `run_field_experiment.py`.
- **Fix:** Point to the directory containing `fold_E01.csv`, `fold_E02.csv`, etc., created by `prepare-field`.

### 6. Keras Weight Download Fails
- **Cause:** No network access (e.g., in Docker or air-gapped environment).
- **Fix:** For demo runs use `--weights none`. For real experiments, ensure network access on first run or pre-cache the weights.

### 7. EXIF Rotation Issues
- **Cause:** The pipeline does not apply EXIF rotation.
- **Fix:** Ensure photos are saved with upright pixel orientation before ingestion.

## Performance Tips

- Use GPU for training (Colab T4 or better).
- The demo pipeline is intentionally tiny — do not judge model quality from synthetic scores.
- For field experiments, start with `--dry-run` to preview the number of training jobs.
- INT8 models are ~4× smaller and faster on edge devices; FP32 is the accuracy reference.
