# =============================================================================
# Dockerfile — Tea Leaf Reliability Research
#
# Builds a CPU-ready image for training, exporting, calibrating, evaluating,
# and predicting with the tea_research pipeline.
#
# Build:
#   docker build -t tea-leaf-research .
#
# Quick smoke-test (demo with synthetic images):
#   docker run --rm tea-leaf-research \
#       sh -c "python -m tea_research make-demo --output work/demo && \
#              python -m tea_research run \
#                  --manifest work/demo/splits/fold_S01.csv \
#                  --image-root work/demo/images \
#                  --output runs/demo/small \
#                  --model mobilenetv3small --weights none \
#                  --image-size 64 --batch-size 8 \
#                  --head-epochs 1 --finetune-epochs 1 \
#                  --representative 18 --bootstrap 50"
#
# Mount real data for public or field experiments:
#   docker run --rm -v /path/to/sltealeaf_v5:/app/data/sltealeaf_v5 \
#                   -v /path/to/output:/app/runs \
#       tea-leaf-research \
#       python -m tea_research prepare-public \
#           --image-root data/sltealeaf_v5 --output work/public
#
# GPU variant: change the base image to tensorflow/tensorflow:2.19.1-gpu
# =============================================================================

# ---------- base ----------
FROM python:3.12-slim AS base

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal OS-level dependencies for Pillow / TFLite runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- dependencies ----------
FROM base AS deps

WORKDIR /app

# Copy only requirements first to cache the pip layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- application ----------
FROM deps AS app

# Copy the full project source
COPY . .

# Persist outputs in named volumes or bind mounts
VOLUME ["/app/work", "/app/runs"]

# Default working directory
WORKDIR /app

# Health-check: make sure the CLI loads
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=2 \
    CMD python -m tea_research --help > /dev/null 2>&1 || exit 1

# Default entrypoint: run the tea_research CLI
ENTRYPOINT ["python", "-m", "tea_research"]

# Default command: show help when no arguments are given
CMD ["--help"]
