"""FastAPI backend for Tea Leaf Disease Classification & Reliability Analysis."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root so we can import tea_research
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tea_research.data import DEMO_CLASSES, PUBLIC_CLASSES, make_demo
    from tea_research.workflow import (
        CONDITIONS,
        checked_calibration,
        checked_exports,
        predict_one,
        read_json,
        run_config,
    )
except ImportError as err:
    print(f"Warning: unable to import tea_research: {err}", file=sys.stderr)

app = FastAPI(
    title="Tea Leaf Disease Reliability API",
    description="Deep Learning API for Ceylon Tea Leaf Disease Detection with Temperature Calibration and INT8 TFLite support.",
    version="1.0.0",
)

# Enable CORS for Vite frontend (default port 5173 or any local origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detailed botanical and agricultural knowledge base (Tea Research Institute of Sri Lanka - TRI guidelines)
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "blister_blight": {
        "title": "Blister Blight",
        "scientific_name": "Exobasidium vexans",
        "category": "Fungal Disease",
        "severity": "High (Economic Threat)",
        "symptoms": [
            "Translucent, circular pale green or yellowish spots on tender young leaves.",
            "Spots become blister-like, blister underside turns powdery white with spores.",
            "Blister centers collapse into necrotic brown depressions; curled distorted shoots."
        ],
        "cause": "Airborne basidiospores during wet, misty, overcast weather with high humidity (>80%) and low sunshine (<4 hours/day).",
        "recommendations": [
            "Prophylactic spraying of Copper Oxychloride (280g/ha) or systemic fungicide (Hexaconazole) at 5–7 day plucking rounds during monsoon.",
            "Regulate shade trees (pollarding/lopping) to increase sunlight penetration and air movement.",
            "Shorten plucking rounds to 5–6 days to harvest tender leaves before blisters sporulate."
        ]
    },
    "gray_blight": {
        "title": "Gray Blight (Grey Blight)",
        "scientific_name": "Pestalotiopsis theae",
        "category": "Fungal Disease",
        "severity": "Moderate to High",
        "symptoms": [
            "Small oval yellowish/brown spots expanding into large ash-grey patches with concentric rings.",
            "Minute black fruiting bodies (acervuli) appear on the upper grey surface.",
            "Brittle, dried leaf tissue with dark margin separating diseased and healthy sections."
        ],
        "cause": "Weak wound pathogen invading leaves injured by mechanical plucking, sun-scorch, insect punctures, or shears.",
        "recommendations": [
            "Foliar application of Copper Hydroxide or Mancozeb to protect wounded foliage.",
            "Avoid plucking during extreme heat to prevent sun-scorch injuries.",
            "Maintain bush vigor with balanced N:K fertilizer ratios; avoid severe shear harvesting."
        ]
    },
    "algal_leaf_spot": {
        "title": "Red Rust / Algal Leaf Spot",
        "scientific_name": "Cephaleuros virescens",
        "category": "Parasitic Algal Disease",
        "severity": "Moderate",
        "symptoms": [
            "Circular, velvety, raised orange-red to reddish-brown patches on upper mature leaves.",
            "Patches age to dull grey-green crusty lesions causing premature leaf drop and twig dieback."
        ],
        "cause": "Parasitic green algae flourishing on unshaded, malnourished, or stressed tea bushes in wet humid climates.",
        "recommendations": [
            "Apply Copper Oxychloride 50% WP (0.25%) directed at foliage and stems.",
            "Improve soil drainage and address potassium/magnesium deficiencies.",
            "Prune congested center branches to improve bush aeration."
        ]
    },
    "black_blight": {
        "title": "Black Blight / Sooty Blight",
        "scientific_name": "Rhizoctonia / Cladosporium complex",
        "category": "Fungal Complex",
        "severity": "Moderate",
        "symptoms": [
            "Dark brown to soot-black velvety crust or necrotic discoloration covering the leaf surface.",
            "Leaves turn leathery and wither; impairs photosynthetic capability of maintenance foliage."
        ],
        "cause": "Secondary saprophytic or parasitic fungi colonizing honeydew secreted by scale insects, mealybugs, or aphids.",
        "recommendations": [
            "Control sap-sucking insect vectors (aphids/scales) using horticultural mineral oil or targeted insecticides.",
            "Wash foliage with mild copper-based fungicide spray.",
            "Improve bush canopy light penetration by proper shade lopping."
        ]
    },
    "spider_mites": {
        "title": "Red Spider Mite Damage",
        "scientific_name": "Oligonychus coffeae",
        "category": "Arthropod Pest Damage",
        "severity": "High (Dry Season)",
        "symptoms": [
            "Rusty bronze or copper discoloration along the main midrib and veins on upper surface.",
            "Foliage loses gloss, turns dusty reddish-brown; fine silken webbing visible under hand lens."
        ],
        "cause": "Proliferates during prolonged dry spells, low humidity, and hot dusty conditions.",
        "recommendations": [
            "Apply sulfur formulations (Wettable Sulfur 80% WP) or acaricide (Propargite/Fenazaquin) at early infestation threshold.",
            "Avoid spraying broad-spectrum pyrethroids that kill natural predatory mites (Stethorus beetles).",
            "Plant windbreaks to reduce dust deposits on road-facing plantation borders."
        ]
    },
    "healthy": {
        "title": "Healthy Leaf",
        "scientific_name": "Camellia sinensis",
        "category": "Normal / Healthy",
        "severity": "None (Target Condition)",
        "symptoms": [
            "Vibrant uniform green color with smooth glossy cuticle.",
            "No necrotic lesions, fungal sporulation, blistering, or mite bronzing."
        ],
        "cause": "Optimal bush nutrition, favorable soil pH (4.5–5.5), and sound agricultural practices.",
        "recommendations": [
            "Continue standard plucking rounds (two leaves and a bud).",
            "Maintain standard fertilizer scheduling (TRI 2025/UVA mixtures).",
            "Routine pest and disease scouting during weather transitions."
        ]
    },
    # Synthetic demo fallback
    "pattern_dots": {
        "title": "Synthetic Pattern: Dots",
        "scientific_name": "Synthetic Demo",
        "category": "Benchmark Verification",
        "severity": "Test Pattern Only",
        "symptoms": ["Synthetic geometric dots for testing pipeline functionality."],
        "cause": "Software check pattern.",
        "recommendations": ["Run public baseline or field dataset for real tea leaf research."]
    },
    "pattern_rings": {
        "title": "Synthetic Pattern: Rings",
        "scientific_name": "Synthetic Demo",
        "category": "Benchmark Verification",
        "severity": "Test Pattern Only",
        "symptoms": ["Synthetic geometric rings for testing pipeline functionality."],
        "cause": "Software check pattern.",
        "recommendations": ["Run public baseline or field dataset for real tea leaf research."]
    },
    "pattern_stripes": {
        "title": "Synthetic Pattern: Stripes",
        "scientific_name": "Synthetic Demo",
        "category": "Benchmark Verification",
        "severity": "Test Pattern Only",
        "symptoms": ["Synthetic geometric stripes for testing pipeline functionality."],
        "cause": "Software check pattern.",
        "recommendations": ["Run public baseline or field dataset for real tea leaf research."]
    }
}


def find_available_runs() -> List[Dict[str, Any]]:
    """Scan the workspace for completed training runs containing export.json and calibration.json."""
    found = []
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return found

    for export_path in sorted(runs_dir.rglob("export.json")):
        run_path = export_path.parent
        cal_path = run_path / "calibration.json"
        train_path = run_path / "training.json"
        if cal_path.is_file():
            try:
                train_meta = read_json(train_path) if train_path.is_file() else {}
                export_meta = read_json(export_path)
                found.append({
                    "id": str(run_path.relative_to(ROOT)).replace("\\", "/"),
                    "name": run_path.name,
                    "path": str(run_path),
                    "architecture": train_meta.get("architecture", "unknown"),
                    "protocol": train_meta.get("protocol", "unknown"),
                    "classes": export_meta.get("classes", []),
                    "has_fp32": (run_path / "fp32.tflite").is_file(),
                    "has_int8": (run_path / "int8.tflite").is_file(),
                })
            except Exception:
                continue
    return found


@app.get("/api/health")
def health_check():
    runs = find_available_runs()
    return {
        "status": "online",
        "root": str(ROOT),
        "available_runs_count": len(runs),
        "conditions": list(CONDITIONS),
        "default_run": runs[0]["id"] if runs else None,
    }


@app.get("/api/runs")
def list_runs():
    return {"runs": find_available_runs()}


@app.get("/api/conditions")
def list_conditions():
    return {
        "conditions": [
            {
                "id": "int8_cal",
                "label": "INT8 (Calibrated) — Recommended",
                "precision": "INT8",
                "calibrated": True,
                "description": "Full integer post-training quantization with temperature-scaled confidence thresholds for mobile edge reliability.",
            },
            {
                "id": "int8_raw",
                "label": "INT8 (Raw)",
                "precision": "INT8",
                "calibrated": False,
                "description": "Full integer quantization without confidence temperature scaling (T=1.0).",
            },
            {
                "id": "fp32_cal",
                "label": "FP32 (Calibrated)",
                "precision": "FP32",
                "calibrated": True,
                "description": "Standard 32-bit floating point model with temperature calibration.",
            },
            {
                "id": "fp32_raw",
                "label": "FP32 (Raw Baseline)",
                "precision": "FP32",
                "calibrated": False,
                "description": "Uncalibrated 32-bit floating point baseline.",
            },
        ]
    }


@app.post("/api/setup-demo-model")
def setup_demo_model():
    """Generates a synthetic demo model in runs/demo/small if no trained runs exist."""
    demo_work = ROOT / "work" / "demo"
    demo_run = ROOT / "runs" / "demo" / "small"

    if (demo_run / "export.json").is_file() and (demo_run / "calibration.json").is_file():
        return {
            "status": "already_exists",
            "message": "Demo run already exists and is ready for inference.",
            "run_path": str(demo_run.relative_to(ROOT)).replace("\\", "/"),
        }

    try:
        import subprocess

        # 1. Generate demo patterns
        subprocess.run(
            [sys.executable, "-m", "tea_research", "make-demo", "--output", str(demo_work)],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
        )

        # 2. Run small demo training + export + calibration + evaluation
        cmd = [
            sys.executable, "-m", "tea_research", "run",
            "--manifest", str(demo_work / "splits" / "fold_S01.csv"),
            "--image-root", str(demo_work / "images"),
            "--output", str(demo_run),
            "--model", "mobilenetv3small",
            "--weights", "none",
            "--image-size", "64",
            "--batch-size", "8",
            "--head-epochs", "1",
            "--finetune-epochs", "1",
            "--representative", "18",
            "--bootstrap", "50",
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True)

        return {
            "status": "success",
            "message": "Demo synthetic model successfully created!",
            "run_path": str(demo_run.relative_to(ROOT)).replace("\\", "/"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate demo model: {e}")


@app.post("/api/predict")
async def predict_leaf(
    file: UploadFile = File(...),
    run_path: Optional[str] = Form(None),
    condition: str = Form("int8_cal"),
):
    if condition not in CONDITIONS:
        raise HTTPException(status_code=400, detail=f"Invalid condition '{condition}'. Allowed: {CONDITIONS}")

    # Determine run path
    runs = find_available_runs()
    selected_run: Optional[Path] = None

    if run_path:
        p = ROOT / run_path if not Path(run_path).is_absolute() else Path(run_path)
        if (p / "export.json").is_file():
            selected_run = p
    elif runs:
        selected_run = Path(runs[0]["path"])

    if not selected_run:
        raise HTTPException(
            status_code=404,
            detail="No trained runs found. Please click 'Initialize Demo Model' or train a model first.",
        )

    # Save uploaded image to temp file
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    start_time = time.perf_counter()
    try:
        result = predict_one(selected_run, tmp_path, condition=condition)
        latency_ms = (time.perf_counter() - start_time) * 1000

        pred_class = result["predicted_class"]
        confidence = float(result["confidence"])
        needs_review = bool(result["needs_expert_review"])
        threshold = float(result["threshold"])
        probs = result["probabilities"]

        knowledge = DISEASE_KNOWLEDGE_BASE.get(
            pred_class,
            {
                "title": pred_class.replace("_", " ").title(),
                "scientific_name": "Unspecified / Custom Class",
                "category": "Agricultural Observation",
                "severity": "Variable",
                "symptoms": ["Visual leaf spot or discoloration detected."],
                "cause": "Visual pattern matching class.",
                "recommendations": ["Consult local agricultural officer or Tea Research Institute (TRI)."],
            },
        )

        # Sort probability distribution descending
        sorted_probs = [
            {
                "class_name": c,
                "display_name": DISEASE_KNOWLEDGE_BASE.get(c, {}).get("title", c.replace("_", " ").title()),
                "probability": float(p),
                "percentage": round(float(p) * 100, 2),
            }
            for c, p in sorted(probs.items(), key=lambda item: item[1], reverse=True)
        ]

        return {
            "status": "success",
            "prediction": {
                "class_name": pred_class,
                "display_name": knowledge["title"],
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 2),
                "threshold": threshold,
                "threshold_percent": round(threshold * 100, 2),
                "needs_expert_review": needs_review,
                "is_confident": not needs_review,
            },
            "probabilities": sorted_probs,
            "disease_info": knowledge,
            "execution": {
                "run_id": str(selected_run.relative_to(ROOT)).replace("\\", "/"),
                "condition": condition,
                "protocol": result.get("protocol", "public"),
                "latency_ms": round(latency_ms, 2),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/api/samples")
def get_sample_images():
    """Returns sample image URLs or paths if demo or public images exist."""
    samples = []
    demo_img_dir = ROOT / "work" / "demo" / "images"
    if demo_img_dir.exists():
        for p in demo_img_dir.rglob("*.png"):
            samples.append({
                "name": p.name,
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "description": f"Synthetic test: {p.parent.name}",
            })
            if len(samples) >= 6:
                break
    return {"samples": samples}


@app.get("/api/sample-file/{path:path}")
def get_sample_file(path: str):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(target)


# Mount static production build of Vite frontend if available
DIST_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")

