# Ceylon Tea Leaf Disease Diagnostic Web Application

A full-stack, real-time diagnostic web application powered by **FastAPI** (Python backend) and **Vite + React + Tailwind CSS** (Frontend).

---

## 🌟 Key Features

1. **Leaf Photograph Diagnosis:**
   - Drag & drop or browse tea leaf photographs (or test with 1-click synthetic demo samples).
   - Fast inference using TensorFlow Lite models.

2. **Model Precision & Quantization Comparison:**
   - **INT8 (Calibrated):** Recommended for mobile/edge devices (~4× smaller memory footprint, fast runtime).
   - **INT8 (Raw):** Direct integer inference.
   - **FP32 (Calibrated):** Full 32-bit floating point with temperature scaling.
   - **FP32 (Raw):** Standard baseline model.

3. **Selective Prediction & Certified Referral:**
   - Implements temperature-scaled confidence cutoffs targeting 80% coverage.
   - **Certified Diagnosis:** When confidence exceeds the threshold.
   - **Expert Referral:** Flags uncertain or out-of-distribution leaves with an alert directing them to an agricultural officer or Tea Research Institute (TRI) pathologist.

4. **Agronomic Knowledge Base (TRI Guidelines):**
   - Diagnostic symptoms, fungal causes, and specific field recommendations for:
     - Blister Blight (*Exobasidium vexans*)
     - Gray Blight (*Pestalotiopsis theae*)
     - Red Rust / Algal Leaf Spot (*Cephaleuros virescens*)
     - Black Blight (*Rhizoctonia / Cladosporium*)
     - Red Spider Mite Damage (*Oligonychus coffeae*)
     - Healthy Leaf (*Camellia sinensis*)

---

## 🚀 Quick Start

### 1. Start Both Backend & Frontend with One Command

Run the unified launcher from inside the `tea_leaf_reliability` folder:

```bash
python web/run_web.py
```

This will automatically:
- Verify and install any missing backend (`fastapi`, `uvicorn`) or frontend dependencies (`npm install`).
- Launch FastAPI on `http://127.0.0.1:8000`.
- Launch Vite React on `http://localhost:5173`.
- Open your default web browser to the dashboard!

---

### 2. Run with Docker Compose (Single Container)

To run the complete production web app inside Docker on port 8000:

```bash
docker compose up web
```

Then visit: **[http://localhost:8000](http://localhost:8000)**

---

### 3. Manual Startup (Alternative)

#### Terminal 1 — FastAPI Backend:
```bash
cd tea_leaf_reliability
python -m pip install -r web/backend/requirements.txt
uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```
API Documentation will be live at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### Terminal 2 — Vite React Frontend:
```bash
cd tea_leaf_reliability/web/frontend
npm install
npm run dev
```
Open your browser at: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Testing When No Real Models Are Trained Yet

If you haven't trained a model on real images yet, simply click the **"Initialize Demo Model"** button on the web dashboard (or let the app generate it). It will create synthetic pattern models in seconds, allowing you to test the complete upload, quantization, calibration, and prediction pipeline immediately!
