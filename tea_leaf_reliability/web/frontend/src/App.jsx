import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Leaf,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  ShieldCheck,
  Activity,
  Layers,
  Sparkles,
  Info,
  Clock,
  RotateCw,
  ExternalLink,
  ChevronRight,
  ChevronDown
} from 'lucide-react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [condition, setCondition] = useState('int8_cal');
  const [selectedRun, setSelectedRun] = useState('');
  const [runs, setRuns] = useState([]);
  const [conditionsList, setConditionsList] = useState([]);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [settingUpDemo, setSettingUpDemo] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [showTechDetails, setShowTechDetails] = useState(false);

  const fileInputRef = useRef(null);

  // Fetch initial health & runs
  useEffect(() => {
    checkHealth();
    fetchConditions();
    fetchSamples();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setApiOnline(true);
        fetchRuns();
      } else {
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
    }
  };

  const fetchRuns = async () => {
    try {
      const res = await fetch('/api/runs');
      if (res.ok) {
        const data = await res.json();
        setRuns(data.runs || []);
        if (data.runs && data.runs.length > 0 && !selectedRun) {
          setSelectedRun(data.runs[0].path);
        }
      }
    } catch (err) {
      console.error('Failed to fetch runs:', err);
    }
  };

  const fetchConditions = async () => {
    try {
      const res = await fetch('/api/conditions');
      if (res.ok) {
        const data = await res.json();
        setConditionsList(data.conditions || []);
      }
    } catch (err) {
      console.error('Failed to fetch conditions:', err);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch('/api/samples');
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
      }
    } catch (err) {
      console.error('Failed to fetch samples:', err);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleSetupDemo = async () => {
    setSettingUpDemo(true);
    setError(null);
    try {
      const res = await fetch('/api/setup-demo-model', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        await fetchRuns();
        await fetchSamples();
      } else {
        setError(data.detail || 'Demo generation failed');
      }
    } catch (err) {
      setError('Could not connect to backend to setup demo model.');
    } finally {
      setSettingUpDemo(false);
    }
  };

  const handleSelectSample = async (sample) => {
    try {
      const res = await fetch(`/api/sample-file/${sample.path}`);
      const blob = await res.blob();
      const file = new File([blob], sample.name, { type: 'image/png' });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(blob));
      setPrediction(null);
      setError(null);
    } catch (err) {
      setError('Failed to load sample image.');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('condition', condition);
    if (selectedRun) {
      formData.append('run_path', selectedRun);
    }

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Analysis failed');
      }
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-tea-50/20 to-slate-100 flex flex-col">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-tea-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-tea-600 flex items-center justify-center text-white shadow-md shadow-tea-600/20">
              <Leaf className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
                Ceylon Tea Leaf Diagnostic System
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-tea-100 text-tea-800 border border-tea-200">
                  TFLite INT8 / FP32
                </span>
              </h1>
              <p className="text-xs text-slate-500">
                Lightweight Deep Learning with Temperature Calibration & Selective Referral
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${apiOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-xs font-medium text-slate-600">
                {apiOnline ? 'FastAPI Online' : 'FastAPI Offline'}
              </span>
            </div>
            <button
              onClick={checkHealth}
              className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition"
              title="Refresh status"
            >
              <RotateCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
        {/* Banner if no runs exist */}
        {runs.length === 0 && apiOnline && (
          <div className="mb-6 p-4 rounded-2xl bg-amber-50 border border-amber-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-semibold text-amber-900">No Model Checkpoint Detected</h3>
                <p className="text-xs text-amber-700 mt-0.5">
                  You haven't trained or exported a tea disease model yet. You can initialize a quick synthetic demo model with 1 click to test the full pipeline.
                </p>
              </div>
            </div>
            <button
              onClick={handleSetupDemo}
              disabled={settingUpDemo}
              className="flex-shrink-0 px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold shadow-sm transition disabled:opacity-50 flex items-center gap-2"
            >
              {settingUpDemo ? (
                <>
                  <RotateCw className="w-3.5 h-3.5 animate-spin" />
                  Generating Demo Model...
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  Initialize Demo Model
                </>
              )}
            </button>
          </div>
        )}

        {/* Global Error Banner */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-600" />
              <span>{error}</span>
            </div>
            <button onClick={() => setError(null)} className="text-rose-500 hover:text-rose-700 font-bold">
              ✕
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Image Upload & Parameters */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 space-y-5">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
                  <Upload className="w-4 h-4 text-tea-600" />
                  Upload Leaf Photograph
                </h2>
                <span className="text-xs text-slate-400">JPG, JPEG, PNG</span>
              </div>

              {/* Dropzone */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                  previewUrl
                    ? 'border-tea-400 bg-tea-50/20'
                    : 'border-slate-300 hover:border-tea-500 hover:bg-slate-50/50'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleFileChange}
                  className="hidden"
                />

                {previewUrl ? (
                  <div className="space-y-3">
                    <div className="relative mx-auto w-48 h-48 rounded-lg overflow-hidden border border-slate-200 shadow-inner bg-slate-100 flex items-center justify-center">
                      <img
                        src={previewUrl}
                        alt="Leaf preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <p className="text-xs text-slate-600 font-medium truncate max-w-xs mx-auto">
                      {selectedFile?.name}
                    </p>
                    <p className="text-[11px] text-tea-700">Click or drag another image to change</p>
                  </div>
                ) : (
                  <div className="py-6 space-y-3">
                    <div className="w-12 h-12 mx-auto rounded-full bg-tea-50 border border-tea-100 flex items-center justify-center text-tea-600">
                      <Upload className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-700">Drag & drop your leaf image here</p>
                      <p className="text-xs text-slate-400 mt-1">or click to browse from device</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Sample images shelf if available */}
              {samples.length > 0 && (
                <div className="pt-2">
                  <p className="text-xs font-semibold text-slate-500 mb-2">Or test with demo samples:</p>
                  <div className="grid grid-cols-3 gap-2">
                    {samples.map((s, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSelectSample(s)}
                        className="p-1.5 rounded-lg border border-slate-200 hover:border-tea-500 hover:bg-tea-50/40 text-left transition flex items-center gap-2 group"
                      >
                        <img
                          src={`/api/sample-file/${s.path}`}
                          alt={s.name}
                          className="w-8 h-8 rounded object-cover bg-slate-100 flex-shrink-0"
                        />
                        <span className="text-[11px] text-slate-600 group-hover:text-tea-800 truncate">
                          {s.name}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Execution Condition / Precision */}
              <div className="space-y-2 pt-2 border-t border-slate-100">
                <label className="text-xs font-semibold text-slate-700 flex items-center justify-between">
                  <span>Model Execution Condition</span>
                  <span className="text-[10px] text-tea-700 font-normal">TFLite Runtime</span>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { id: 'int8_cal', label: 'INT8 Calibrated', desc: 'Mobile/Edge Recommended' },
                    { id: 'int8_raw', label: 'INT8 Raw', desc: 'Unscaled integer' },
                    { id: 'fp32_cal', label: 'FP32 Calibrated', desc: 'Float32 + Temperature' },
                    { id: 'fp32_raw', label: 'FP32 Raw', desc: 'Standard float baseline' },
                  ].map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setCondition(item.id)}
                      className={`p-2.5 rounded-xl text-left border transition-all ${
                        condition === item.id
                          ? 'border-tea-600 bg-tea-50/60 ring-2 ring-tea-500/20'
                          : 'border-slate-200 hover:border-slate-300 bg-white'
                      }`}
                    >
                      <p className={`text-xs font-semibold ${condition === item.id ? 'text-tea-900' : 'text-slate-800'}`}>
                        {item.label}
                      </p>
                      <p className="text-[10px] text-slate-500 mt-0.5">{item.desc}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model Run Selector if multiple */}
              {runs.length > 1 && (
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-700">Trained Run</label>
                  <select
                    value={selectedRun}
                    onChange={(e) => setSelectedRun(e.target.value)}
                    className="w-full text-xs rounded-xl border-slate-200 py-2 px-3 focus:border-tea-500 focus:ring-tea-500 text-slate-700 bg-slate-50"
                  >
                    {runs.map((r) => (
                      <option key={r.path} value={r.path}>
                        {r.id} ({r.architecture})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Analyze CTA Button */}
              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                className="w-full py-3.5 px-4 rounded-xl bg-tea-600 hover:bg-tea-700 text-white font-semibold text-sm shadow-md shadow-tea-600/25 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <RotateCw className="w-4 h-4 animate-spin" />
                    Running TFLite Inference...
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4" />
                    Analyze & Diagnose Leaf
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right Column: Diagnostic Results & Clinical Recommendations */}
          <div className="lg:col-span-7 space-y-6">
            {!prediction ? (
              <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-12 text-center space-y-4">
                <div className="w-16 h-16 mx-auto rounded-2xl bg-tea-50 border border-tea-100 flex items-center justify-center text-tea-600 shadow-inner">
                  <Leaf className="w-8 h-8" />
                </div>
                <div className="max-w-md mx-auto space-y-1">
                  <h3 className="text-base font-bold text-slate-800">Ready to Analyze</h3>
                  <p className="text-xs text-slate-500">
                    Upload or select a tea leaf image on the left, choose your runtime precision, and click <strong>Analyze & Diagnose Leaf</strong>.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-6 max-w-lg mx-auto text-left">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <Cpu className="w-4 h-4 text-tea-600 mb-1" />
                    <p className="text-xs font-semibold text-slate-800">INT8 Quantization</p>
                    <p className="text-[11px] text-slate-500 mt-0.5">~4x smaller models for instant mobile evaluation.</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <ShieldCheck className="w-4 h-4 text-tea-600 mb-1" />
                    <p className="text-xs font-semibold text-slate-800">Calibrated Referral</p>
                    <p className="text-[11px] text-slate-500 mt-0.5">Flags uncertain cases for TRI expert confirmation.</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
                    <Layers className="w-4 h-4 text-tea-600 mb-1" />
                    <p className="text-xs font-semibold text-slate-800">Multi-Estate Folds</p>
                    <p className="text-[11px] text-slate-500 mt-0.5">Reliability tested across unseen plantation sites.</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-6 animate-fadeIn">
                {/* Top Prediction Banner */}
                <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 overflow-hidden relative">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                          Predicted Diagnosis
                        </span>
                        <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-slate-100 text-slate-700">
                          {prediction.disease_info?.category || 'Classification'}
                        </span>
                      </div>
                      <h2 className="text-2xl font-extrabold text-slate-900 mt-1">
                        {prediction.prediction.display_name}
                      </h2>
                      <p className="text-xs italic text-slate-500 mt-0.5">
                        {prediction.disease_info?.scientific_name}
                      </p>
                    </div>

                    <div className="text-right flex-shrink-0">
                      <div className="text-3xl font-black text-tea-700">
                        {prediction.prediction.confidence_percent}%
                      </div>
                      <span className="text-[11px] text-slate-400 font-medium">Confidence Score</span>
                    </div>
                  </div>

                  {/* Selective Referral / Reliability Status */}
                  <div className="mt-4">
                    {prediction.prediction.needs_expert_review ? (
                      <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="text-xs font-bold text-amber-900 flex items-center gap-1.5">
                            Low Confidence Referral (Needs Expert Review)
                          </p>
                          <p className="text-xs text-amber-800 mt-0.5">
                            The confidence score ({prediction.prediction.confidence_percent}%) fell below the calibrated threshold ({prediction.prediction.threshold_percent}%). Under strict safety protocol, this leaf is referred to an estate field officer or Tea Research Institute (TRI) pathologist for manual inspection.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 flex items-start gap-3">
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="text-xs font-bold text-emerald-900 flex items-center gap-1.5">
                            Reliable Diagnosis (Passed 80% Coverage Cutoff)
                          </p>
                          <p className="text-xs text-emerald-800 mt-0.5">
                            The prediction confidence ({prediction.prediction.confidence_percent}%) satisfies the source-calibrated threshold ({prediction.prediction.threshold_percent}%). It meets the reliability criterion for automated field screening.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Class Probabilities Distribution */}
                <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 space-y-4">
                  <h3 className="text-sm font-bold text-slate-900 flex items-center justify-between">
                    <span>Class Probability Distribution</span>
                    <span className="text-xs font-normal text-slate-400">Softmax outputs</span>
                  </h3>

                  <div className="space-y-3">
                    {prediction.probabilities.map((item, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className={`font-medium ${idx === 0 ? 'text-slate-900 font-bold' : 'text-slate-600'}`}>
                            {item.display_name}
                          </span>
                          <span className="font-semibold text-slate-700">{item.percentage}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              idx === 0
                                ? prediction.prediction.needs_expert_review
                                  ? 'bg-amber-500'
                                  : 'bg-tea-600'
                                : 'bg-slate-300'
                            }`}
                            style={{ width: `${Math.max(item.percentage, 1)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Disease Details & Agronomic Recommendations */}
                {prediction.disease_info && (
                  <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 space-y-4">
                    <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                      <Info className="w-4 h-4 text-tea-600" />
                      Agronomic Guide & Management Protocols (TRI Sri Lanka)
                    </h3>

                    {/* Symptoms */}
                    <div className="space-y-1.5">
                      <p className="text-xs font-semibold text-slate-700">Diagnostic Symptoms:</p>
                      <ul className="list-disc list-inside text-xs text-slate-600 space-y-1 pl-1">
                        {prediction.disease_info.symptoms?.map((s, idx) => (
                          <li key={idx}>{s}</li>
                        ))}
                      </ul>
                    </div>

                    {/* Recommended Action */}
                    <div className="space-y-1.5 pt-2 border-t border-slate-100">
                      <p className="text-xs font-semibold text-slate-700">Recommended Agricultural Measures:</p>
                      <ul className="list-disc list-inside text-xs text-slate-600 space-y-1 pl-1">
                        {prediction.disease_info.recommendations?.map((r, idx) => (
                          <li key={idx} className="text-tea-950 font-medium">
                            {r}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {/* Technical Execution Metrics Accordion */}
                <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-4">
                  <button
                    onClick={() => setShowTechDetails(!showTechDetails)}
                    className="w-full flex items-center justify-between text-xs font-semibold text-slate-700 hover:text-slate-900"
                  >
                    <span className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-tea-600" />
                      Technical Runtime Diagnostics
                    </span>
                    {showTechDetails ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                  </button>

                  {showTechDetails && (
                    <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                      <div className="p-2.5 rounded-xl bg-slate-50">
                        <span className="text-[10px] text-slate-400 block">Condition</span>
                        <span className="font-semibold text-slate-800">{prediction.execution.condition}</span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-slate-50">
                        <span className="text-[10px] text-slate-400 block">Inference Latency</span>
                        <span className="font-semibold text-slate-800">{prediction.execution.latency_ms} ms</span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-slate-50">
                        <span className="text-[10px] text-slate-400 block">Protocol</span>
                        <span className="font-semibold text-slate-800 capitalize">{prediction.execution.protocol}</span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-slate-50 truncate">
                        <span className="text-[10px] text-slate-400 block">Active Run</span>
                        <span className="font-semibold text-slate-800 truncate block" title={prediction.execution.run_id}>
                          {prediction.execution.run_id}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200/80 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-slate-500">
          Reliable Tea Leaf Disease Detection Across Sri Lankan Plantations Using Lightweight Deep Learning • MobileNetV3-Small & EfficientNet-B0
        </div>
      </footer>
    </div>
  );
}
