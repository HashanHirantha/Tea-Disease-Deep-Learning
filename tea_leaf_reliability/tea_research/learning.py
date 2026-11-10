"""Keras transfer learning and actual TensorFlow Lite inference."""

from __future__ import annotations

import os
import platform
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import image_path, load_manifest, new_directory, sha256, write_json
from .metrics import macro_f1

ARCHITECTURES = ("mobilenetv3small", "efficientnetb0")


@lru_cache(maxsize=1)
def tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the project dependencies: python -m pip install -r requirements.txt") from exc
    threads = int(os.environ.get("TEA_TF_THREADS", "4"))
    if threads > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass  # TensorFlow may already be active in an interactive session.
    return tf


def preprocess_image(path, size):
    tf = tensorflow()
    x = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    x.set_shape([None, None, 3])
    x = tf.image.resize_with_pad(tf.cast(x, tf.float32), size, size, method="bilinear")
    return tf.ensure_shape(x, [size, size, 3])


def dataset(rows, root, class_ids, size, batch, seed, training=False, augment=False):
    tf = tensorflow()
    paths = [str(image_path(root, r["image_path"])) for r in rows]
    labels = np.asarray([class_ids[r["class_label"]] for r in rows], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda path, y: (preprocess_image(path, size), y), num_parallel_calls=2)
    ds = ds.batch(batch)
    if training and augment:
        transform = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.035, fill_mode="reflect", seed=seed + 1),
            tf.keras.layers.RandomZoom(0.07, fill_mode="reflect", seed=seed + 2),
            tf.keras.layers.RandomContrast(0.08, seed=seed + 3),
        ])
        ds = ds.map(lambda x, y: (tf.clip_by_value(transform(x, training=True), 0., 255.), y))
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 2
    return ds.with_options(options).prefetch(1)


def build_model(arch, size, n_classes, weights="imagenet"):
    tf = tensorflow()
    kw = dict(input_shape=(size, size, 3), include_top=False, pooling="avg",
              weights=None if weights == "none" else weights)
    if arch == "mobilenetv3small":
        backbone = tf.keras.applications.MobileNetV3Small(include_preprocessing=True, **kw)
    elif arch == "efficientnetb0":
        backbone = tf.keras.applications.EfficientNetB0(**kw)
    else:
        raise ValueError(f"Unknown architecture: {arch}.")
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(size, size, 3), dtype="float32", name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    logits = tf.keras.layers.Dense(n_classes, dtype="float32", name="logits")(x)
    return tf.keras.Model(inputs, logits, name="tea_classifier")


def train(args):
    tf = tensorflow()
    if not 32 <= args.image_size <= 1024:
        raise ValueError("Image size must be between 32 and 1024.")
    if args.head_epochs < 1 or args.finetune_epochs < 0 or args.finetune_layers < 1:
        raise ValueError("Use at least one head epoch, nonnegative fine-tune epochs and positive fine-tune layers.")
    if args.batch_size < 1 or args.patience < 1:
        raise ValueError("Batch size and patience must be positive.")
    root = Path(args.image_root).resolve()
    rows, audit = load_manifest(args.manifest, root, verify_images=True)
    classes = audit["classes"]
    class_ids = {label: i for i, label in enumerate(classes)}
    source = [r for r in rows if r["split"] == "train"]
    tuning = [r for r in rows if r["split"] == "tune"]
    out = new_directory(args.output)
    manifest_path = out / "manifest.csv"
    manifest_path.write_bytes(Path(args.manifest).read_bytes())
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    train_ds = dataset(source, root, class_ids, args.image_size, args.batch_size,
                       args.seed, training=True, augment=args.augment)
    tune_ds = dataset(tuning, root, class_ids, args.image_size, args.batch_size, args.seed)
    model = build_model(args.model, args.image_size, len(classes), args.weights)
    weight_counts = np.bincount([class_ids[r["class_label"]] for r in source],
                                minlength=len(classes))
    weights = {i: float(len(source) / (len(classes) * n)) for i, n in enumerate(weight_counts)}
    history = []
    best_file = out / "checkpoint.keras"

    class TuneF1(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = -1.0
            self.stage = "head"

        def on_epoch_end(self, epoch, logs=None):
            ys, predictions = [], []
            for x, y in tune_ds:
                ys.append(y.numpy())
                predictions.append(self.model(x, training=False).numpy().argmax(axis=1))
            score = macro_f1(np.concatenate(ys), np.concatenate(predictions), len(classes))
            logs = logs if logs is not None else {}
            logs["val_macro_f1"] = score
            history.append({"stage": self.stage, "epoch": epoch + 1,
                            **{k: float(v) for k, v in logs.items()}})
            if score > self.best:
                self.best = score
                self.model.save(best_file)

    metric = TuneF1()

    def fit(current, epochs, lr):
        current.compile(optimizer=tf.keras.optimizers.Adam(lr),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
                        jit_compile=False)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                                patience=args.patience)
        current.fit(train_ds, validation_data=tune_ds, epochs=epochs, class_weight=weights,
                    callbacks=[metric, stop], verbose=2)

    fit(model, args.head_epochs, args.head_lr)
    if args.finetune_epochs:
        model = tf.keras.models.load_model(best_file, compile=False)
        backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
        backbone.trainable = True
        boundary = max(0, len(backbone.layers) - args.finetune_layers)
        for i, layer in enumerate(backbone.layers):
            layer.trainable = i >= boundary and not isinstance(layer, tf.keras.layers.BatchNormalization)
        metric.stage = "finetune"
        fit(model, args.finetune_epochs, args.finetune_lr)
    config = {
        "architecture": args.model, "weights": args.weights, "seed": args.seed,
        "image_size": args.image_size, "batch_size": args.batch_size,
        "head_epochs": args.head_epochs, "finetune_epochs": args.finetune_epochs,
        "finetune_layers": args.finetune_layers, "head_lr": args.head_lr,
        "finetune_lr": args.finetune_lr, "patience": args.patience, "augment": args.augment,
        "image_root": str(root), "classes": classes, "protocol": audit["protocol"],
        "test_estate": next(iter(audit["estate_ids"]["test"]), None),
        "manifest_sha256": sha256(manifest_path), "checkpoint_sha256": sha256(best_file),
        "tensorflow": tf.__version__, "keras": tf.keras.__version__,
        "python": platform.python_version(), "numpy": np.__version__,
        "devices": [d.name for d in tf.config.list_physical_devices()],
        "best_tune_macro_f1": metric.best, "class_weights": weights,
    }
    write_json(out / "history.json", history)
    write_json(out / "audit.json", audit)
    write_json(out / "training.json", config)
    return config


def representative_rows(rows, count, seed):
    """Use original training images, alternating classes and then estates."""
    from collections import defaultdict, deque

    rng = np.random.default_rng(seed)
    by_class = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: r["image_path"]):
        if row["split"] == "train" and row["is_augmented"] == "0":
            by_class[row["class_label"]][row["estate_id"]].append(row)
    if count < len(by_class):
        raise ValueError("Representative sample budget must include at least one image per class.")
    queues = {}
    for label, estates in sorted(by_class.items()):
        site_queues = [deque(rng.permutation(items).tolist()) for _, items in sorted(estates.items())]
        merged = deque()
        while any(site_queues):
            for q in site_queues:
                if q:
                    merged.append(q.popleft())
        queues[label] = merged
    result = []
    while len(result) < count and any(queues.values()):
        for q in queues.values():
            if q and len(result) < count:
                result.append(q.popleft())
    return result


class LiteRunner:
    """One-image inference using the model's actual input and output quantizers."""

    def __init__(self, model_path, threads=2):
        tf = tensorflow()
        self.runner = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
        self.runner.allocate_tensors()
        self.input = self.runner.get_input_details()[0]
        self.output = self.runner.get_output_details()[0]
        self.input_type = np.dtype(self.input["dtype"])
        self.output_type = np.dtype(self.output["dtype"])

    def tensor(self, x):
        x = np.asarray(x, dtype=np.float32)[None, ...]
        if np.issubdtype(self.input_type, np.integer):
            scale, zero = self.input["quantization"]
            if scale <= 0:
                raise ValueError("Integer model has an invalid input quantization scale.")
            bounds = np.iinfo(self.input_type)
            x = np.clip(np.rint(x / scale + zero), bounds.min, bounds.max)
        return x.astype(self.input_type, copy=False)

    def infer_tensor(self, x):
        self.runner.set_tensor(self.input["index"], x)
        self.runner.invoke()
        output = self.runner.get_tensor(self.output["index"])[0].astype(np.float64)
        if np.issubdtype(self.output_type, np.integer):
            scale, zero = self.output["quantization"]
            if scale <= 0:
                raise ValueError("Integer model has an invalid output quantization scale.")
            output = (output - zero) * scale
        return output

    def predict(self, x):
        return self.infer_tensor(self.tensor(x))

    def contract(self):
        return {
            "input_shape": self.input["shape"].tolist(), "input_dtype": self.input_type.name,
            "output_shape": self.output["shape"].tolist(), "output_dtype": self.output_type.name,
            "input_scale_zero_point": list(self.input["quantization"]),
            "output_scale_zero_point": list(self.output["quantization"]),
            "tensor_dtypes": sorted({np.dtype(t["dtype"]).name for t in self.runner.get_tensor_details()}),
        }


def export_models(run, config, rows, root, budget=200):
    tf = tensorflow()
    size = config["image_size"]
    path = run / "checkpoint.keras"
    if sha256(path) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since training. Start a new run.")
    model = tf.keras.models.load_model(path, compile=False)
    reps = representative_rows(rows, budget, config["seed"])
    if not reps:
        raise ValueError("No original training images for INT8 conversion.")

    def representatives():
        for r in reps:
            yield [preprocess_image(str(image_path(root, r["image_path"])), size)[None, ...].numpy()]

    with tempfile.TemporaryDirectory(prefix="tea_export_") as temp:
        saved = str(Path(temp) / "saved_model")
        archive = tf.keras.export.ExportArchive()
        archive.track(model)

        @tf.function(input_signature=[tf.TensorSpec([1, size, size, 3], tf.float32, name="image")])
        def serve(image):
            return {"logits": model(image, training=False)}

        archive.add_endpoint(name="serve", fn=serve)
        archive.write_out(saved, verbose=False)
        fp_converter = tf.lite.TFLiteConverter.from_saved_model(saved)
        fp_bytes = fp_converter.convert()
        int_converter = tf.lite.TFLiteConverter.from_saved_model(saved)
        int_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        int_converter.representative_dataset = representatives
        int_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        int_converter.inference_input_type = tf.int8
        int_converter.inference_output_type = tf.int8
        int_bytes = int_converter.convert()
        exports = {}
        for kind, contents in (("fp32", fp_bytes), ("int8", int_bytes)):
            p = Path(temp) / (kind + ".tflite")
            p.write_bytes(contents)
            runner = LiteRunner(p)
            contract = runner.contract()
            expected = "int8" if kind == "int8" else "float32"
            if contract["input_dtype"] != expected or contract["output_dtype"] != expected:
                raise ValueError(f"{kind}: unexpected converter input/output type.")
            if kind == "int8" and any(d.startswith("float") for d in contract["tensor_dtypes"]):
                raise ValueError("INT8 export contains floating-point tensors; cannot claim full integer export.")
            x = preprocess_image(str(image_path(root, reps[0]["image_path"])), size)
            logits = runner.predict(x)
            if logits.shape != (len(config["classes"]),) or not np.isfinite(logits).all():
                raise ValueError(f"{kind}: invalid exported logits.")
            exports[kind] = {**contract, "file": p.name, "bytes": len(contents), "sha256": sha256(p)}
        for name, contents in (("fp32.tflite", fp_bytes), ("int8.tflite", int_bytes)):
            tmp = run / (name + ".tmp")
            tmp.write_bytes(contents)
            tmp.replace(run / name)
    result = {"input_preprocessing": "Decode stored RGB pixels; bilinear aspect-preserving resize "
              "with black padding; float range 0..255 before quantization.",
              "output": "Logits; dequantize if integer, then apply temperature and softmax.",
              "classes": config["classes"], "manifest_sha256": config["manifest_sha256"],
              "checkpoint_sha256": config["checkpoint_sha256"],
              "representative_budget": budget, "representative_images": [r["image_path"] for r in reps],
              "models": exports}
    write_json(run / "export.json", result)
    return result
