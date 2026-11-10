# Model and experiment plan

The project asks how integer compression and confidence calibration affect tea leaf classification at an estate excluded from development. The public baseline is an implementation milestone. The new field dataset and the held-out estate measurements are the main experiment.

## Milestones and outputs

| Weeks | Work | Evidence before moving on |
|---|---|---|
| 1–2 | Run the synthetic check, acquire SLTeaLeaf v5, prepare the supplied splits and train one MobileNetV3-Small baseline. Confirm scope and access with the supervisors. | A saved checkpoint, audit and tuning history; a successful FP32/INT8 export. |
| 3–4 | Pilot field photography and expert review. Check which diseases are available across planned sites. | Agreed labels, one-photo-per-leaf protocol, metadata schema, permissions and sampling targets. |
| 5–10 | Collect and review images. Record estate and plant IDs, devices, dates and sessions. Check exact duplicates and inspect near duplicates. | Frozen annotations and documented exclusions, class counts and field provenance. |
| 11–13 | Generate estate folds, fix model and evaluation settings, run a small timing pilot. | Audited manifests, settings for both models and a defined primary comparison. |
| 14–17 | Train both models at seeds 11, 22 and 33 on each fold; export, calibrate and evaluate. | Per-image predictions and paired intervals for all specified conditions. |
| 18–20 | Run the exported models on at least one named Android device and document preprocessing and runtime. | FP32/INT8 model size and measured latency; memory measurements where available. |
| 21–24 | Compare estates and classes, document errors and limitations, write thesis and paper draft. | Reproducible tables, code, frozen manifests and a review-ready manuscript. |

Adjust weeks to the university calendar and field access. Start with a one-fold pilot rather than launching the full experiment before the data audit. Agree the sample targets with the supervisors; 50 distinct leaves per estate/class is a collection target, not a power calculation. The current working scope is healthy plus two or three expert-confirmed diseases.

## Controls that make the comparison useful

- **Images:** Use one primary RGB photograph per distinct leaf. Photograph multiple bushes and field blocks. Distribute devices across estates where possible so estate and device are not interchangeable. The code checks exact duplicates; near duplicates still need human review.
- **Splits:** For each fold, reserve one whole estate for the test. Keep bushes together within source splits. Freeze the class list before the main experiment; don't relabel diseases by region. A class-balanced collection cannot estimate disease prevalence.
- **Training:** Compare the two pretrained architectures at the same resolution, batch size, epoch limits and seed list. Save the checkpoint with highest tuning macro F1. The implementation freezes the backbone for head training, then fine-tunes its last layers with batch-normalization layers frozen. Class weights use the training set.
- **Integer export:** Use original training images for the representative set, balanced across classes and spread across estates where possible. Convert the same checkpoint to FP32 and full INT8. Verify integer tensors, then test the actual exports.
- **Confidence:** Fit a separate positive temperature for each precision using only source calibration logits and labels. Choose a confidence cutoff targeting 80% acceptance on that same split. Ties can raise achieved acceptance above 80%; use the observed coverage on the held-out test. Temperature scaling changes probabilities and referral decisions while preserving top-class order for a fixed logit vector.
- **Evaluation:** Report disease recall and macro F1 before referral. Report multiclass Brier score and ECE for confidence. Give coverage and error among answered cases together. The confidence cutoff is a research referral rule; a novel disease can still be predicted with high confidence.

## Primary contrasts

| Contrast | What stays fixed | What to report |
|---|---|---|
| INT8 raw minus FP32 raw | Same checkpoint, test images and class order | Macro F1 difference, disease recall differences, file size and latency |
| FP32 calibrated minus FP32 raw | Same FP32 logits and test images | Brier difference, ECE, coverage and retained error |
| INT8 calibrated minus INT8 raw | Same INT8 logits and test images | Brier difference, ECE, coverage and retained error |

Use paired bootstrap intervals resampling whole bushes within each held-out estate. The intervals describe uncertainty conditional on the sampled estates. The code averages training seeds within each estate before giving estates equal weight. Keep the per-estate results visible. A small or negative change is a valid result. Prefer paired differences with intervals to a single headline accuracy.

## Android measurements

Use a matching LiteRT/TensorFlow Lite runtime on one named Android device. An Android app is not required to collect initial model-only timings. The official [benchmark tool instructions](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) explain how to obtain or build the Android benchmark binary. Run both models with the same CPU thread count and delegate settings, for example:

~~~bash
adb push benchmark_model /data/local/tmp/benchmark_model
adb push fp32.tflite /data/local/tmp/tea_fp32.tflite
adb push int8.tflite /data/local/tmp/tea_int8.tflite
adb shell chmod +x /data/local/tmp/benchmark_model
adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/tea_fp32.tflite --num_threads=2 --warmup_runs=30 --num_runs=200
adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/tea_int8.tflite --num_threads=2 --warmup_runs=30 --num_runs=200
~~~

Run these from a directory containing the binary and model files. Record phone model, OS, runtime build, delegate, threads, warm-up, number of repetitions, battery and thermal conditions. Repeat matched measurements and report the statistics the tool actually supplies. If median or p95 is needed and the binary reports only averages, collect per-invocation timings with a small instrumented runner. Use a device profiler for process memory. Measure preprocessing and total app latency separately if you build the interface.

For photo inference, decode upright RGB, bilinearly resize preserving aspect ratio and pad to the model's square input with zeros. Match the training pipeline's resize/pad convention. Feed pixel values 0–255 to the FP32 network. For INT8 use round(pixel / input_scale + input_zero_point), clipped to the int8 range. Dequantize logits as (output − output_zero_point) × output_scale. Apply the saved temperature and softmax outside the network. Use "export.json" and "calibration.json" to reproduce the input and output contract. Check desktop/phone predictions for the same images before interpreting the phone results.

## Literature and publication notes

- The closest Sri Lankan [HDBO-R study](https://doi.org/10.1007/s44443-026-01275-4) reports a lightweight classifier and FP32 Android deployment, and identifies stricter estate-held-out validation, integer export and calibration as areas for follow-up.
- [TEA-LWNet](https://doi.org/10.3389/fpls.2026.1879101) already reports tea detection on regions excluded from training using UAV multispectral images. Position your contribution around the controlled, paired phone-image reliability experiment.
- [Xiang et al.](https://doi.org/10.3389/fpls.2026.1826962) study reliability under controlled-to-field plant-image shift. A calibration method that works on source validation may still fail on a different estate.
- [Weerasekara et al., iPURSE 2024](https://ir.lib.pdn.ac.lk/handle/20.500.14444/7581) is useful biological background on gray blight associated fungi sampled across Sri Lankan tea sites. The proposed model classifies visually confirmed disease categories; it does not identify fungal species from pixels.

Keep the public baseline and field evaluation in separate result tables. Report which estates and classes were included, why others were excluded, sample sizes, label verification, all planned contrasts, and device measurements. Preserve seeds, manifests, checkpoints, exports and evaluation records. Release images or detailed site metadata only with the required permission. Frame the paper as empirical evidence about the sampled plantations and selected classes.
