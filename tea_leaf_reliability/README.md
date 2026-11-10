# Tea Leaf Reliability Research

This project implements the model and evaluation pipeline for **Reliable Tea Leaf Disease Detection Across Sri Lankan Plantations Using Lightweight Deep Learning**. It classifies a photograph of one leaf. The two architectures are MobileNetV3-Small and EfficientNet-B0. Each selected checkpoint can be exported as FP32 and full INT8 TensorFlow Lite models and tested before and after confidence calibration.

Start with [START_HERE_SI.md](START_HERE_SI.md) for the Sinhala guide, [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the experiment plan, or [Tea_Leaf_Training_Colab.ipynb](Tea_Leaf_Training_Colab.ipynb) for a guided run.

**Data status:** This package includes the metadata and splits you supplied, plus code to generate synthetic test patterns. Real tea leaf photographs and tea-trained weights are not included. A smoke test confirms that the software runs; its scores are not tea disease results.

## Choose the run you need

| Run | Use it to | Data |
|---|---|---|
| Demo | Check setup, training, export and prediction | 72 generated geometric images |
| Public baseline | Build a first real tea classifier and learn the pipeline | SLTeaLeaf v5 image archive, downloaded separately |
| Field research | Test reliability at plantations excluded from development | Your photos with estate and plant IDs and expert-confirmed labels |

The public metadata has original-photo family IDs and no estate IDs. Its test score cannot establish generalisation across plantations.

## 1. Install

Use Python 3.11 or 3.12. Extract the project ZIP and open a terminal inside the **tea_leaf_reliability** directory.

Windows PowerShell:

~~~powershell
py -3.12 -m venv .venv
$teaPython = ".\.venv\Scripts\python.exe"
& $teaPython -m pip install -r requirements.txt
& $teaPython -m tea_research audit-public-metadata
~~~

On macOS/Linux, create the environment with your Python 3.11/3.12 interpreter and run the commands with the environment activated:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tea_research audit-public-metadata
~~~

In the remaining commands, PowerShell users can replace the first word "python" with "& $teaPython". Use CPU for the native Windows setup. A compatible Linux or Colab GPU environment is useful for the full experiment. For a notebook workflow, open the included Colab notebook, upload the project ZIP in its first cell and follow the instructions. The first real run downloads the pretrained ImageNet weights through Keras.

## 2. Run a small software check

~~~bash
python -m tea_research make-demo --output work/demo
python -m tea_research run --manifest work/demo/splits/fold_S01.csv --image-root work/demo/images --output runs/demo/small --model mobilenetv3small --weights none --image-size 64 --batch-size 8 --head-epochs 1 --finetune-epochs 1 --representative 18 --bootstrap 50
python -m tea_research predict --run runs/demo/small --image work/demo/images/S01/S01_0_00.png
~~~

This run holds synthetic site S01 out, trains on the other sites, exports both precisions, fits temperatures and confidence thresholds on a separate source split, and writes four test conditions. The generated images represent patterns, not diseases. Use new output directory names for a fresh run.

## 3. Train your first tea model

1. Download and extract the original photographs from [Sri Lankan Tea Leaf Dataset v5](https://data.mendeley.com/datasets/sjmy6k24d6/5). Point the commands to the directory containing the image folders. The example below uses "data/sltealeaf_v5". Extract any inner image archives too; a ZIP file cannot be used as an image root.
2. Run preparation. It matches the supplied image IDs to actual paths, handles JPG/JPEG aliases, and reports missing, unreadable or ambiguous files. It checks that original-photo families and identical files do not cross splits.
3. Train MobileNetV3-Small using the resulting manifest. Start with one seed. Use the tuning split for development and checkpoint choice.

~~~bash
python -m tea_research prepare-public --image-root data/sltealeaf_v5 --output work/public
python -m tea_research train --manifest work/public/public.csv --image-root data/sltealeaf_v5 --output runs/public/mobile_seed11 --model mobilenetv3small --seed 11
~~~

At default settings, the CSVs produce **1,065 training originals, 96 tuning originals, 92 confidence-calibration originals and 538 test originals**. The supplied 188 validation originals are divided by class for tuning and calibration. The test family IDs are unchanged. The public baseline has six classes, including **spider mite damage** as a pest class. The default training pipeline applies mild online augmentation only to training images.

For a separate comparison with the published offline augmentations, use a new output folder and pass "--packaged-augmentations" to preparation. This adds **2,493 variants from training families only**. Use "--no-augment" on the training command if the planned comparison uses offline augmentation alone. Keep the same test and source validation images. Do not append the expanded manifest to the original manifest; it includes copies of the originals.

### Export and score the trained model

~~~bash
python -m tea_research export --run runs/public/mobile_seed11
python -m tea_research calibrate --run runs/public/mobile_seed11 --coverage 0.8
python -m tea_research evaluate --run runs/public/mobile_seed11 --bootstrap 2000
~~~

Use evaluate after deciding the settings. Its test metrics should not be used to choose epochs, thresholds or model settings. For a preset experiment, the single "run" command performs all four stages. If you move the images, pass "--image-root" to export, calibrate or evaluate. Keep the original manifest, class order and image contents.

## 4. Collect and test your own plantation data

1. Copy [the annotation template](data/field_annotations_template.csv) to your working area. Capture a primary photo of each physical leaf. Keep any supporting views for expert review outside the primary benchmark.
2. Use actual coded estates and bushes. Record "estate_id", "region", "plant_id", "leaf_id", "session_id", "device_name" and "capture_date". A plant ID may recur within its own estate; the code groups by both fields. Keep the primary photos upright in the stored pixels; the pipeline does not apply EXIF rotation.
3. Save image paths relative to the image root. Use "expert_confirmed" only after a qualified reviewer has verified the label. Pending, uncertain and mixed cases are excluded. An example row below shows the format; it is fictional.

~~~csv
image_path,class_label,label_status,estate_id,region,plant_id,leaf_id,session_id,device_name,capture_date
E01/E01_P012_L01.jpg,gray_blight,expert_confirmed,E01,Central,P012,L01,V01,Phone_A,2026-11-10
~~~

4. Use the pilot to choose common classes. The code requires at least three documented estates, all selected classes in the test estate, and enough source bushes to give each training, tuning and calibration split every class. For the planned study, aim for four to six estates, with two estates per chosen region where feasible. Use your actual classes in place of the example.

~~~bash
python -m tea_research prepare-field --annotations data/field_annotations.csv --image-root data/field_images --classes healthy,gray_blight,blister_blight --output work/field
~~~

Each "fold_E01.csv"-style file holds a complete estate out. The source estates are split by bush into roughly 70% train, 15% tune and 15% confidence calibration. All views and visits from a bush must stay together; the benchmark accepts one primary image per leaf. Sessions are recorded and any source-session overlap is reported. All sessions from the held-out estate stay out of development.

Run one pilot fold before scaling up. Use a new folder for the final experiment and freeze the classes, manifests, settings and metrics. The full run below defaults to two models and three seeds per fold. It can take hours; preview the number of trainings first.

~~~bash
python scripts/run_field_experiment.py --manifests work/field --image-root data/field_images --output runs/field --dry-run
python scripts/run_field_experiment.py --manifests work/field --image-root data/field_images --output runs/field
python -m tea_research summarize --runs runs/field --output runs/field_summary
~~~

Six estates × two architectures × three seeds = 36 trained checkpoints. The FP32/INT8 and confidence comparisons reuse each checkpoint. The default experiment runner uses the same main settings for every fold. If a job fails, inspect its error and rerun the missing stages or use a new run folder for that combination; do not mix pilot and final results. The summary gives equal weight to estates after averaging seeds within each estate.

## 5. Read the outputs

| File | What you can use it for |
|---|---|
| checkpoint.keras | Best tuning macro-F1 checkpoint for this run |
| manifest.csv, audit.json, training.json | Reproduce class order, splits and training settings |
| history.json | Loss, accuracy and tuning macro F1 by training stage |
| fp32.tflite, int8.tflite | Matched deployment files from the same checkpoint |
| export.json | Tensor types, quantization scales, representative training image IDs, file sizes and hashes |
| calibration.json | Source-only temperatures and confidence thresholds for each precision |
| evaluation.json | Four test conditions, per-class recall, confusion matrices, Brier score, ECE and paired intervals |
| predictions.csv | Per-image class probabilities, predicted label and referral decision |
| risk_coverage.csv | Threshold, answered fraction and error among answered cases |
| benchmark_host.json | Optional host CPU timing, separate from Android measurements |

Predict with "predict --run … --image … --condition int8_cal". To measure the two exports on your computer:

~~~bash
python -m tea_research benchmark-host --run runs/public/mobile_seed11 --image data/sltealeaf_v5/path/to/a/leaf.jpg
~~~

Replace the example image path with a real file. This is a CPU runtime check and excludes image decoding. For the research phone measurements, follow [the Android benchmark notes](RESEARCH_PLAN.md#android-measurements).

## Troubleshooting

- **Missing images:** The CSVs give names and labels, not pixels. Verify the extracted JPG/JPEG files and image root. Some raw names end in ".jpeg" while supplied split paths end in ".jpg"; the resolver handles that alias.
- **No GPU:** Confirm the runtime, device drivers and TensorFlow installation. Try a smaller batch such as 8. Keep the final experiment's settings consistent.
- **Model download fails:** Keras needs network access to its official pretrained weights on the first run. Demo mode runs with random initialisation. Use the pretrained setting for the planned experiments.
- **Missing class in a field fold:** Collect more expert-confirmed leaves from different bushes or revise the common class list before training. A photo augmentation does not supply an independent leaf.
- **Integer export fails:** Inspect the error. The converter is configured to fail if an operator cannot be quantized; a float fallback is not reported as full INT8.
- **Old output path:** Choose a new run folder for a new experiment. Separate stage commands reuse the selected run.
- **Privacy:** The copied manifest can contain estate/plant metadata. Remove restricted details from any public release, following estate agreements.

Run the integrity tests with "python -m unittest discover -s tests -v". See [VERIFICATION.md](VERIFICATION.md) for what has been tested.

## Sources and data attribution

- Ganesalingam et al., [Sri Lankan Tea Leaf Dataset, version 5](https://data.mendeley.com/datasets/sjmy6k24d6/5), Mendeley Data, 2026, DOI 10.17632/sjmy6k24d6.5, CC BY 4.0. The metadata and split CSVs in "data/public_metadata" are your supplied files, copied unchanged.
- Official Keras applications: [MobileNetV3Small](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small) and [EfficientNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0). Both backbones use their built-in input preprocessing, so the pipeline provides RGB pixels in the 0–255 range.
- Google AI Edge, [Post-training integer quantization](https://ai.google.dev/edge/litert/models/post_training_integer_quant).
- Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017.
