# Project verification

The package was exercised on Linux with Python 3.12, TensorFlow CPU 2.19.1, Keras 3.11.3 and NumPy 1.26.4.

- Both MobileNetV3-Small and EfficientNet-B0 completed a software smoke run: grouped split, one head epoch, one fine-tuning epoch, paired exports, separate source calibration and four held-out test conditions.
- The smoke data consists of 72 generated pattern images across three synthetic sites. The model used 64 × 64 inputs and random initialisation. None of its accuracy or confidence scores measures tea disease recognition.
- Both integer models were loaded in the interpreter and checked for INT8 input/output, finite logits and integer tensor types. Single-image prediction, a host CPU timing run and multi-run summaries were also exercised.
- Ten integrity tests passed, covering the supplied public test families, augmentation isolation, JPEG path aliases, path containment, estate/plant separation, duplicate-file leakage, calibration and paired resampling.
- The supplied metadata resolves to 1,065 train, 96 tune, 92 calibration and 538 test originals by default. With the optional packaged variants, only training grows to 3,558 records. The images themselves were not available for verification.

The final real-image results, ImageNet fine-tuning on real tea, Colab execution on a GPU, Windows execution and Android device measurements remain to be performed in your environment. The notebook contains no precomputed research results. The dependency versions are pinned to the tested TensorFlow API; a future upgrade may require moving inference to the standalone LiteRT interpreter.
