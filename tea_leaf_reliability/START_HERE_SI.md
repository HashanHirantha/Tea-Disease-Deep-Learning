# මුලින්ම කරන්න ඕන දේ

මේ project එක ඔබේ **Reliable Tea Leaf Disease Detection Across Sri Lankan Plantations Using Lightweight Deep Learning** research එකටයි. MobileNetV3-Small සහ EfficientNet-B0 models දෙක train කරලා, FP32 සහ INT8 versions සසඳන්න පුළුවන්.

**මේ ZIP එකේ සැබෑ තේ කොළ photos හෝ තේ රෝග සඳහා train කළ weights ඇතුළත් නැහැ.** ඔබ දුන් metadata CSV සහ splits ඇතුළත්. Demo එකේ කෘත්‍රිම රූප භාවිතයෙන් code එක ක්‍රියාකරනවාද බලනවා. එහි scores ඔබේ research results ලෙස භාවිත කරන්න එපා.

1. **මුලින් demo එක.** ZIP එක extract කරන්න. Colab භාවිත කරනවා නම් "Tea_Leaf_Training_Colab.ipynb" විවෘත කරලා project ZIP එක upload කරන්න. Settings cell එකේ MODE = "demo" තබා cells run කරන්න. Windows terminal එකෙන් කරන්න නම් README එකේ Windows setup සහ demo commands භාවිත කරන්න.
2. **පළමු සැබෑ model එක.** [Sri Lankan Tea Leaf Dataset v5](https://data.mendeley.com/datasets/sjmy6k24d6/5) රූප download කර extract කරන්න. Notebook එකේ MODE = "public" කර IMAGE_ROOT එක රූප තිබෙන folder එකට යොමු කරන්න. MobileNetV3-Small එකෙන් පටන් ගන්න. Test images 538ක් ඔබ දුන් split එකෙන්ම වෙන් වෙනවා.
3. **Field pilot එක.** Miss Praveena සහ supervisor සමඟ වතු, expert reviewer සහ disease classes තහවුරු කරන්න. එක් කොළයක ප්‍රධාන photo එකක් ගන්න. වත්ත, ගස, කොළය, visit, phone එක සහ දිනය CSV එකට ලියන්න. Expert විසින් තහවුරු කළ පසුව label_status එක "expert_confirmed" කරන්න.
4. **Research run එක.** Field CSV එකෙන් "prepare-field" command එකෙන් splits සාදන්න. වත්තක් සම්පූර්ණයෙන් test සඳහා වෙන් වෙනවා. ඒ වත්තේ photos training, tuning, INT8 representative set සහ confidence calibration සඳහා යොදාගන්නේ නැහැ. මුලින් එක් fold එකක් run කරලා පසුව models දෙක සහ seeds තුනට වැඩි කරන්න.
5. **ප්‍රතිඵල කියවන්න.** "evaluation.json" එකේ disease recall, macro F1 සහ confidence metrics තිබෙනවා. "predictions.csv" එකෙන් එක් photo එකකට prediction සහ expert review අවශ්‍යද බලන්න පුළුවන්. "fp32.tflite" සහ "int8.tflite" files පසුව Android device එකේ මැනීමට භාවිත කරන්න.

**ඔබේ මුල් milestone එක:** Public images ලැබුණාම එක් MobileNetV3-Small model එකක් train කර "checkpoint.keras" සහ TFLite files දෙක ලබාගන්න. ඒ අතරතුර ඔබේ අලුත් field images සඳහා estate සහ bush IDs අනිවාර්යයෙන් සටහන් කරන්න.

සම්පූර්ණ commands සඳහා [README](README.md) බලන්න. සති 24ක සැලසුම සහ පර්යේෂණ ප්‍රතිඵල සසඳන ආකාරය [research plan](RESEARCH_PLAN.md) එකේ තිබෙනවා.
