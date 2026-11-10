SLTeaLeaf metadata documentation
================================

Dataset
-------
SLTeaLeaf: Sri Lankan tea leaf disease image dataset.

Metadata package contents
-------------------------
1. SLTeaLeaf_metadata_raw.csv
   Metadata for the 1,791 standardized original images.

2. SLTeaLeaf_metadata_augmented.csv
   Metadata for the 5,968-image expanded balanced set.
   This set contains:
   - 1,791 standardized-original copies (is_augmented = 0)
   - 4,177 generated offline-augmentation variants (is_augmented = 1)

3. SLTeaLeaf_class_count_summary.csv
   Class-wise counts for standardized originals, generated augmentation
   variants, expanded-set totals, and acquisition-condition counts.

4. README_metadata.txt
   This file.

Data provenance and preprocessing
---------------------------------
The standardized original images were created before offline augmentation.
The preprocessing sequence was:
  mobile-phone image
  -> visual screening
  -> leaf-/symptom-focused cropping where required
  -> RGB conversion where required
  -> aspect-ratio-preserving proportional resizing
  -> white padding where required
  -> standardized 1024 x 1024 image.

The white padding represents unused canvas area introduced during image-size
standardization. It was used to avoid directly stretching non-square
leaf/symptom crops to a square image.

Offline augmentation
--------------------
Offline augmentation was applied to the already standardized 1024 x 1024
images. The expanded set remains at 1024 x 1024 pixels.

The supplied augmentation manifest contains 5,968 records:
  1,791 standardized-original copies + 4,177 generated variants.

Each generated variant is linked to its standardized source through
base_image_id.

Class counts
------------
Algal leaf spot : 417 originals + 823 generated variants = 1,240
Black blight    : 345 originals + 621 generated variants =   966
Blister blight  : 126 originals + 631 generated variants =   757
Gray blight     : 431 originals + 814 generated variants = 1,245
Spider mites    : 138 originals + 650 generated variants =   788
Healthy         : 334 originals + 638 generated variants =   972

TOTAL           : 1,791 originals + 4,177 generated variants = 5,968

Acquisition-condition counts for standardized originals
-------------------------------------------------------
Field images                  : 1151
Controlled-environment images : 640

Class-wise acquisition-condition counts are provided in
SLTeaLeaf_class_count_summary.csv.

Metadata field definitions
--------------------------
image_id
    Unique image identifier derived from the filename without its extension.

file_name
    Image filename including extension.

relative_path
    Repository-relative path. The metadata uses portable paths beginning with
    raw/ or augmented/ rather than the local absolute paths contained in the
    source augmentation manifest.

class_folder
    Repository class-folder name:
    Algal_leaf_spot, Black_blight, Blister_blight, Gray_blight,
    Spider_mites, or Healthy.

class_label
    Human-readable class label.

image_set
    standardized_original:
        record belongs to the 1,791-image standardized original set.
    expanded_balanced_set:
        record belongs to the 5,968-image expanded balanced set.

is_augmented
    0 = standardized original or standardized-original copy.
    1 = generated offline-augmentation variant.

base_image_id
    Identifier linking a generated variant to the standardized original image
    from which it was derived. This field should be used for leakage-safe
    train/validation/test splitting.

source_file_name
    Filename of the standardized original source image. Blank for records in
    SLTeaLeaf_metadata_raw.csv.

source_relative_path
    Repository-relative path to the standardized original source image. Blank
    for records in SLTeaLeaf_metadata_raw.csv.

image_format
    Image file format. JPG/JPEG source labels were normalized to JPEG.

original_width_px / original_height_px
    Dimensions recorded in the supplied per-class metadata before the final
    1024 x 1024 standardization step.

width_px / height_px
    Final stored image dimensions. The standardized original and expanded
    datasets are represented as 1024 x 1024 images.

color_mode
    Image colour mode from the supplied source metadata, normally RGB.

preprocessing_status
    Summary of preprocessing performed before offline augmentation.

augmentation_status
    not_augmented:
        record in the standardized original metadata file.
    standardized_original_copy:
        non-synthetic original copy retained in the expanded balanced set.
    generated_augmented_variant:
        synthetically transformed image generated during offline augmentation.

acquisition_condition
    field, controlled_environment, or uncertain where unavailable.

device_name
    Mobile device/camera name recovered from the supplied class metadata where
    available. Missing source values are retained as blank rather than inferred.

datetime
    Date-time value copied from the supplied datetime_file_exif field where
    available. Blank when the source metadata did not contain a recoverable
    value.

exif_datetime
    EXIF-associated date-time copied from datetime_file_exif where available.
    It should not be treated as confirmed capture time unless independently
    verified.

Important normalization decisions
---------------------------------
- Source class labels such as "Blister_blight-original" and
  "Spider_mites-original" were normalized to the canonical class labels
  Blister blight and Spider mites.
- base_id and base_image_id from the source spreadsheets were harmonized into
  the single field base_image_id.
- width/height and image_mode/color_mode naming differences between source
  spreadsheets were harmonized.
- JPG and JPEG labels were normalized to JPEG.
- Missing device or EXIF metadata were not inferred.
- The 5,968-image expanded set is NOT 5,968 newly generated images; it consists
  of 1,791 standardized-original copies plus 4,177 generated variants.

Recommended experimental use
----------------------------
When the expanded dataset is used for machine-learning experiments, all images
sharing the same base_image_id should remain in the same data partition.
Generated variants should normally be restricted to the training partition,
with validation and test evaluation conducted on standardized original images.

Acquisition conditions should also be reported when field and
controlled-environment images are combined, because they represent distinct
image-acquisition settings.
