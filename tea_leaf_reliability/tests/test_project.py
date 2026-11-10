import csv
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image

from tea_research.data import (
    PUBLIC_METADATA, audit_rows, load_manifest, make_demo, public_records,
    relative_path, resolve_public_paths,
)
from tea_research.learning import representative_rows
from tea_research.metrics import (
    coverage_threshold, evaluate_probs, fit_temperature, nll, paired_intervals, probabilities,
)


class MetadataTest(unittest.TestCase):
    def test_supplied_public_split_preserves_all_538_test_families(self):
        rows = public_records()
        report = audit_rows(rows)
        self.assertEqual(report["split_sizes"], {"train": 1065, "tune": 96, "calibration": 92, "test": 538})
        with (PUBLIC_METADATA / "test_original_only_baseid_split.csv").open(newline="", encoding="utf-8-sig") as f:
            original_test = {r["base_id"] for r in csv.DictReader(f)}
        self.assertEqual({r["base_image_id"] for r in rows if r["split"] == "test"}, original_test)
        self.assertTrue(all(not r["estate_id"] for r in rows))

    def test_packaged_variants_stay_in_training_and_out_of_ptq_sample(self):
        rows = public_records(packaged_aug=True)
        report = audit_rows(rows)
        variants = [r for r in rows if r["is_augmented"] == "1"]
        self.assertEqual(len(variants), 2493)
        self.assertEqual(report["split_sizes"]["train"], 3558)
        self.assertEqual({r["split"] for r in variants}, {"train"})
        reps = representative_rows(rows, 200, 11)
        self.assertEqual(len(reps), 200)
        self.assertTrue(all(r["split"] == "train" and r["is_augmented"] == "0" for r in reps))
        self.assertEqual(len({r["class_label"] for r in reps}), 6)

    def test_resolve_jpg_copy_of_jpeg_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "SLTea-raw").mkdir()
            Image.new("RGB", (9, 12)).save(root / "SLTea-raw" / "Leaf_004.jpg")
            row = {"image_path": "raw/Gray_blight/Leaf_004.jpeg", "is_augmented": "0"}
            resolve_public_paths([row], root)
            self.assertEqual(row["image_path"], "SLTea-raw/Leaf_004.jpg")

    def test_path_traversal_and_absolute_paths_are_rejected(self):
        for name in ("../img.jpg", "/etc/img.jpg", "C:\\images\\img.jpg"):
            with self.assertRaises(ValueError):
                relative_path(name)


class EstateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.demo = Path(self.temp.name) / "demo"
        make_demo(self.demo)
        self.root = self.demo / "images"
        self.rows, self.report = load_manifest(self.demo / "splits" / "fold_S01.csv", self.root)

    def test_site_and_plants_do_not_cross_splits(self):
        self.assertEqual(self.report["estate_ids"]["test"], ["S01"])
        for split in ("train", "tune", "calibration"):
            self.assertNotIn("S01", self.report["estate_ids"][split])
        for group in {r["group_id"] for r in self.rows}:
            self.assertEqual(len({r["split"] for r in self.rows if r["group_id"] == group}), 1)
        for split in ("train", "tune", "calibration", "test"):
            self.assertEqual(len(self.report["counts"][split]), 3)

    def test_rejects_a_test_estate_photo_reassigned_to_training(self):
        test_row = next(r for r in self.rows if r["split"] == "test")
        test_row["split"] = "train"
        with self.assertRaisesRegex(ValueError, "estate absent"):
            audit_rows(self.rows)

    def test_rejects_identical_files_across_train_and_test(self):
        source = next(r for r in self.rows if r["split"] == "train")
        test = next(r for r in self.rows if r["split"] == "test")
        copyfile(self.root / source["image_path"], self.root / test["image_path"])
        with self.assertRaisesRegex(ValueError, "Identical image bytes across splits"):
            audit_rows(self.rows, self.root)


class ConfidenceTest(unittest.TestCase):
    def test_temperature_improves_calibration_nll_without_changing_argmax(self):
        z = np.array([[8., 0.], [9., 0.], [0., 8.], [0., 9.]])
        y = np.array([0, 1, 1, 1])
        temp = fit_temperature(z, y)
        self.assertGreater(temp, 1)
        self.assertLess(nll(z, y, temp), nll(z, y))
        np.testing.assert_array_equal(z.argmax(axis=1), probabilities(z, temp).argmax(axis=1))

    def test_referral_reports_coverage_and_error_separately(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([[.95, .05], [.80, .20], [.40, .60], [.45, .55]])
        cutoff = coverage_threshold(p.max(axis=1), .5)
        report = evaluate_probs(y, p, ["a", "b"], cutoff)
        self.assertAlmostEqual(cutoff, .8)
        self.assertEqual(report["coverage"], .5)
        self.assertEqual(report["retained_error"], .5)
        self.assertEqual(report["per_class"]["b"]["support"], 2)

    def test_paired_resampling_identical_models_has_zero_differences(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        p = np.eye(2)[y] * .8 + .1
        scores = dict.fromkeys(("fp32_raw", "fp32_cal", "int8_raw", "int8_cal"), p)
        result = paired_intervals(y, ["p1", "p1", "p2", "p2", "p3", "p3"], scores, 50)
        for info in result.values():
            self.assertEqual(info["difference"], 0)
            self.assertEqual(info["ci95"], [0, 0])


if __name__ == "__main__":
    unittest.main()
