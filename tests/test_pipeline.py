import hashlib
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prepare_splits import source_identity  # noqa: E402


class SplitIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = {
            split: pd.read_csv(PROJECT_ROOT / "data" / "splits" / f"{split}.csv")
            for split in ("train", "val", "test")
        }

    def test_copy_suffixes_share_identity(self):
        base = "abc___R.S_HL 0632.JPG"
        copy = "xyz___R.S_HL 0632 copy 5.jpg"
        self.assertEqual(source_identity(base), source_identity(copy))

    def test_source_groups_are_disjoint(self):
        group_sets = {
            split: set(frame["source_group"])
            for split, frame in self.frames.items()
        }
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["val"]))
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["test"]))
        self.assertTrue(group_sets["val"].isdisjoint(group_sets["test"]))

    def test_exact_file_hashes_are_disjoint(self):
        def digest(relative_path):
            return hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()

        hash_sets = {
            split: {digest(path) for path in frame["filepath"]}
            for split, frame in self.frames.items()
        }
        self.assertTrue(hash_sets["train"].isdisjoint(hash_sets["val"]))
        self.assertTrue(hash_sets["train"].isdisjoint(hash_sets["test"]))
        self.assertTrue(hash_sets["val"].isdisjoint(hash_sets["test"]))

    def test_every_class_is_represented_in_every_split(self):
        expected = set(self.frames["train"]["class_name"])
        self.assertGreater(len(expected), 1)
        for frame in self.frames.values():
            self.assertEqual(expected, set(frame["class_name"]))

    def test_integrity_report_passes(self):
        report_path = PROJECT_ROOT / "outputs" / "metrics" / "split_integrity_report.json"
        report = json.loads(report_path.read_text())
        self.assertTrue(report["checks_passed"])
        self.assertEqual(report["source_groups_spanning_splits"], 0)
        self.assertEqual(report["exact_hashes_spanning_splits"], 0)

    def test_perceptual_hash_audit_passes(self):
        report_path = PROJECT_ROOT / "outputs" / "metrics" / "perceptual_hash_audit.json"
        report = json.loads(report_path.read_text())
        self.assertTrue(report["checks_passed"])
        self.assertEqual(report["groups_spanning_splits"], 0)


if __name__ == "__main__":
    unittest.main()
