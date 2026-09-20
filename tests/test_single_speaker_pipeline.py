import tempfile
import unittest
from pathlib import Path

import pandas as pd

from s2t.common import get_manifest_path, load_manifest, normalize_speaker


class SingleSpeakerManifestTests(unittest.TestCase):
    def test_normalize_and_validate_vietnamese_speaker(self):
        self.assertEqual(normalize_speaker(" hqtv ", accent="vietnamese"), "HQTV")
        with self.assertRaisesRegex(ValueError, "not a vietnamese speaker"):
            normalize_speaker("ABA", accent="vietnamese")

    def test_native_cmu_arctic_targets_are_registered(self):
        self.assertEqual(normalize_speaker(" bdl ", accent="cmu_us"), "BDL")
        self.assertEqual(normalize_speaker("slt", accent="cmu_us"), "SLT")

    def test_personalized_manifest_path_is_isolated(self):
        path = get_manifest_path(accent="vietnamese", speaker="HQTV")
        self.assertEqual(path.parts[-3:], ("speakers", "HQTV", "manifest.csv"))

    def test_load_manifest_filters_speaker_and_split(self):
        rows = [
            {"speaker": "HQTV", "split": "train", "text": "one"},
            {"speaker": "HQTV", "split": "test", "text": "two"},
            {"speaker": "PNV", "split": "train", "text": "three"},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            result = load_manifest(
                "train",
                accent="vietnamese",
                speaker="hqtv",
                manifest_path=manifest,
            )
        self.assertEqual(result["text"].tolist(), ["one"])
        self.assertEqual(result["speaker"].tolist(), ["HQTV"])


if __name__ == "__main__":
    unittest.main()
