"""CMU ARCTIC dialect groups, as listed by the official FestVox index."""
from pathlib import Path

from s2t.paths import PROJECT_ROOT as ROOT
SOURCE_URL = "http://festvox.org/cmu_arctic/"
CMU_ACCENT_SPEAKERS = {
    "cmu_us": ["AEW", "BDL", "CLB", "EEY", "LJM", "LNH", "RMS", "SLT"],
    "cmu_indian": ["AUP", "AXB", "GKA", "KSP", "SLP"],
    "cmu_german": ["AHW", "FEM"],
    "cmu_canadian": ["JMK"],
    "cmu_scottish": ["AWB"],
    "cmu_israeli": ["RXR"],
}
CMU_SPEAKERS = sorted(s for group in CMU_ACCENT_SPEAKERS.values() for s in group)


def speaker_dir(speaker):
    return ROOT / "corpus" / "cmu_arctic" / f"cmu_us_{speaker.lower()}_arctic"
