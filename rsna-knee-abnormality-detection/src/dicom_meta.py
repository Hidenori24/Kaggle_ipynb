"""Acquisition metadata from DICOM headers (scanner, slice thickness, resolution, TR/TE, ...).

Images are intensity-normalized per series, so the CNN cannot tell which site/scanner a study came from; the headers
can (sites differ in label prevalence and report style). Only headers of the first file of the selected series are read.
All values are scaled by fixed constants (no fitted statistics), and missing values are 0.
"""
from pathlib import Path
import numpy as np
import pydicom

_TAGS = ["Manufacturer", "MagneticFieldStrength", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing", "Rows",
         "Columns", "RepetitionTime", "EchoTime", "FlipAngle"]
# (name, scale): value / scale, clipped to [0, 5]
_NUM = [("SliceThickness", 5.0), ("SpacingBetweenSlices", 5.0), ("PixelSpacing", 1.0), ("Rows", 512.0),
        ("Columns", 512.0), ("RepetitionTime", 4000.0), ("EchoTime", 100.0), ("FlipAngle", 180.0),
        ("MagneticFieldStrength", 3.0)]
N_NUM = len(_NUM) + 1                      # + number of slices in the series
_MFR = ["siemens", "ge", "philips", "toshiba", "canon", "hitachi", "fuji", "other"]
N_PLANES = 3
META_DIM = N_PLANES * N_NUM + len(_MFR)


def _float(v):
    try:
        if hasattr(v, "__len__") and not isinstance(v, str):
            v = v[0]
        return float(v)
    except Exception:
        return 0.0


def _mfr_index(name):
    s = str(name or "").lower().strip()
    for i, key in enumerate(_MFR[:-1]):
        if s.startswith(key) or key in s.split():
            return i
    return len(_MFR) - 1 if s else None


def series_meta(series_dir):
    """-> (vector of N_NUM floats, manufacturer index or None). Zeros when the headers cannot be read."""
    vec = np.zeros(N_NUM, np.float32)
    files = sorted(Path(series_dir).glob("*.dcm"))
    if not files:
        return vec, None
    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, specific_tags=_TAGS)
    except Exception:
        return vec, None
    for i, (name, scale) in enumerate(_NUM):
        vec[i] = np.clip(_float(getattr(ds, name, 0.0)) / scale, 0.0, 5.0)
    vec[-1] = np.clip(len(files) / 50.0, 0.0, 5.0)
    return vec, _mfr_index(getattr(ds, "Manufacturer", None))
