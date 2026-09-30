"""Load a DICOM series as a normalized (D, H, W) float32 volume."""
from pathlib import Path
import cv2
import numpy as np
import pydicom


def _slice_key(ds, fallback):
    for tag in ("InstanceNumber",):
        v = getattr(ds, tag, None)
        if v is not None:
            return int(v)
    return fallback


def load_series(series_dir, size=128, max_slices=None):
    """Return (D, size, size) float32 in [0, 1], or None if nothing decodes.

    Handles compressed transfer syntaxes when pylibjpeg/gdcm are installed;
    unreadable slices are skipped.
    """
    items = []
    for i, p in enumerate(sorted(Path(series_dir).glob("*.dcm"))):
        try:
            ds = pydicom.dcmread(str(p))
            img = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if img.ndim != 2:
            continue
        items.append((_slice_key(ds, i), cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)))
    if not items:
        return None
    items.sort(key=lambda t: t[0])
    vol = np.stack([im for _, im in items])
    if max_slices and len(vol) > max_slices:
        idx = np.linspace(0, len(vol) - 1, max_slices).round().astype(int)
        vol = vol[idx]
    lo, hi = np.percentile(vol, [1, 99])
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
