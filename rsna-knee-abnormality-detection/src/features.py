"""Handcrafted per-study features (quick baseline; replace with CNN embeddings later)."""
from pathlib import Path
import numpy as np
from dicom_io import load_series

PLANES = ["Sagittal", "Coronal", "Axial"]
N_POS, GRID, N_HIST = 5, 8, 16
DIM_PER_SERIES = N_POS * GRID * GRID + N_HIST + 2


def pick_series(rows, plane):
    """Prefer fluid-sensitive + fat-suppressed, then fluid-sensitive, then any."""
    r = rows[rows["Anatomical_Plane"] == plane]
    if r.empty:
        return None
    for cond in ((r["Fluid_Sensitive"] == 1) & (r["Fat_Suppression"] == 1), r["Fluid_Sensitive"] == 1):
        if cond.any():
            return r[cond].iloc[0]["SeriesInstanceUID"]
    return r.iloc[0]["SeriesInstanceUID"]


def series_features(vol):
    if vol is None:
        return np.zeros(DIM_PER_SERIES, np.float32)
    d = len(vol)
    idx = np.linspace(0, d - 1, N_POS).round().astype(int)
    h, w = vol.shape[1:]
    grids = vol[idx].reshape(N_POS, GRID, h // GRID, GRID, w // GRID).mean(axis=(2, 4)).ravel()
    hist = np.histogram(vol, bins=N_HIST, range=(0, 1))[0] / vol.size
    return np.concatenate([grids, hist, [float(d), 1.0]]).astype(np.float32)


def study_features(study_dir, study_series):
    feats = []
    for plane in PLANES:
        uid = pick_series(study_series, plane)
        vol = load_series(Path(study_dir) / uid, size=128) if uid else None
        feats.append(series_features(vol))
    return np.concatenate(feats)
