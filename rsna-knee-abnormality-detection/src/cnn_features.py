"""Study-level features from a pretrained 2D CNN applied to MRI slices."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from joblib import Parallel, delayed
from dicom_io import load_series
from features import PLANES, pick_series

N_SLICES = 16
EMB = 512  # resnet18


def load_model(device, pretrained=True):
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def series_embedding(vol, model, device, size=224):
    """(D,H,W) volume in [0,1] -> [mean, max] over N_SLICES slices = 2*EMB dims."""
    if vol is None:
        return np.zeros(2 * EMB, np.float32)
    idx = np.linspace(0, len(vol) - 1, N_SLICES).round().astype(int)
    x = torch.from_numpy(vol[idx]).unsqueeze(1)                       # (K,1,H,W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = ((x.repeat(1, 3, 1, 1) - _MEAN) / _STD).to(device)
    f = model(x)                                                      # (K,EMB)
    return torch.cat([f.mean(0), f.max(0).values]).cpu().numpy().astype(np.float32)


def _load_study(uid, series_df, root):
    out = []
    for plane in PLANES:
        s = pick_series(series_df, plane) if series_df is not None else None
        out.append(load_series(Path(root) / uid / s, size=128) if s else None)
    return out


def extract(df, series, root, model, device, chunk=48, n_jobs=4):
    """Returns (len(df), 3*2*EMB). Loads DICOMs in parallel per chunk, embeds on `device`."""
    groups = dict(tuple(series.groupby("StudyInstanceUID")))
    uids = list(df["StudyInstanceUID"])
    rows = []
    for i in range(0, len(uids), chunk):
        part = uids[i:i + chunk]
        vols = Parallel(n_jobs=n_jobs)(delayed(_load_study)(u, groups.get(u), root) for u in part)
        for v3 in vols:
            rows.append(np.concatenate([series_embedding(v, model, device) for v in v3]))
        print(f"embedded {min(i + chunk, len(uids))}/{len(uids)}", flush=True)
    return np.stack(rows)
