"""End-to-end fine-tuning of a ResNet18 on 3-plane knee MRI study volumes (multi-label).

Each study -> (3 planes, K slices, S, S) uint8. A shared 1-channel ResNet18 embeds every slice,
attention pooling over slices gives one vector per plane, the 3 plane vectors are concatenated
(missing planes zeroed) and a linear head predicts the 12 findings.
"""
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from dicom_io import load_series
from features import PLANES, pick_series


# ---------------------------------------------------------------- data
def study_volume(study_dir, series_df, size, k):
    vol = np.zeros((len(PLANES), k, size, size), np.uint8)
    mask = np.zeros(len(PLANES), bool)
    for p, plane in enumerate(PLANES):
        uid = pick_series(series_df, plane) if series_df is not None else None
        v = load_series(Path(study_dir) / uid, size=size) if uid else None
        if v is None:
            continue
        idx = np.linspace(0, len(v) - 1, k).round().astype(int)
        vol[p] = (v[idx] * 255).round().astype(np.uint8)
        mask[p] = True
    return vol, mask


def cache_volumes(df, series, root, size, k, cache_dir, name, n_jobs=4, chunk=64):
    """Decode DICOMs once into an on-disk uint8 memmap. Returns (X memmap, mask)."""
    groups = dict(tuple(series.groupby("StudyInstanceUID")))
    uids = list(df["StudyInstanceUID"])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(str(Path(cache_dir) / f"{name}_X.npy"), mode="w+", dtype=np.uint8,
                                  shape=(len(uids), len(PLANES), k, size, size))
    M = np.zeros((len(uids), len(PLANES)), bool)
    t0 = time.time()
    for i in range(0, len(uids), chunk):
        part = uids[i:i + chunk]
        res = Parallel(n_jobs=n_jobs)(delayed(study_volume)(Path(root) / u, groups.get(u), size, k) for u in part)
        for j, (v, m) in enumerate(res):
            X[i + j], M[i + j] = v, m
        print(f"cached {min(i + chunk, len(uids))}/{len(uids)}  {time.time() - t0:.0f}s", flush=True)
    X.flush()
    return X, M


class VolDS(torch.utils.data.Dataset):
    def __init__(self, X, M, Y, idx):
        self.X, self.M, self.Y, self.idx = X, M, Y, np.asarray(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        y = torch.zeros(1) if self.Y is None else torch.from_numpy(self.Y[j]).float()
        return torch.from_numpy(np.asarray(self.X[j])), torch.from_numpy(self.M[j]), y


# ---------------------------------------------------------------- model
class KneeNet(nn.Module):
    def __init__(self, n_out=12, pretrained=True, drop=0.3):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        conv.weight.data = m.conv1.weight.data.sum(1, keepdim=True)  # RGB filters -> 1 channel
        m.conv1, m.fc = conv, nn.Identity()
        self.backbone = m
        self.attn = nn.Linear(512, 1)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(len(PLANES) * 512, n_out)

    def forward(self, x, mask):
        """x: (B,P,K,H,W) float in [0,1]; mask: (B,P) bool."""
        B, P, K, H, W = x.shape
        f = self.backbone(((x - 0.45) / 0.225).reshape(B * P * K, 1, H, W)).view(B, P, K, -1)
        a = torch.softmax(self.attn(f).squeeze(-1), dim=2).unsqueeze(-1)
        pooled = (a * f).sum(2) * mask.unsqueeze(-1).to(f.dtype)
        return self.head(self.drop(pooled.reshape(B, -1)))


def augment(x):
    """x float (B,P,K,H,W): per-sample brightness/contrast and a per-batch small shift (no flips:
    medial/lateral labels depend on left/right)."""
    B = x.shape[0]
    a = 1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device=x.device) - 0.5)
    b = 0.1 * (torch.rand(B, 1, 1, 1, 1, device=x.device) - 0.5)
    x = (x * a + b).clamp(0, 1)
    dy, dx = np.random.randint(-8, 9, 2)
    return torch.roll(x, shifts=(int(dy), int(dx)), dims=(-2, -1))


# ---------------------------------------------------------------- train / predict
def macro_auc(Y, P):
    r = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if len(np.unique(Y[:, j])) > 1]
    return float(np.mean(r)) if r else float("nan")


@torch.no_grad()
def predict(model, X, M, idx, device, bs=8):
    model.eval()
    dl = torch.utils.data.DataLoader(VolDS(X, M, None, idx), batch_size=bs, num_workers=2)
    out = []
    for x, m, _ in dl:
        with torch.autocast(device_type=device.split(":")[0], enabled=device != "cpu"):
            out.append(torch.sigmoid(model(x.to(device).float() / 255, m.to(device)).float()).cpu().numpy())
    return np.concatenate(out)


def per_label_auc(Y, P, labels):
    """(n_pos, AUC) per label; AUC is nan when a label has one class only."""
    import pandas as pd
    rows = {}
    for j, c in enumerate(labels):
        two = len(np.unique(Y[:, j])) > 1
        rows[c] = {"n_pos": int(Y[:, j].sum()), "auc": roc_auc_score(Y[:, j], P[:, j]) if two else float("nan")}
    return pd.DataFrame(rows).T


def fit(X, M, Y, tr_idx, evals, device, epochs=8, bs=8, lr=3e-4, pretrained=True, select="pseudo", ema=0.998, log=print):
    """evals: {name: idx array} held-out sets. The epoch with the best `select` macro AUC is kept.
    ema: decay of an exponential moving average of the weights, which is what gets evaluated and returned
    (smooths out the late-epoch drift towards pseudo-label noise); None disables it."""
    model = KneeNet(Y.shape[1], pretrained=pretrained).to(device)
    dl = torch.utils.data.DataLoader(VolDS(X, M, Y, tr_idx), batch_size=bs, shuffle=True, num_workers=2,
                                     drop_last=len(tr_idx) > bs, pin_memory=device != "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.15)
    amp = device != "cpu"
    scaler = torch.amp.GradScaler(enabled=amp)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, hist = -1.0, None, []
    avg = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if ema else None
    step = 0

    def ema_update():
        d = min(ema, (1 + step) / (10 + step))  # warm-up: follow the model closely at the start
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    avg[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
                else:
                    avg[k].copy_(v)

    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, m, y in dl:
            x = augment(x.to(device).float() / 255)
            with torch.autocast(device_type=device.split(":")[0], enabled=amp):
                loss = lossf(model(x, m.to(device)).float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); step += 1
            if ema:
                ema_update()
        if ema:  # evaluate the averaged weights, then go back to the raw training weights
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: avg[k].to(raw[k].dtype) for k in raw})
        scores = {n: macro_auc(Y[i], predict(model, X, M, i, device)) for n, i in evals.items() if len(i)}
        hist.append(scores)
        log(f"epoch {ep + 1}/{epochs} loss {tot / len(dl):.4f} {scores} {time.time() - t0:.0f}s")
        s = scores.get(select, -1.0)
        s = -1.0 if s != s else s  # nan (a label with one class only) must not freeze the first epoch
        if best_state is None or s > best:
            best = s; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema:
            model.load_state_dict(raw)
    model.load_state_dict(best_state)
    return model, hist
