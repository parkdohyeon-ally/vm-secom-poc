"""Small CNN for WM-811K wafer-map defect classification (PyTorch).

Class-imbalanced (none dominates) -> class-weighted loss + macro-F1 eval.
CPU-runnable on a subsample; GPU auto-used if available.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class WaferCNN(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_model(X, y, n_classes=9, epochs=8, batch=128, lr=1e-3,
                val_frac=0.2, device=None, seed=42):
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=val_frac, stratify=y, random_state=seed)

    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    model = WaferCNN(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr, va = _loader(Xtr, ytr, batch, True), _loader(Xva, yva, batch, False)
    history = []
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        f1 = _macro_f1(model, va, device)
        history.append({"epoch": ep + 1, "val_macro_f1": round(f1, 4)})
        print(f"epoch {ep+1}/{epochs}  val_macro_f1={f1:.4f}")
    return model, history


def _macro_f1(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb.to(device)).argmax(1).cpu().numpy()
            yp.append(p)
            yt.append(yb.numpy())
    return f1_score(np.concatenate(yt), np.concatenate(yp), average="macro")


def predict(model, X, device=None, batch=256):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    out = []
    with torch.no_grad():
        for xb, in _loader(X, np.zeros(len(X)), batch, False):
            out.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)
