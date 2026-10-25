
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. INCARCAM DATELE - doar TRAIN si VALIDATION
# ============================================================

def load_data():
    path = ROOT / "data" / "creditcard.csv"

    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc {path}")

    df = pd.read_csv(path)

    X = df.drop(columns="Class")
    y = df["Class"].to_numpy(dtype=int)

    split_path = ROOT / "split_indices.npz"

    if split_path.exists():
        with np.load(split_path) as split:
            tr = split["train"]
            va = split["val"]
    else:
        indices = np.arange(len(y))

        trva, te = train_test_split(
            indices,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        tr, va = train_test_split(
            trva,
            test_size=0.25,
            stratify=y[trva],
            random_state=42
        )

        np.savez(
            split_path,
            train=tr,
            val=va,
            test=te
        )

    return (
        X.iloc[tr],
        y[tr],
        X.iloc[va],
        y[va]
    )


# ============================================================
# 2. MODELUL LINIAR
# ============================================================

class LinearFraudModel(torch.nn.Module):

    def __init__(self, n_features=30, n_angles=4):
        super().__init__()

        # Acesta este encoderul care va fi folosit de VQC
        self.encoder = torch.nn.Linear(
            n_features, n_angles
        )

        # Folosit DOAR pentru preantrenarea clasica
        self.head = torch.nn.Linear(n_angles, 1)

    def encode(self, x):
        # Transforma cele 4 iesiri in unghiuri (0, pi)
        return torch.pi * torch.sigmoid(
            self.encoder(x)
        )

    def forward(self, x):
        angles = self.encode(x)

        # Logit pentru clasificarea temporara
        return self.head(angles).squeeze(-1)


# ============================================================
# 3. ANTRENAM MODELUL CLASIC
# ============================================================

def train_encoder(
    seed=0,
    epochs=10,
    lr=0.003,
    batch_size=1024
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_val, y_val = load_data()

    # Standardizare antrenata NUMAI pe train
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)

    Xt = torch.tensor(Xtr, dtype=torch.float64)
    Xv = torch.tensor(Xva, dtype=torch.float64)
    yt = torch.tensor(y_train, dtype=torch.float64)

    model = LinearFraudModel(
        n_features=Xt.shape[1],
        n_angles=4
    ).double()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # Fraudele sunt rare, deci le ponderam mai mult
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())

    pos_weight = torch.tensor(
        n_negative / n_positive,
        dtype=torch.float64
    )

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    best_ap = -1.0
    best_epoch = 0
    best_state = None

    print(
        f"Pretraining linear | seed={seed} | "
        f"train={len(Xt)} | epochs={epochs}",
        flush=True
    )

    for epoch in range(1, epochs + 1):

        model.train()

        indices = torch.randperm(len(Xt))
        total_loss = 0.0

        for batch_idx in indices.split(batch_size):

            optimizer.zero_grad()

            logits = model(Xt[batch_idx])

            loss = loss_fn(
                logits,
                yt[batch_idx]
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        # Evaluam clasificatorul temporar pe validation
        model.eval()

        with torch.no_grad():
            scores = torch.sigmoid(
                model(Xv)
            ).numpy()

        ap = average_precision_score(y_val, scores)

        if ap > best_ap:
            best_ap = ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss {total_loss / len(Xt):.4f} | "
            f"Val AUPRC {ap:.4f}",
            flush=True
        )

    # Revenim la cea mai buna epoca
    model.load_state_dict(best_state)

    # Salvam encoderul si standardizarea necesara
    checkpoint = ROOT / f"linear_only_seed{seed}.pt"

    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "head_state": model.head.state_dict(),
        "scaler_mean": torch.tensor(
            scaler.mean_, dtype=torch.float64
        ),
        "scaler_scale": torch.tensor(
            scaler.scale_, dtype=torch.float64
        ),
        "feature_columns": list(X_train.columns),
        "n_angles": 4,
        "seed": seed,
        "best_epoch": best_epoch
    }, checkpoint)

    print(
        f"Encoder salvat: {checkpoint.name} | "
        f"Best epoch={best_epoch} | "
        f"AUPRC={best_ap:.4f}",
        flush=True
    )

    return checkpoint


# ============================================================
# 4. INCARCARE AUTOMATA DIN SCRIPTUL QUANTIC
# ============================================================

def get_or_train_encoder(
    seed,
    epochs=50,
    lr=0.003,
    batch_size=1024,
    force=False
):
    checkpoint = ROOT / f"linear_only_seed{seed}.pt"

    if force or not checkpoint.exists():

        print(
            f"Encoderul seed={seed} nu e disponibil "
            f"sau se cere reantrenare.",
            flush=True
        )

        return train_encoder(
            seed=seed,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size
        )

    print(
        f"Folosesc encoderul deja antrenat: "
        f"{checkpoint.name}",
        flush=True
    )

    return checkpoint


# ============================================================
# 5. TRANSFORMAM TRANZACTIILE IN 4 UNGHIURI
# ============================================================

def encode_transactions(frame, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    columns = checkpoint["feature_columns"]

    X = frame.loc[:, columns].to_numpy(
        dtype=np.float64
    )

    # Folosim standardizarea salvata din train
    mean = checkpoint["scaler_mean"].numpy()
    scale = checkpoint["scaler_scale"].numpy()

    X = (X - mean) / scale

    n_angles = checkpoint["n_angles"]

    encoder = torch.nn.Linear(
        len(columns),
        n_angles
    ).double()

    encoder.load_state_dict(
        checkpoint["encoder_state"]
    )

    encoder.eval()

    with torch.no_grad():

        angles = torch.pi * torch.sigmoid(
            encoder(
                torch.tensor(X, dtype=torch.float64)
            )
        )

    return angles.numpy()


# ============================================================
# 6. RULARE INDEPENDENTA (OPTIONAL)
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=1024)

    args = parser.parse_args()

    train_encoder(
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
