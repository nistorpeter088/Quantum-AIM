"""Task 2 - modelul final: VQC cu encoding pe unghiuri, transformare cuantila, KAN pe fiecare feature.

Instalare: pip install pennylane torch scikit-learn pandas numpy scipy
Exemple:
  python kan_vqc.py                                   # Adam, validare completa, 30 epoci, 5 seeds
  python kan_vqc.py --val cheap --out kan_cheap       # Adam, validare ieftina
  python kan_vqc.py --optimizer spsa --out kan_spsa   # SPSA (validare ieftina)
  python kan_vqc.py --quick                           # verificare rapida, NU rezultate
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED_SPLIT = 42
SEEDS = [0, 1, 2, 3, 4]
NOISE = [0.0, 0.001, 0.01, 0.05]
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}
SHOTS = 1024
N_QUBITS, ANSATZ_REPS, READOUT_WIRE, KAN_BASIS = 4, 2, 1, 6
DIAGONAL_GATES = {"RZ", "PauliZ", "S", "T", "PhaseShift", "CZ", "IsingZZ", "MultiRZ"}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split():
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    X, y = df.drop(columns="Class"), df["Class"].to_numpy(dtype=int)
    split_file = ROOT / "split_indices.npz"
    if split_file.exists():
        with np.load(split_file) as split:
            tr, va, te = (split[k] for k in ("train", "val", "test"))
    else:
        trva, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED_SPLIT)
        tr, va = train_test_split(trva, test_size=0.25, stratify=y[trva], random_state=SEED_SPLIT)
        np.savez(split_file, train=tr, val=va, test=te)
    return X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], y[te]


def undersample(X, y, seed):
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y == 1)
    legit = rng.choice(np.flatnonzero(y == 0), size=len(fraud), replace=False)
    order = rng.permutation(np.concatenate((fraud, legit)))
    return X.iloc[order], y[order]


class QuantileEncoder:
    def __init__(self, seed):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=N_QUBITS, random_state=seed)
        self.quantiles = QuantileTransformer(n_quantiles=1000, output_distribution="uniform",
                                             subsample=10**6, random_state=seed)

    def scaled(self, frame):
        out = frame.copy()
        out[["Time", "Amount"]] = self.scaler.transform(frame[["Time", "Amount"]])
        return out.to_numpy()

    def fit(self, frame):
        self.scaler.fit(frame[["Time", "Amount"]])
        self.quantiles.fit(self.pca.fit_transform(self.scaled(frame)))
        return self

    def transform(self, frame):
        return np.pi * self.quantiles.transform(self.pca.transform(self.scaled(frame)))


class StratifiedSampler:
    def __init__(self, y, batch_size, seed):
        self.pools = [np.flatnonzero(y == 0), np.flatnonzero(y == 1)]
        n_pos = batch_size // 2
        self.counts = [batch_size - n_pos, n_pos]
        self.weights = torch.tensor(np.concatenate([np.full(n, 0.5 / n) for n in self.counts]))
        self.rng = np.random.default_rng(seed)

    def draw(self):
        idx = np.concatenate([self.rng.choice(pool, size=n, replace=False)
                              for pool, n in zip(self.pools, self.counts)])
        return idx, self.weights


def validation_subset(y_val, neg_rate, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_val == 1), np.flatnonzero(y_val == 0)
    kept = rng.choice(neg, size=max(1, round(neg_rate * len(neg))), replace=False)
    subset = np.sort(np.concatenate((pos, kept)))
    return subset, np.where(y_val[subset] == 1, 1.0, len(neg) / len(kept))


def init_weights(seed):
    generator = torch.Generator().manual_seed(seed)
    rows = [2 * np.pi * torch.rand(ANSATZ_REPS + 1, N_QUBITS, 2, generator=generator, dtype=torch.float64),
            torch.stack([torch.ones(N_QUBITS), torch.zeros(N_QUBITS)], dim=-1)[None].double(),
            torch.zeros(KAN_BASIS // 2, N_QUBITS, 2, dtype=torch.float64)]
    out = torch.zeros(1, N_QUBITS, 2, dtype=torch.float64)
    out[0, 0, 0] = np.log(2.0)
    return torch.nn.Parameter(torch.cat(rows + [out]))


def kan_basis(x):
    centers = torch.linspace(0, np.pi, KAN_BASIS, dtype=torch.float64)
    width = np.pi / (KAN_BASIS - 1)
    return torch.exp(-((torch.as_tensor(x, dtype=torch.float64)[..., None] - centers) / width) ** 2)


def light_cone(ops, readout_wire):
    live, only_z, kept = {readout_wire}, True, []
    for op in reversed(ops):
        wires = set(op.wires.tolist())
        if not wires & live:
            continue
        if only_z and (op.name in DIAGONAL_GATES or (op.name == "CNOT" and op.wires[0] == readout_wire)):
            continue
        kept.append(op)
        live |= wires
        only_z = False
    return kept[::-1]


def gates(x, weights):
    scale = weights[ANSATZ_REPS + 1]
    kan = weights[ANSATZ_REPS + 2:ANSATZ_REPS + 2 + KAN_BASIS // 2].permute(1, 0, 2).reshape(N_QUBITS, KAN_BASIS)
    for i in range(N_QUBITS):
        qml.RY(scale[i, 0] * x[..., i] + scale[i, 1] + kan_basis(x[..., i]) @ kan[i], wires=i)
    for layer in range(ANSATZ_REPS + 1):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        if layer < ANSATZ_REPS:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])


def _circuit(x, weights):
    with qml.queuing.AnnotatedQueue() as queue:
        gates(x, weights)
    for op in light_cone(queue.queue, READOUT_WIRE):
        qml.apply(op)
    return qml.expval(qml.PauliZ(READOUT_WIRE))


circuit = qml.transforms.cancel_inverses(qml.transforms.merge_rotations(_circuit))


def to_score(z):
    return (1 - z) / 2


def to_probability(z, weights):
    return torch.sigmoid(-torch.exp(weights[-1, 0, 0]) * z + weights[-1, 0, 1])


def clean_qnode():
    return qml.QNode(circuit, qml.device("default.qubit", wires=N_QUBITS), interface="torch", diff_method="backprop")


def noisy_qnode(p):
    dev = qml.noise.insert(qml.device("default.mixed", wires=N_QUBITS), qml.DepolarizingChannel, p, position="all")
    return qml.transforms.decompose(qml.QNode(circuit, dev, interface="torch", diff_method="backprop"),
                                    gate_set=NATIVE_GATES)


def predict(qnode, Z, weights, chunk=2000):
    with torch.no_grad():
        return np.concatenate([to_score(qnode(torch.tensor(Z[i:i + chunk]), weights)).numpy()
                               for i in range(0, len(Z), chunk)])


def spsa_gradient(loss_of, weights, c, q, generator):
    g = torch.zeros_like(weights)
    with torch.no_grad():
        for _ in range(q):
            delta = 2 * torch.randint(0, 2, weights.shape, generator=generator).to(weights.dtype) - 1
            g += (loss_of(weights + c * delta) - loss_of(weights - c * delta)) / (2 * c) * delta
    return g / q


def train(Z_train, y_train, Z_val, y_val, seed, args):
    set_seed(seed)
    qnode = clean_qnode()
    weights = init_weights(seed)
    Xt, yt = torch.tensor(Z_train), torch.tensor(y_train, dtype=torch.float64)
    sampler = StratifiedSampler(y_train, args.batch_size, seed)
    if args.val == "cheap":
        subset, sub_weights = validation_subset(y_val, args.val_neg_rate, seed)
        Z_check, y_check = Z_val[subset], y_val[subset]
    else:
        Z_check, y_check, sub_weights = Z_val, y_val, None
    runs = dict(train=0, val=0, ps_examples=0)
    best = dict(ap=-1.0, weights=weights.detach().clone(), step=0, since=0)

    def minibatch_loss(v_model=None):
        idx, w = sampler.draw()
        return lambda v: torch.nn.functional.binary_cross_entropy(
            to_probability(qnode(Xt[idx], v), v), yt[idx], weight=w, reduction="sum")

    def validate(w, step):
        runs["val"] += len(Z_check)
        ap = average_precision_score(y_check, predict(qnode, Z_check, w), sample_weight=sub_weights)
        if ap > best["ap"]:
            best.update(ap=ap, weights=w.detach().clone(), step=step, since=0)
        else:
            best["since"] += 1
        return best["since"] >= args.patience

    if args.optimizer == "adam":
        P = trainable_parameters(weights)
        optimizer = torch.optim.Adam([weights], lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            for _ in range(args.steps_per_epoch):
                optimizer.zero_grad()
                minibatch_loss()(weights).backward()
                optimizer.step()
            examples = args.steps_per_epoch * args.batch_size
            runs["train"] += examples * (1 + 2 * P)
            runs["ps_examples"] += examples
            if epoch % args.val_every == 0 or epoch == args.epochs:
                if validate(weights, epoch):
                    break
    else:
        w = weights.detach().clone()
        generator = torch.Generator().manual_seed(seed)
        A, c = 0.1 * args.spsa_steps, args.spsa_c
        g0 = spsa_gradient(minibatch_loss(), w, c, 10, generator)
        runs["train"] += 2 * 10 * args.batch_size
        a = args.spsa_target_step * (A + 1) ** 0.602 / max(g0.abs().mean().item(), 1e-12)
        for k in range(1, args.spsa_steps + 1):
            g = spsa_gradient(minibatch_loss(), w, c / k ** 0.101, args.spsa_q, generator)
            w = w - a / (k + A) ** 0.602 * g
            runs["train"] += 2 * args.spsa_q * args.batch_size
            if k % args.spsa_val_every == 0 or k == args.spsa_steps:
                if validate(w, k):
                    break
    return best["weights"], best["step"], runs


def threshold_from_validation(y_val, scores):
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(AUPRC=average_precision_score(y, scores), AUROC=roc_auc_score(y, scores),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0), MCC=matthews_corrcoef(y, pred), TP=tp, FP=fp, FN=fn, TN=tn)


def evaluate(weights, Z_val, y_val, Z_test, y_test, seed, label, levels):
    clean = clean_qnode()
    clean_test = predict(clean, Z_test, weights)
    rows = []
    for p in levels:
        if p == 0:
            val_scores, test_scores = predict(clean, Z_val, weights), clean_test
        else:
            noisy = noisy_qnode(p)
            val_scores, test_scores = predict(noisy, Z_val, weights), predict(noisy, Z_test, weights)
        threshold = threshold_from_validation(y_val, val_scores)
        rows.append(dict(seed=seed, model=label, p=p, threshold=threshold,
                         val_AUPRC=average_precision_score(y_val, val_scores),
                         spearman_rho=1.0 if p == 0 else spearmanr(clean_test, test_scores).statistic,
                         **metrics(y_test, test_scores, threshold)))
    return rows


def trainable_parameters(weights):
    marked = weights.detach().clone().requires_grad_(True)
    tape = qml.tape.make_qscript(circuit)(torch.rand(N_QUBITS, dtype=torch.float64) * np.pi, marked)
    return sum(any(torch.is_tensor(d) and d.requires_grad for d in op.data) for op in tape.operations)


def circuit_resources(weights):
    qnode = qml.transforms.decompose(qml.QNode(circuit, qml.device("default.qubit", wires=N_QUBITS)),
                                     gate_set=NATIVE_GATES)
    resources = qml.specs(qnode)(torch.rand(N_QUBITS, dtype=torch.float64) * np.pi, weights)["resources"]
    return resources.num_gates, sum(size * count for size, count in resources.gate_sizes.items())


def tracker_per_example(weights, batch_size=32):
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.random((batch_size, N_QUBITS)) * np.pi)
    y = torch.tensor((np.arange(batch_size) % 2).astype(float))
    dev = qml.device("default.qubit", wires=N_QUBITS, seed=0)
    qnode = qml.set_shots(qml.QNode(circuit, dev, interface="torch", diff_method="parameter-shift"), shots=SHOTS)
    w = weights.detach().clone().requires_grad_(True)
    with qml.Tracker(dev) as tracker:
        for i in range(batch_size):
            torch.nn.functional.binary_cross_entropy(to_probability(qnode(x[i], w), w), y[i]).backward()
    return tracker.totals["executions"] / batch_size


def cost_row(weights, runs, seed, best_step, args):
    P = trainable_parameters(weights)
    gate_count, noise_sites = circuit_resources(weights)
    per_example = tracker_per_example(weights)
    train_tracker = runs["train"] - runs["ps_examples"] * (1 + 2 * P) + runs["ps_examples"] * per_example
    total, total_tracker = runs["train"] + runs["val"], train_tracker + runs["val"]
    return dict(seed=seed, optimizer=args.optimizer, validation=args.val, best_epoch_or_step=best_step,
                qubits=N_QUBITS, circuit_parameters=P, gates=gate_count, noise_locations=noise_sites,
                tracker_per_example=per_example, training_executions=runs["train"],
                training_executions_tracker=train_tracker, validation_executions=runs["val"],
                train_val_executions=total, train_val_executions_tracker=total_tracker,
                train_val_shots=SHOTS * total, train_val_shots_tracker=SHOTS * total_tracker)


def main(args):
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_split()
    seeds = args.seeds
    if args.quick:
        seeds, args.epochs, args.spsa_steps = seeds[:1], 3, min(args.spsa_steps, 100)
        vi, _ = train_test_split(np.arange(len(y_val)), train_size=2000, stratify=y_val, random_state=SEED_SPLIT)
        ti, _ = train_test_split(np.arange(len(y_test)), train_size=2000, stratify=y_test, random_state=SEED_SPLIT)
        X_val, y_val, X_test, y_test = X_val.iloc[vi], y_val[vi], X_test.iloc[ti], y_test[ti]
        print("ATENTIE --quick: metricile NU sunt rezultate finale.")
    label = "kan_quant_" + args.optimizer + ("_cheapval" if args.val == "cheap" else "")
    rows, costs = [], []
    for seed in seeds:
        X_bal, y_bal = undersample(X_tr, y_tr, seed)
        encoder = QuantileEncoder(seed).fit(X_bal)
        Z_bal, Z_tr, Z_val, Z_test = (encoder.transform(X) for X in (X_bal, X_tr, X_val, X_test))
        weights, best_step, runs = train(Z_tr, y_tr, Z_val, y_val, seed, args)
        rows.extend(evaluate(weights, Z_val, y_val, Z_test, y_test, seed, label, args.noise))
        costs.append(cost_row(weights, runs, seed, best_step, args))
        ref = LogisticRegression(max_iter=1000).fit(Z_bal, y_bal)
        ref_val, ref_test = ref.predict_proba(Z_val)[:, 1], ref.predict_proba(Z_test)[:, 1]
        threshold = threshold_from_validation(y_val, ref_val)
        rows.extend(dict(seed=seed, model="logistic_regression", p=p, threshold=threshold,
                         val_AUPRC=average_precision_score(y_val, ref_val), spearman_rho=1.0,
                         **metrics(y_test, ref_test, threshold)) for p in args.noise)
        ap = next(r["AUPRC"] for r in reversed(rows) if r["model"] == label and r["seed"] == seed and r["p"] == max(args.noise))
        print(f"seed {seed}: best epoch/step {best_step} | test AUPRC p={max(args.noise)}: {ap:.4f} | "
              f"executii train+val: {costs[-1]['train_val_executions']:,}", flush=True)

    result, cost = pd.DataFrame(rows), pd.DataFrame(costs)
    names = ["AUPRC", "AUROC", "spearman_rho", "precision", "recall", "F1", "MCC", "val_AUPRC"]
    summary = result.groupby(["model", "p"])[names].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    result.to_csv(ROOT / f"{args.out}_runs.csv", index=False)
    summary.to_csv(ROOT / f"{args.out}_summary.csv")
    cost.to_csv(ROOT / f"{args.out}_cost.csv", index=False)
    fmt = lambda m: summary[f"{m}_mean"].map("{:.3f}".format) + " ± " + summary[f"{m}_std"].map("{:.3f}".format)
    pd.set_option("display.width", 200)
    print("\nREZULTATE (test, media ± std pe seeds):")
    print(pd.DataFrame({m: fmt(m) for m in ["AUPRC", "AUROC", "spearman_rho", "F1", "MCC"]}).to_string())
    print("\nCOST (medie pe rulare de training; shots = 1024 / executie):")
    print(cost[["training_executions", "training_executions_tracker", "validation_executions",
                "train_val_executions", "train_val_executions_tracker", "train_val_shots"]].mean().round(0).to_string())
    print(f"\nSalvat: {args.out}_runs.csv, {args.out}_summary.csv, {args.out}_cost.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimizer", choices=["adam", "spsa"], default="adam")
    parser.add_argument("--val", choices=["full", "cheap"], default=None,
                        help="full = tot validation dupa fiecare epoca; cheap = subset stratificat ponderat "
                             "(implicit: full pentru Adam, cheap pentru SPSA)")
    parser.add_argument("--val-neg-rate", type=float, default=0.05, help="fractiunea de legitime in validarea ieftina")
    parser.add_argument("--val-every", type=int, default=1, help="epoci Adam intre validari")
    parser.add_argument("--patience", type=int, default=10**9, help="validari fara imbunatatire pana la oprire")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=19, help="19 x 32 ~ 590 = o epoca a baseline-ului")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--spsa-steps", type=int, default=600)
    parser.add_argument("--spsa-q", type=int, default=2)
    parser.add_argument("--spsa-target-step", type=float, default=0.03)
    parser.add_argument("--spsa-c", type=float, default=0.05)
    parser.add_argument("--spsa-val-every", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--noise", type=float, nargs="+", default=NOISE)
    parser.add_argument("--out", default="kan_vqc")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.val is None:
        args.val = "cheap" if args.optimizer == "spsa" else "full"
    if args.optimizer == "spsa" and args.patience == 10**9:
        args.patience = 6
    main(args)