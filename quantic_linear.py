import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
SEEDS = range(5)
NOISE_LEVELS = (0., .001, .01, .05)
NATIVE_GATES = {'RX', 'RY', 'RZ', 'Rot', 'Hadamard', 'CNOT', 'CZ'}
DT = torch.float64


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split():
    df = pd.read_csv(ROOT / 'data' / 'creditcard.csv')
    X = df.drop(columns='Class')
    y = df.Class.to_numpy(dtype=int)
    f = ROOT / 'split_indices.npz'
    if not f.exists():
        a, te = train_test_split(np.arange(len(y)), test_size=.2,
            random_state=42, stratify=y)
        tr, va = train_test_split(a, test_size=.25, random_state=42,
            stratify=y[a])
        np.savez(f, train=tr, val=va, test=te)
        print('ATENTIE: split_indices.npz a fost generat; verifica potrivirea cu splitul oficial.')
    with np.load(f) as s:
        tr, va, te = (s[k] for k in ('train', 'val', 'test'))
    return X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], y[te]


def undersample(X, y, seed):
    rng = np.random.default_rng(seed)
    p = np.flatnonzero(y == 1)
    n = rng.choice(np.flatnonzero(y == 0), size=len(p), replace=False)
    ix = rng.permutation(np.r_[p, n])
    return X.iloc[ix], y[ix]


def validation_selection(X, y, negatives, seed=42):
    if negatives <= 0 or negatives >= int((y == 0).sum()):
        return X, y, np.ones(len(y), dtype=np.float64)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    pick = rng.choice(neg, size=negatives, replace=False)
    ix = np.sort(np.r_[pos, pick])
    weights = np.where(y[ix] == 1, 1., len(neg) / negatives)
    return X.iloc[ix], y[ix], weights


def fit_baseline_encoder(X_bal, seed):
    ss = StandardScaler().fit(X_bal[['Time', 'Amount']])
    z = X_bal.copy()
    z[['Time', 'Amount']] = ss.transform(X_bal[['Time', 'Amount']])
    pca = PCA(n_components=4, random_state=seed).fit(z.to_numpy())
    scaler = MinMaxScaler(feature_range=(0., np.pi)).fit(pca.transform(z))
    return {'kind': 'pca', 'feature_columns': list(X_bal.columns),
            'time_mean': ss.mean_.copy(), 'time_scale': ss.scale_.copy(),
            'pca_mean': pca.mean_.copy(), 'pca_components': pca.components_.copy(),
            'angle_scale': scaler.scale_.copy(), 'angle_min': scaler.min_.copy()}


def transform_baseline(X, enc):
    z = X.loc[:, enc['feature_columns']].to_numpy(dtype=np.float64).copy()
    names = enc['feature_columns']
    for j, col in enumerate(('Time', 'Amount')):
        i = names.index(col)
        z[:, i] = (z[:, i] - enc['time_mean'][j]) / enc['time_scale'][j]
    z = (z - enc['pca_mean']) @ enc['pca_components'].T
    return np.clip(z * enc['angle_scale'] + enc['angle_min'], 0., np.pi)


class LinearAngles(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint['encoder_state']
        input_size = state['weight'].shape[1]
        output_size = state['weight'].shape[0]
        self.layer = torch.nn.Linear(input_size, output_size).double()
        self.layer.load_state_dict(state)

    def forward(self, X_std):
        return torch.pi * torch.sigmoid(self.layer(X_std))


def standardized_input(X, enc):
    data = X.loc[:, enc['feature_columns']].to_numpy(dtype=np.float64)
    mean = np.asarray(enc['scaler_mean'], dtype=np.float64)
    scale = np.asarray(enc['scaler_scale'], dtype=np.float64)
    return torch.as_tensor((data - mean) / scale, dtype=DT)


def get_linear_encoder(seed, runfolder, epochs, retrain, X_train, y_train, X_val, y_val):
    f = runfolder / f'linear_seed{seed}.pt'
    if f.exists() and not retrain:
        print('  Encoder din ACEST experiment:', f)
        return torch.load(f, map_location='cpu', weights_only=True)
    seed_everything(seed)
    scaler = StandardScaler().fit(X_train)
    xt = torch.as_tensor(scaler.transform(X_train), dtype=torch.float32)
    xv = torch.as_tensor(scaler.transform(X_val), dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    encoder = torch.nn.Linear(xt.shape[1], 4)
    head = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=.003)
    npos = max(int((y_train == 1).sum()), 1)
    nneg = int((y_train == 0).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos, dtype=torch.float32))
    best_ap, best_state, stale = -1., None, 0
    for epoch in range(1, epochs + 1):
        for idx in torch.randperm(len(xt)).split(2048):
            opt.zero_grad()
            angles = torch.pi * torch.sigmoid(encoder(xt[idx]))
            loss = criterion(head(angles).flatten(), yt[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            score = torch.sigmoid(head(torch.pi * torch.sigmoid(encoder(xv)))).flatten().numpy()
        ap = float(average_precision_score(y_val, score))
        print(f'  Encoder NOU epoch={epoch}/{epochs}, val AUPRC={ap:.4f}', flush=True)
        if ap > best_ap + 1e-4:
            best_ap, stale = ap, 0
            best_state = (copy.deepcopy(encoder.state_dict()),
                          copy.deepcopy(head.state_dict()), epoch)
        else:
            stale += 1
            if stale >= 4:
                break
    enc_state, head_state, best_epoch = best_state
    enc = {
        'kind': 'linear', 'encoder_state': enc_state, 'head_state': head_state,
        'feature_columns': list(X_train.columns), 'n_angles': 4,
        'scaler_mean': torch.as_tensor(scaler.mean_, dtype=DT),
        'scaler_scale': torch.as_tensor(scaler.scale_, dtype=DT),
        'seed': seed, 'best_epoch': best_epoch, 'best_ap_linear': best_ap,
    }
    torch.save(enc, f)
    print(f'  Encoder NOU salvat: {f} (best AP={best_ap:.4f})', flush=True)
    return enc


def encode_np(X, enc):
    if enc['kind'] == 'pca':
        return transform_baseline(X, enc)
    with torch.no_grad():
        return LinearAngles(enc).eval()(standardized_input(X, enc)).numpy()


def baseline(x, w):
    for _ in range(2):
        for q in range(4):
            qml.Hadamard(wires=q)
            qml.RZ(2 * x[..., q], wires=q)
        for i in range(4):
            for j in range(i+1, 4):
                phi = 2 * (np.pi-x[..., i]) * (np.pi-x[..., j])
                qml.CNOT(wires=[i, j])
                qml.RZ(phi, wires=j)
                qml.CNOT(wires=[i, j])
    for layer in range(4):
        for q in range(4):
            qml.RY(w[layer, q, 0], wires=q)
            qml.RZ(w[layer, q, 1], wires=q)
        if layer < 3:
            for q in range(3):
                qml.CNOT(wires=[q, q+1])
    return qml.expval(qml.PauliZ(0))


def improved(x, w):
    for q in range(4):
        qml.RY(x[..., q], wires=q)
    for layer in range(3):
        for q in range(4):
            qml.RY(w[layer, q, 0], wires=q)
            qml.RZ(w[layer, q, 1], wires=q)
        if layer < 2:
            for q in range(3):
                qml.CNOT(wires=[q, q+1])
    return qml.expval(qml.PauliZ(0))


def reverse2(x, w):
    for q in range(4):
        qml.Rot(torch.zeros_like(x[..., q]), x[..., q] + w[2*q], w[2*q+1], wires=q)
    for q in (3, 2, 1):
        qml.CNOT(wires=[q, q-1])
    for q in range(4):
        qml.Rot(torch.zeros_like(w[8+2*q]), w[8+2*q], w[9+2*q], wires=q)
    for q in (3, 2, 1):
        qml.CNOT(wires=[q, q-1])
    qml.RY(w[16], wires=0)
    return qml.expval(qml.PauliZ(0))


def pruned(x, w):
    for q in range(3):
        qml.Rot(torch.zeros_like(x[..., q]), x[..., q] + w[2*q], w[2*q+1], wires=q)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    for q in range(2):
        qml.Rot(torch.zeros_like(w[6+2*q]), w[6+2*q], w[7+2*q], wires=q)
    qml.CNOT(wires=[0, 1])
    qml.RY(w[10], wires=0)
    return qml.expval(qml.PauliZ(0))


CIRCUITS = {'baseline': (baseline, 4, (4,4,2)),
            'improved': (improved, 4, (3,4,2)),
            'reverse2': (reverse2, 4, (17,)),
            'pruned': (pruned, 3, (11,))}


def make_qnode(name, p=0.):
    fn, n, _ = CIRCUITS[name]
    if p == 0:
        return qml.QNode(fn, qml.device('default.qubit', wires=n),
            interface='torch', diff_method='backprop')
    dev = qml.noise.insert(qml.device('default.mixed', wires=n),
                           qml.DepolarizingChannel, p, position='all')
    return qml.transforms.decompose(
        qml.QNode(fn, dev, interface='torch', diff_method='backprop'),
        gate_set=NATIVE_GATES)


def init_w(name, seed, init):
    rng = torch.Generator().manual_seed(seed)
    shape = CIRCUITS[name][2]
    if init == 'small':
        x = .15 * torch.randn(*shape, generator=rng, dtype=DT)
    else:
        x = 2 * np.pi * torch.rand(*shape, generator=rng, dtype=DT)
    return torch.nn.Parameter(x)


def scores(qnode, Z, w, chunk):
    res = []
    with torch.no_grad():
        for k in range(0, len(Z), chunk):
            t = torch.as_tensor(Z[k:k+chunk], dtype=DT)
            value = qnode(t, w)
            res.append(((1. - value)/2.).detach().cpu().numpy().reshape(-1))
    return np.concatenate(res)


def make_enc_model(enc):
    if enc['kind'] == 'pca':
        return None
    return LinearAngles(enc).double()


def initial_enc_snapshot(enc, name):
    if name == 'baseline':
        return copy.deepcopy(enc)
    return copy.deepcopy(enc)


def weighted_ap(y, p, sw):
    return float(average_precision_score(y, p, sample_weight=sw))


def model_file(folder, name, seed):
    return folder / f'{name}_seed{seed}.pt'


def fit_one(name, seed, folder, data, args):
    X_tr, y_tr, X_val, y_val, _, _ = data
    target = model_file(folder, name, seed)
    if target.exists() and not args.overwrite:
        print('[EXISTA] Nu reantrenez:', target, '(foloseste alt tag sau --overwrite)')
        return
    seed_everything(seed)
    X_bal, y_bal = undersample(X_tr, y_tr, seed)
    if name == 'baseline':
        enc = fit_baseline_encoder(X_bal, seed)
    else:
        enc = get_linear_encoder(seed, folder, args.encoder_epochs,
            args.retrain_encoder, X_tr, y_tr, X_val, y_val)
        enc = copy.deepcopy(enc)
        enc['kind'] = 'linear'
    Xvs, yvs, sw = validation_selection(X_val, y_val, args.val_negatives)
    Ztrain = encode_np(X_bal, enc)
    Zval = encode_np(Xvs, enc)
    if name == 'pruned':
        Ztrain, Zval = Ztrain[:, :3], Zval[:, :3]
    if name == 'improved':
        for dim in (3, 4):
            ref = LogisticRegression(max_iter=1000).fit(Ztrain[:, :dim], y_bal)
            ap_lr = weighted_ap(yvs, ref.predict_proba(Zval[:, :dim])[:, 1], sw)
            print(f'  [diagnostic] LR pe {dim} unghiuri: AP aproximativ={ap_lr:.4f}')
        if 'head_state' in enc:
            h = enc['head_state']
            hw = h['weight'].detach().cpu().numpy().reshape(-1)
            hb = float(h['bias'].detach().cpu().numpy()[0])
            logits = Zval @ hw + hb
            cls = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
            print(f'  [diagnostic] Head-ul encoderului preantrenat: '
                  f'AP aproximativ={weighted_ap(yvs, cls, sw):.4f}')
    seed_everything(seed)
    y_t = torch.as_tensor(y_bal, dtype=DT)
    x_t = torch.as_tensor(Ztrain, dtype=DT)
    qnode = make_qnode(name)
    w = init_w(name, seed, args.init)
    opt = torch.optim.Adam([w], lr=args.lr)
    loss_fn = torch.nn.BCELoss()
    best = {'ap': -1., 'weights': None, 'encoder': None, 'epoch': 0, 'stage': ''}
    checks = 0

    def score_current(Xangles):
        return scores(qnode, Xangles, w, args.chunk)

    def remember(stage, ep, validation_angles, current_enc):
        nonlocal checks
        checks += 1
        ap = weighted_ap(yvs, score_current(validation_angles), sw)
        print(f'  {name}, seed={seed}, {stage} ep={ep} '
              f'AP estimat={ap:.4f} | validare: {len(yvs)} ({(yvs==1).sum()} fraude)',
              flush=True)
        if ap > best['ap']:
            best.update(ap=ap, weights=w.detach().clone(),
                        encoder=copy.deepcopy(current_enc), epoch=ep, stage=stage)
            torch.save({'model': name, 'seed': seed, 'weights': best['weights'],
                        'encoder': best['encoder'], 'best_ap_approx': best['ap'],
                        'best_epoch': ep, 'best_stage': stage,
                        'train_count': len(y_bal), 'n_val_per_check': len(yvs),
                        'val_checks_so_far': checks,
                        'epochs_clean': args.epochs,
                        'joint_epochs': args.joint_epochs,
                        'init': args.init, 'learning_rate': args.lr}, target)

    for ep in range(1, args.epochs+1):
        order = torch.randperm(len(y_t))
        for ix in order.split(args.batch_size):
            opt.zero_grad()
            p = ((1. - qnode(x_t[ix], w))/2.).clamp(1e-8, 1-1e-8).reshape(-1)
            loss = loss_fn(p, y_t[ix])
            loss.backward()
            opt.step()
        if ep % args.val_every == 0 or ep == args.epochs:
            remember('clean', ep, Zval, enc)

    if args.joint_epochs > 0 and name != 'baseline':
        with torch.no_grad():
            w.copy_(best['weights'])
        enc = copy.deepcopy(best['encoder'])
        em = make_enc_model(enc)
        Xtr_std = standardized_input(X_bal, enc)
        Xva_std = standardized_input(Xvs, enc)
        joint_opt = torch.optim.Adam([{'params': [w], 'lr': args.lr * .25},
                                      {'params': em.parameters(), 'lr': args.encoder_lr}])
        for ep in range(1, args.joint_epochs+1):
            order = torch.randperm(len(y_t))
            for ix in order.split(args.batch_size):
                joint_opt.zero_grad()
                angles = em(Xtr_std[ix])[:, :CIRCUITS[name][1]]
                p = ((1. - qnode(angles, w))/2.).clamp(1e-8, 1-1e-8).reshape(-1)
                loss = loss_fn(p, y_t[ix]); loss.backward(); joint_opt.step()
            with torch.no_grad():
                joint_zval = em(Xva_std).cpu().numpy()[:, :CIRCUITS[name][1]]
            current_enc = copy.deepcopy(enc)
            current_enc['encoder_state'] = {
                k: v.detach().clone() for k, v in em.layer.state_dict().items()}
            remember('joint', ep, joint_zval, current_enc)
    if target.exists():
        saved = torch.load(target, map_location='cpu', weights_only=False)
        saved['val_checks'] = checks
        torch.save(saved, target)
    print('  BEST SALVAT:', target, '| AP estimate:', round(best['ap'], 5),
          '| etapa:', best['stage'], flush=True)


def threshold_f1(y, p, sw=None):
    precision, recall, thresholds = precision_recall_curve(y, p, sample_weight=sw)
    if len(thresholds) == 0:
        return .5
    f = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f)])


def metrics(y, p, th, sw=None):
    pred = (p >= th).astype(int)
    return {'AUPRC': weighted_ap(y, p, sw),
            'AUROC': float(roc_auc_score(y, p, sample_weight=sw)),
            'F1': float(f1_score(y, pred, sample_weight=sw, zero_division=0)),
            'precision': float(precision_score(y, pred, sample_weight=sw, zero_division=0)),
            'recall': float(recall_score(y, pred, sample_weight=sw, zero_division=0)),
            'MCC': float(matthews_corrcoef(y, pred, sample_weight=sw))}


def load_model(name, seed, folder):
    f = model_file(folder, name, seed)
    if not f.exists():
        raise FileNotFoundError(f'{f} nu exista: mai intai ruleaza train.')
    d = torch.load(f, map_location='cpu', weights_only=False)
    assert d['model'] == name and d['seed'] == seed
    return d


def estimate_cost(name, saved):
    fn, n, _ = CIRCUITS[name]
    device = qml.device('default.qubit', wires=n)
    node = qml.transforms.decompose(qml.QNode(fn, device), gate_set=NATIVE_GATES)
    dummy = torch.full((n,), .5, dtype=DT)
    resource = qml.specs(node)(dummy, saved['weights'])['resources']
    gates = int(resource.num_gates)
    noise_sites = sum(int(size) * int(count) for size, count in resource.gate_sizes.items())
    P = int(saved['weights'].numel())
    E = int(saved['epochs_clean'])
    J = int(saved['joint_epochs']) if name != 'baseline' else 0
    N = int(saved['train_count'])
    if saved.get('derived_from'):
        P_train, n_train = 24, 4
    else:
        P_train, n_train = P, n
    train_budget = E*N*(1+2*P_train) + J*N*(1+2*P_train+2*n_train)
    validation_budget = int(saved.get('val_checks', 0)) * int(saved['n_val_per_check'])
    budget = train_budget + validation_budget
    return {'qubits': n, 'parameters': P, 'gates': gates,
            'noise_sites': noise_sites,
            'hardware_train_execution_upper_bound': train_budget,
            'checkpoint_validation_executions': validation_budget,
            'train_and_validation_execution_upper_bound': budget,
            'train_and_validation_shots_upper_bound': 1024*budget}


def evaluate_one(name, seed, folder, data, args):
    Xtr, ytr, Xv, yv, Xt, yt = data
    d = load_model(name, seed, folder)
    enc, w = d['encoder'], d['weights']
    if args.official or args.full_val:
        xv, yval, sw = Xv, yv, np.ones(len(yv))
    else:
        xv, yval, sw = validation_selection(Xv, yv, args.val_negatives)
    zv = encode_np(xv, enc)[:, :CIRCUITS[name][1]]
    zt = encode_np(Xt, enc)[:, :CIRCUITS[name][1]] if args.official else None
    clean_node = make_qnode(name)
    clean_v = scores(clean_node, zv, w, args.chunk)
    clean_t = scores(clean_node, zt, w, args.chunk) if args.official else None
    output = []
    for noise in args.noise:
        if noise == 0:
            noisy_v, noisy_t = clean_v, clean_t
        else:
            node = make_qnode(name, p=noise)
            noisy_v = scores(node, zv, w, args.chunk)
            noisy_t = scores(node, zt, w, args.chunk) if args.official else None
        th = threshold_f1(yval, noisy_v, sw)
        if args.official:
            actual_y, actual_scores, clean_scores, weights = yt, noisy_t, clean_t, None
            split = 'test'
        else:
            actual_y, actual_scores, clean_scores, weights = yval, noisy_v, clean_v, sw
            split = 'validation_weighted' if not args.full_val else 'validation_full'
        row = {'model': name, 'seed': seed, 'noise': noise, 'split': split,
               'threshold': th, 'spearman': 1. if noise == 0 else float(spearmanr(
                    clean_scores, actual_scores).statistic),
               **metrics(actual_y, actual_scores, th, weights)}
        output.append(row)
        print(f'{name:9s} seed={seed} p={noise:0.3f}: '
              f'AUPRC={row["AUPRC"]:.4f}, rho={row["spearman"]:.4f}, '
              f'F1={row["F1"]:.4f} [{split}]', flush=True)
    return output, estimate_cost(name, d)


def classical_reference(name, seed, folder, data):
    Xtr, ytr, Xv, yv, Xt, yt = data
    ck = load_model(name, seed, folder)
    Xb, yb = undersample(Xtr, ytr, seed)
    enc = ck['encoder']
    dim = CIRCUITS[name][1]
    zb = encode_np(Xb, enc)[:, :dim]
    zv = encode_np(Xv, enc)[:, :dim]
    zt = encode_np(Xt, enc)[:, :dim]
    lr = LogisticRegression(max_iter=1000).fit(zb, yb)
    s_val = lr.predict_proba(zv)[:, 1]
    s_test = lr.predict_proba(zt)[:, 1]
    th = threshold_f1(yv, s_val)
    return [{'model': 'LR_' + name, 'seed': seed, 'noise': p, 'split': 'test',
             'threshold': th, 'spearman': 1., **metrics(yt, s_test, th)}
            for p in NOISE_LEVELS]


def transfer_improved(seed, folder, data, overwrite=False):
    """Conversie EXACTA (clean), folosind ACELASI encoder si aceleasi ponderi."""
    source = load_model('improved', seed, folder)
    f = model_file(folder, 'pruned', seed)
    if f.exists() and not overwrite:
        print('[EXISTA] pruned:', f, '--overwrite ca sa inlocuiesti')
        return
    W = source['weights']
    assert tuple(W.shape) == (3,4,2)
    v = torch.cat((W[0, :3, :].reshape(-1), W[1, :2, :].reshape(-1),
                   W[2, 0, :1].reshape(-1))).clone()
    new = copy.deepcopy(source)
    new['model'] = 'pruned'
    new['weights'] = v
    new['derived_from'] = 'improved (no extra training, clean equivalence)'
    Xv = data[2]
    Z = encode_np(Xv.iloc[:128], source['encoder'])
    a = scores(make_qnode('improved'), Z, W, 128)
    b = scores(make_qnode('pruned'), Z[:, :3], v, 128)
    maxdiff = np.max(np.abs(a - b))
    print('TRANSFER | diferenta maxima clean (128 tranzactii):', f'{maxdiff:.3e}')
    if maxdiff > 1e-8:
        raise AssertionError('Pruning-ul NU este echivalent in aceasta implementare!')
    torch.save(new, f)
    print('Salvat:', f)


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('mode', choices=['sanity', 'train', 'eval', 'quick', 'transfer'])
    pa.add_argument('--tag', default='exp_fix')
    pa.add_argument('--models', nargs='+', choices=sorted(CIRCUITS), default=['improved','reverse2'])
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--all-seeds', action='store_true')
    pa.add_argument('--epochs', type=int, default=12)
    pa.add_argument('--joint-epochs', type=int, default=0,
            help='Fine-tuning end-to-end encoder + VQC, DUPA training clean (default 0)')
    pa.add_argument('--encoder-epochs', type=int, default=12,
            help='Epoci pretraining encoder NOU pe train (numai validare pentru early stop)')
    pa.add_argument('--retrain-encoder', action='store_true',
            help='Forteaza reantrenarea encoderului din acest experiment')
    pa.add_argument('--lr', type=float, default=.05)
    pa.add_argument('--encoder-lr', type=float, default=.0005)
    pa.add_argument('--init', choices=['uniform','small'], default='uniform')
    pa.add_argument('--batch-size', type=int, default=32)
    pa.add_argument('--val-every', type=int, default=3)
    pa.add_argument('--val-negatives', type=int, default=5000,
            help='Nr. negative pastrate la validare; TOATE fraudele; 0=validation complet')
    pa.add_argument('--chunk', type=int, default=1024)
    pa.add_argument('--noise', nargs='+', type=float, default=[0., .05])
    pa.add_argument('--full-val', action='store_true')
    pa.add_argument('--official', action='store_true', help='TEST complet; doar dupa tuning')
    pa.add_argument('--overwrite', action='store_true')
    args = pa.parse_args()
    if args.mode == 'sanity':
        for name in ('baseline', 'improved', 'reverse2', 'pruned'):
            n = CIRCUITS[name][1]
            x = torch.randn(3, n, dtype=DT)
            w = init_w(name, 0, 'small')
            z = make_qnode(name)(x, w)
            assert z.shape == (3,), (name, z.shape)
            (z.sum()).backward()
            assert w.grad is not None and torch.isfinite(w.grad).all(), name
            print(f'[OK] {name}: batch=3, backprop OK', flush=True)
        x = torch.randn(1, 4, dtype=DT)
        w = init_w('reverse2', 0, 'small')
        z = make_qnode('reverse2', p=.05)(x, w)
        assert torch.isfinite(z).all()
        print('[OK] reverse2 noise p=0.05', flush=True)
        return
    if args.tag in ('', '.', '..') or '/' in args.tag or '\\' in args.tag:
        pa.error('--tag trebuie sa fie un nume simplu de folder.')
    if args.val_every < 1 or args.epochs < 1:
        pa.error('epochs si val-every trebuie sa fie pozitive')
    folder = ROOT / 'vqc_runs' / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    data = load_split()
    seeds = SEEDS if args.all_seeds else [args.seed]
    if args.mode == 'transfer':
        for seed in seeds:
            transfer_improved(seed, folder, data, args.overwrite)
        return
    if args.mode in ('train', 'quick'):
        if args.mode == 'quick':
            args.epochs = 3
            args.joint_epochs = 0
            args.val_every = 3
            args.val_negatives = 2000
            print('QUICK: 3 epoci, TOATE fraudele din validation, 2000 negative.')
        for seed in seeds:
            for name in args.models:
                fit_one(name, seed, folder, data, args)
        return
    if args.official and not args.all_seeds:
        print('ATENTIE: evaluare test 1 seed nu reprezinta scor oficial 5 seeds!')
    rows, costs = [], []
    for seed in seeds:
        for name in args.models:
            current, cost = evaluate_one(name, seed, folder, data, args)
            rows.extend(current)
            costs.append({'model': name, 'seed': seed, **cost})
            if args.official:
                rows.extend(classical_reference(name, seed, folder, data))
    df = pd.DataFrame(rows)
    prefix = 'test' if args.official else 'validation'
    df.to_csv(folder / f'{prefix}_results.csv', index=False)
    pd.DataFrame(costs).to_csv(folder / 'cost.csv', index=False)
    print('\nMEDII (AUPRC / Spearman):')
    print(df.groupby(['model','noise'])[['AUPRC','spearman']].agg(['mean','std']).round(4))
    print('\nSalvat:', folder)
    if not args.official:
        print('IMPORTANT: validation subesantionat / ponderat NU este AUPRC oficial.')


if __name__ == '__main__':
    main()