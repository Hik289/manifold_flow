#!/usr/bin/env python3
"""
E2 — New baselines on Adult + Covertype (Adam lr=0.001, 3 seeds: 42, 123, 7)
B1: W = Q @ diag(s),  s > 0 (exp parametrization)
B2: Unconstrained dense nn.Linear  (matched param count, no Stiefel)
B3: Spectral normalization (Miyato 2018) on each Linear

Outputs per baseline per dataset:
  experiments/comparison/baselines_v2/{B1_qdiag,B2_dense,B3_specnorm}/{adult,covertype}/results.json
"""
import sys, os, json, time, math, warnings, atexit, signal, traceback
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

BASE = Path("./experiments")
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from manifoldflow.retraction import qr_retract
from manifoldflow.tangent import decompose_tangent_normal, project_tangent

def js(obj):
    if isinstance(obj, dict):           return {k: js(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [js(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, torch.Tensor):   return float(obj.item())
    return obj

def _qr_init(n, r, seed=None):
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    A = torch.randn(n, r, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.float()

# ─── B1: QDiag layer ───────────────────────────────────────────────────────
class QDiagLinear(nn.Module):
    """W = Q @ diag(exp(log_s)), Q on Stiefel (fixed, not trained via manifold step), s>0."""
    def __init__(self, in_dim, out_dim, seed=None):
        super().__init__()
        if out_dim <= in_dim:
            n, r = in_dim, out_dim; self.transpose = True
        else:
            n, r = out_dim, in_dim; self.transpose = False
        self.n, self.r = n, r
        self.Q     = nn.Parameter(_qr_init(n, r, seed))
        self.log_s = nn.Parameter(torch.zeros(r))  # exp(log_s) = 1 init
        self.bias  = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        s     = torch.exp(self.log_s)          # positive scaling
        W_base = self.Q * s.unsqueeze(0)       # broadcast: (n,r) * (r,) = (n,r)
        W = W_base.T if self.transpose else W_base
        return F.linear(x, W, self.bias)

class QDiagMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        super().__init__()
        dims = [in_dim] + [hidden_dim]*4 + [out_dim]
        self.layers = nn.ModuleList([
            QDiagLinear(dims[i], dims[i+1], seed=seed*1000+i)
            for i in range(len(dims)-1)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: x = F.relu(x)
        return x

# ─── B2: Dense MLP (matched params) ────────────────────────────────────────
class DenseMLP(nn.Module):
    """Standard nn.Linear, same architecture (no manifold constraint)."""
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        dims = [in_dim] + [hidden_dim]*4 + [out_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: x = F.relu(x)
        return x

# ─── B3: SpectralNorm MLP ───────────────────────────────────────────────────
class SpectralNormMLP(nn.Module):
    """nn.Linear + spectral_norm on each weight, same architecture."""
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        dims = [in_dim] + [hidden_dim]*4 + [out_dim]
        self.layers = nn.ModuleList([
            nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1]))
            for i in range(len(dims)-1)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: x = F.relu(x)
        return x


# ─── Data loaders ──────────────────────────────────────────────────────────
def load_adult(seed=0):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    print("  Fetching Adult...", flush=True)
    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    X = data.data.copy(); y = data.target
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.values.astype(np.float32)
    y_enc = LabelEncoder().fit_transform(y.astype(str)); y_arr = y_enc.astype(np.int64)
    X_tv, X_test, y_tv, y_test = train_test_split(X, y_arr, test_size=0.15, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15, random_state=seed)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train); X_val = sc.transform(X_val); X_test = sc.transform(X_test)
    return (torch.from_numpy(X_train), torch.from_numpy(y_train),
            torch.from_numpy(X_val),   torch.from_numpy(y_val),
            torch.from_numpy(X_test),  torch.from_numpy(y_test), X_train.shape[1], 2)

def load_covertype(seed=0, n_subset=100_000):
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    print("  Fetching Covertype...", flush=True)
    data = fetch_covtype()
    X, y = data.data.astype(np.float32), (data.target - 1).astype(np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_subset, replace=False)
    X, y = X[idx], y[idx]
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15, random_state=seed)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train); X_val = sc.transform(X_val); X_test = sc.transform(X_test)
    return (torch.from_numpy(X_train), torch.from_numpy(y_train),
            torch.from_numpy(X_val),   torch.from_numpy(y_val),
            torch.from_numpy(X_test),  torch.from_numpy(y_test), X_train.shape[1], 7)

def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=256):
    def _ld(X, y, sh): return DataLoader(TensorDataset(X.float(), y.long()),
                                          batch_size=batch_size, shuffle=sh, num_workers=0)
    return _ld(X_train, y_train, True), _ld(X_val, y_val, False), _ld(X_test, y_test, False)

def eval_acc(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
    return correct / total if total > 0 else 0.0


def train_model(model, tr_ld, val_ld, test_ld, device, n_epochs, lr=0.001, wd=1e-4):
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    history = []; t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train(); correct = total = 0
        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
            with torch.no_grad():
                pred = model(X).argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
        tr_acc  = correct / total
        val_acc = eval_acc(model, val_ld, device)
        test_acc= eval_acc(model, test_ld, device)
        elapsed = time.time() - t0
        history.append({'epoch': ep, 'train_acc': tr_acc, 'val_acc': val_acc,
                        'test_acc': test_acc, 'elapsed_s': elapsed})
        if ep % 10 == 0 or ep == n_epochs:
            print(f"    ep{ep:3d} train={tr_acc:.4f} val={val_acc:.4f} test={test_acc:.4f} {elapsed:.0f}s", flush=True)
    return history


SEEDS = [42, 123, 7]
LR    = 0.001

def run_baseline(baseline_name, task_name, model_fn, device, out_base):
    out_dir = Path(out_base) / baseline_name / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'results.json'

    if task_name == 'adult':
        n_epochs, hidden, batch_sz = 100, 128, 256; data_fn = load_adult
    elif task_name == 'covertype':
        n_epochs, hidden, batch_sz = 80, 128, 512; data_fn = load_covertype

    X_train, y_train, X_val, y_val, X_test, y_test, in_dim, out_dim = data_fn(seed=42)
    tr_ld, val_ld, test_ld = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_sz)

    results = {'baseline': baseline_name, 'task': task_name, 'lr': LR, 'seeds': SEEDS, 'runs': []}

    def flush():
        try:
            with open(out_file, 'w') as f: json.dump(js(results), f, indent=2)
        except Exception as e:
            print(f"[flush error] {e}", flush=True)
    atexit.register(flush)

    print(f"\n{'='*60}", flush=True)
    print(f"{baseline_name} | {task_name}  in={in_dim} out={out_dim} hidden={hidden}", flush=True)
    print(f"n_epochs={n_epochs} batch={batch_sz} lr={LR}", flush=True)

    for seed in SEEDS:
        print(f"\n  seed={seed}", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = model_fn(in_dim, hidden, out_dim, seed).to(device)
        history = train_model(model, tr_ld, val_ld, test_ld, device, n_epochs, lr=LR)
        results['runs'].append({
            'seed': seed,
            'final_test_acc': history[-1]['test_acc'],
            'final_val_acc':  history[-1]['val_acc'],
            'history_summary': [{'epoch': h['epoch'], 'test_acc': h['test_acc']}
                                 for h in history if h['epoch'] % 10 == 0 or h['epoch'] == n_epochs],
        })
        flush()
        del model; torch.cuda.empty_cache()

    accs = [r['final_test_acc'] for r in results['runs']]
    results['summary'] = {
        'mean': float(np.mean(accs)), 'std': float(np.std(accs, ddof=1)), 'n': len(accs)
    }
    flush()

    print(f"\n=== {baseline_name} | {task_name} ===", flush=True)
    for r in results['runs']:
        print(f"  seed={r['seed']}: {r['final_test_acc']:.4f}", flush=True)
    print(f"  MEAN={results['summary']['mean']:.4f} ± {results['summary']['std']:.4f}", flush=True)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['adult', 'covertype'], required=True)
    parser.add_argument('--baseline', choices=['B1_qdiag', 'B2_dense', 'B3_specnorm', 'all'], default='all')
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    def _sig(sig, frame):
        print(f"\n[SIGNAL {sig}] flushing...", flush=True); sys.exit(0)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[E2] Task={args.task} Baseline={args.baseline} GPU={args.cuda} device={device}", flush=True)

    out_base = BASE / "comparison" / "baselines_v2"

    baselines = {
        'B1_qdiag':    lambda in_d, hid, out_d, seed: QDiagMLP(in_d, hid, out_d, seed),
        'B2_dense':    lambda in_d, hid, out_d, seed: DenseMLP(in_d, hid, out_d, seed),
        'B3_specnorm': lambda in_d, hid, out_d, seed: SpectralNormMLP(in_d, hid, out_d, seed),
    }
    to_run = list(baselines.keys()) if args.baseline == 'all' else [args.baseline]

    for bname in to_run:
        print(f"\n\n{'#'*70}", flush=True)
        print(f"# Baseline: {bname}  Task: {args.task}", flush=True)
        print(f"{'#'*70}", flush=True)
        try:
            run_baseline(bname, args.task, baselines[bname], device, out_base)
        except Exception as e:
            print(f"\n[ERROR] {bname}: {e}", flush=True)
            traceback.print_exc()

    print(f"\n[DONE E2] task={args.task}", flush=True)
