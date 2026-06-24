#!/usr/bin/env python3
"""
Batch 10 Stream β — LeNet CIFAR-10 (fc Stiefel/MF, conv free)

Architecture: conv1(3→6) → pool → conv2(6→16) → pool → fc1(400→120) → fc2(120→84) → fc3(84→10)
- Conv layers: FREE (standard nn.Conv2d)
- FC layers: Stiefel-parameterized (StiefelLinear)
- Goal: test whether G3 confirm extends to CNN family

Outputs: method_1/lenet_cifar10/{stage_b_results.json, h1_pressure_persistence.json,
                                   g4_spectral_trace.json, REPORT.md}
"""
import sys, os, json, time, math, warnings, atexit, signal, traceback
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

BASE = Path("./experiments")
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from manifoldflow.spd_ops import (sym, fp32_eigh, symlogm,
                                   affine_invariant_step, spectral_clip, matrix_sqrt)
from manifoldflow.retraction import qr_retract, procrustes_align
from manifoldflow.tangent import decompose_tangent_normal, project_tangent
from manifoldflow.manifoldflow_optimizer import _stiefel_sgd_step, ManifoldFlowConfig

# ─── Constants ────────────────────────────────────────────────────────────
METHODS = ['fs-sgd', 'fs-adam', 'mf-sgd', 'mf-adam']
SEEDS   = [42, 123, 2024]
N_EPOCHS_A = 15    # Stage A (tuning)
N_EPOCHS_B = 35    # Stage B (full run)
BATCH_SZ = 128
LR_GRID  = [1e-3, 3e-3, 1e-2]
RHO = 1e-2; LS = 1e-3; KG = 10  # MF hyper-params

# ─── Serializer ────────────────────────────────────────────────────────────
def js(obj):
    if isinstance(obj, dict):           return {k: js(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [js(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, torch.Tensor):   return float(obj.item()) if obj.numel()==1 else obj.tolist()
    if obj is None:                     return None
    return obj

# ─── QR init ─────────────────────────────────────────────────────────────
def _qr_init(n, r, seed=None):
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    A = torch.randn(n, r, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.float()

# ─── StiefelLinear ────────────────────────────────────────────────────────
class StiefelLinear(nn.Module):
    def __init__(self, in_dim, out_dim, seed=None, mode='fs'):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim; self.mode = mode
        if out_dim <= in_dim:
            n, r = in_dim, out_dim; self.transpose = True
        else:
            n, r = out_dim, in_dim; self.transpose = False
        self.n, self.r = n, r
        self.Q    = nn.Parameter(_qr_init(n, r, seed))
        self._sqrtS_cache = torch.eye(r)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        if self.mode == 'fs':
            W_base = self.Q
        else:
            sqrtS = self._sqrtS_cache.to(self.Q.device, self.Q.dtype)
            W_base = self.Q @ sqrtS
        W = W_base.T if self.transpose else W_base
        return F.linear(x, W, self.bias)

    def update_sqrtS_cache(self, S):
        with torch.no_grad():
            self._sqrtS_cache = matrix_sqrt(sym(S)).detach().cpu()


# ─── LeNet with Stiefel FC layers ─────────────────────────────────────────
class LeNetStiefel(nn.Module):
    """LeNet-5 adapted for CIFAR-10 (3-channel, 32×32).
    Conv layers: free.  FC layers: Stiefel-parameterized.
    """
    def __init__(self, mode='fs', seed=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)    # → 6×28×28
        self.conv2 = nn.Conv2d(6, 16, 5)   # → 16×10×10
        self.pool  = nn.MaxPool2d(2, 2)
        # FC layers with Stiefel
        self.fc1 = StiefelLinear(400, 120, seed=seed*10+1, mode=mode)
        self.fc2 = StiefelLinear(120, 84,  seed=seed*10+2, mode=mode)
        self.fc3 = StiefelLinear(84,  10,  seed=seed*10+3, mode=mode)
        self.mode = mode

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → 6×14×14
        x = self.pool(F.relu(self.conv2(x)))  # → 16×5×5
        x = x.view(x.size(0), -1)             # → 400
        x = F.relu(self.fc1(x))               # → 120
        x = F.relu(self.fc2(x))               # → 84
        x = self.fc3(x)                        # → 10
        return x

    def stiefel_fc_layers(self):
        return [self.fc1, self.fc2, self.fc3]

    def stiefel_params(self):
        return [l.Q for l in self.stiefel_fc_layers()]

    def stiefel_layer_names(self):
        return ['fc1', 'fc2', 'fc3']

    def conv_and_bias_params(self):
        """All params EXCEPT Stiefel Q matrices."""
        stiefel_ids = {id(p) for p in self.stiefel_params()}
        return [p for p in self.parameters() if id(p) not in stiefel_ids]

    def set_mode(self, mode):
        self.mode = mode
        for l in self.stiefel_fc_layers():
            l.mode = mode

    def update_sqrtS_cache(self, layer_idx, S):
        self.stiefel_fc_layers()[layer_idx].update_sqrtS_cache(S)


# ─── Riemannian Adam state ─────────────────────────────────────────────────
class StiefelAdamState:
    def __init__(self):
        self.step = 0; self.m = None; self.v = None

def stiefel_adam_step(Q, G_tan, state_obj, lr, betas=(0.9, 0.999), eps=1e-8):
    beta1, beta2 = betas
    if state_obj.m is None:
        state_obj.m = torch.zeros_like(G_tan)
        state_obj.v = torch.zeros_like(G_tan)
    state_obj.step += 1; t = state_obj.step
    m_new = beta1 * state_obj.m + (1 - beta1) * G_tan
    v_new = beta2 * state_obj.v + (1 - beta2) * G_tan.pow(2)
    state_obj.m = m_new; state_obj.v = v_new
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    v_hat_safe = v_hat.clamp(min=0.0)
    D = project_tangent(Q, m_hat / (v_hat_safe.sqrt() + eps))
    Q_new = qr_retract(Q, -lr * D)
    state_obj.m = project_tangent(Q_new, state_obj.m)
    return Q_new


# ─── G3 paired test ───────────────────────────────────────────────────────
def g3_paired_test(mf_results, fs_results):
    mf_acc = [r['history'][-1]['test_acc'] for r in mf_results]
    fs_acc = [r['history'][-1]['test_acc'] for r in fs_results]
    diffs = [m - f for m, f in zip(mf_acc, fs_acc)]
    mean_d = float(np.mean(diffs))
    se_d   = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    t_stat, p_2s = sp_stats.ttest_rel(mf_acc, fs_acc) if len(diffs) > 1 else (0.0, 1.0)
    p_1s = float(p_2s / 2 if t_stat > 0 else 1.0)
    return {
        'mf_accs': mf_acc, 'fs_accs': fs_acc, 'diffs': diffs,
        'mean_diff': mean_d, 'se_diff': se_d,
        't_stat': float(t_stat), 'p_val_1sided': p_1s,
        'significant_1se': abs(mean_d) > se_d and mean_d > 0,
        'significant_p05': p_1s < 0.05,
    }


# ─── CIFAR-10 Data Loading (HuggingFace) ─────────────────────────────────
def load_cifar10_hf(device):
    """Load CIFAR-10 via HuggingFace datasets (no toronto.edu download)."""
    print("[Data] Loading CIFAR-10 via HuggingFace datasets...", flush=True)
    from datasets import load_dataset
    import numpy as np

    ds = load_dataset("uoft-cs/cifar10", trust_remote_code=True)
    print(f"[Data] Dataset loaded: {ds}", flush=True)

    def extract(split):
        imgs = np.array([np.array(x['img']) for x in ds[split]])  # (N,32,32,3)
        labels = np.array([x['label'] for x in ds[split]])
        # Normalize: CIFAR-10 mean/std per channel
        imgs = imgs.astype(np.float32) / 255.0
        mean = np.array([0.4914, 0.4822, 0.4465])
        std  = np.array([0.2470, 0.2435, 0.2616])
        imgs = (imgs - mean) / std  # (N,32,32,3)
        imgs = imgs.transpose(0, 3, 1, 2)  # (N,3,32,32)
        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    X_train, y_train = extract('train')
    X_test,  y_test  = extract('test')

    # Val split: last 5000 of train
    n_val = 5000
    X_val = X_train[-n_val:]; y_val = y_train[-n_val:]
    X_train = X_train[:-n_val]; y_train = y_train[:-n_val]

    print(f"[Data] Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}", flush=True)
    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te, batch_sz):
    tr_ld  = DataLoader(TensorDataset(X_tr, y_tr),   batch_sz, shuffle=True,  drop_last=True)
    val_ld = DataLoader(TensorDataset(X_val, y_val), batch_sz, shuffle=False)
    te_ld  = DataLoader(TensorDataset(X_te, y_te),  batch_sz, shuffle=False)
    return tr_ld, val_ld, te_ld


# ─── Evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    model.train()
    return correct / total


# ─── MF S updater ────────────────────────────────────────────────────────
class MFSState:
    """Per-layer SPD geometry state for ManifoldFlow."""
    def __init__(self, r):
        self.S = torch.eye(r)
        self.step = 0
        self.m = None; self.v = None  # Adam moments for Q step

def mf_update_S(S, P_t, rho_geo, lambda_S, lambda_min, lambda_max, mf_cfg, device):
    S = S.to(device)
    P_t = P_t.to(device)
    PPt = P_t @ P_t.t()
    grad_S = sym(PPt)
    S_new = affine_invariant_step(S, grad_S, rho_geo)
    S_new = spectral_clip(S_new, lambda_min, lambda_max)
    return S_new.cpu()


# ─── Train one method ─────────────────────────────────────────────────────
def train_one(method, lr, seed, n_epochs, tr_ld, val_ld, te_ld, device,
              rho_geo=1e-2, lambda_S=1e-3, K_geo=10):
    mode_str = method.split('-')[0]    # 'fs' or 'mf'
    base_opt = method.split('-')[1]    # 'sgd' or 'adam'

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LeNetStiefel(mode=mode_str, seed=seed).to(device)

    # Conv + bias params → regular optimizer
    conv_bias_params = model.conv_and_bias_params()
    if base_opt == 'sgd':
        reg_opt = torch.optim.SGD(conv_bias_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        reg_opt = torch.optim.Adam(conv_bias_params, lr=lr, weight_decay=5e-4)

    # Stiefel states
    stiefel_q_states = {}   # id(Q) → SGD state dict
    stiefel_adam_st  = {}   # id(Q) → StiefelAdamState
    mf_S_states      = {}   # id(Q) → MFSState
    mf_cfg = ManifoldFlowConfig(rho_geo=rho_geo, lambda_S=lambda_S, K_geo=K_geo)

    criterion = nn.CrossEntropyLoss()

    # Tracking
    history = []
    per_epoch_trace = []
    # H1: running cos(P_t, P_{t-1})
    _P_prev = {}
    h1_cos_accumulator = {nm: [] for nm in model.stiefel_layer_names()}  # epoch-level means
    # G4: per-epoch S stats
    _step_count = 0

    total_steps = n_epochs * len(tr_ld)
    warmup_steps = max(1, int(0.05 * total_steps))

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_cos_P = {nm: [] for nm in model.stiefel_layer_names()}

        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            _step_count += 1

            reg_opt.zero_grad()
            for p in model.stiefel_params():
                if p.grad is not None: p.grad.zero_()

            out  = model(X)
            loss = criterion(out, y)
            loss.backward()

            # ── Conv/bias step ──
            reg_opt.step()

            # ── Stiefel Q step ──
            fc_names  = model.stiefel_layer_names()
            fc_layers = model.stiefel_fc_layers()

            for nm, layer in zip(fc_names, fc_layers):
                Q = layer.Q
                if Q.grad is None: continue

                G_bar = Q.grad.float()
                split = decompose_tangent_normal(Q.detach().float(), G_bar)
                G_tan = split.G_tan; P_t = split.P
                qid   = id(Q)

                # H1 cos
                if qid in _P_prev:
                    P_prev = _P_prev[qid]
                    cos_val = float(((P_t * P_prev).sum() / (P_t.norm() * P_prev.norm() + 1e-12)).item())
                    epoch_cos_P[nm].append(cos_val)
                _P_prev[qid] = P_t.detach().clone()

                # MF geometry update
                if mode_str == 'mf' and _step_count >= warmup_steps and _step_count % K_geo == 0:
                    if qid not in mf_S_states:
                        mf_S_states[qid] = MFSState(layer.r)
                    S_state = mf_S_states[qid]
                    S_state.S = mf_update_S(
                        S_state.S, P_t, rho_geo, lambda_S,
                        mf_cfg.lambda_min, mf_cfg.lambda_max, mf_cfg, device)
                    layer.update_sqrtS_cache(S_state.S)

                # MF tangent modification: G_tan_mf = S⁻¹ G_tan
                if mode_str == 'mf' and qid in mf_S_states:
                    S = mf_S_states[qid].S.to(device, G_tan.dtype)
                    try:
                        # G_tan_mf = S_inv @ G_tan (using solve for numerical stability)
                        # S is small (r×r), direct inv is OK
                        S_inv = torch.linalg.inv(S)
                        # G_tan is (n×r), S is (r×r): G_tan_mf = G_tan @ S_inv.T
                        G_tan_eff = G_tan @ S_inv.t()
                        G_tan_eff = project_tangent(Q.detach().float(), G_tan_eff)
                    except Exception:
                        G_tan_eff = G_tan
                else:
                    G_tan_eff = G_tan

                # Q update
                if base_opt == 'sgd':
                    state = stiefel_q_states.setdefault(qid, {'step': 0})
                    Q_new = _stiefel_sgd_step(Q.detach().float(), G_tan_eff, state, lr, momentum=0.9)
                else:
                    astate = stiefel_adam_st.setdefault(qid, StiefelAdamState())
                    Q_new = stiefel_adam_step(Q.detach().float(), G_tan_eff, astate, lr)
                Q.data.copy_(Q_new.to(Q.dtype))

        # ── End of epoch: record stats ──
        val_acc  = evaluate(model, val_ld, device)
        test_acc = evaluate(model, te_ld, device)

        # H1: mean cos_P per layer this epoch
        cos_P_epoch = {}
        for nm in fc_names:
            vals = epoch_cos_P[nm]
            if vals:
                mean_cos = float(np.mean(vals))
                cos_P_epoch[nm] = mean_cos
                h1_cos_accumulator[nm].append(mean_cos)

        # G4: S stats per layer
        S_stats_epoch = {}
        if mode_str == 'mf':
            for nm, layer in zip(fc_names, fc_layers):
                qid = id(layer.Q)
                if qid in mf_S_states:
                    S = mf_S_states[qid].S
                    try:
                        eigvals = torch.linalg.eigvalsh(S.float()).clamp(min=1e-12)
                        lam_max = float(eigvals.max().item())
                        lam_min = float(eigvals.min().item())
                        ratio   = lam_max / (lam_min + 1e-12)
                    except Exception:
                        lam_max = lam_min = ratio = 0.0
                    S_stats_epoch[nm] = {
                        'lambda_max': lam_max,
                        'lambda_min': lam_min,
                        'lambda_ratio': ratio,
                    }

        gate_epoch = {}
        if mode_str == 'mf':
            # approximate gate a_t (ratio of normal to tangent norm)
            for nm, layer in zip(fc_names, fc_layers):
                qid = id(layer.Q)
                if layer.Q.grad is not None:
                    g = layer.Q.grad.float()
                    sp = decompose_tangent_normal(layer.Q.detach().float(), g)
                    g_tan_n = float(sp.G_tan.norm().item())
                    g_nor_n = float(sp.G_nor.norm().item()) if hasattr(sp, 'G_nor') else 0.0
                    gate_epoch[nm] = g_nor_n / (g_tan_n + 1e-12)

        rec = {
            'epoch': epoch, 'val_acc': val_acc, 'test_acc': test_acc,
            'cos_P': cos_P_epoch, 'S_stats': S_stats_epoch, 'gate': gate_epoch,
        }
        per_epoch_trace.append(rec)
        history.append({'epoch': epoch, 'val_acc': val_acc, 'test_acc': test_acc})

        if epoch % 5 == 0 or epoch == n_epochs:
            print(f"    [{method} seed={seed}] ep={epoch}/{n_epochs} "
                  f"val={val_acc:.4f} test={test_acc:.4f} "
                  f"cos_P={[f'{v:.3f}' for v in cos_P_epoch.values()]}", flush=True)

    # H1 stats: mean cos over first 1/3 of training
    cutoff = n_epochs // 3
    h1_stats = {}
    for nm in fc_names:
        early_vals = h1_cos_accumulator[nm][:cutoff]
        if early_vals:
            h1_stats[nm] = {
                'mean': float(np.mean(early_vals)),
                'std': float(np.std(early_vals)),
                'n': len(early_vals),
                'all_means': early_vals,
            }
        else:
            h1_stats[nm] = {'mean': None, 'std': None, 'n': 0, 'all_means': []}

    # Free GPU memory
    model.cpu()
    del model, reg_opt, stiefel_q_states, stiefel_adam_st, mf_S_states, _P_prev
    torch.cuda.empty_cache()

    return {
        'method': method, 'seed': seed, 'lr': lr,
        'history': history,
        'per_epoch_trace': per_epoch_trace,
        'h1_stats': h1_stats,
    }


# ─── Stage A: LR grid search ──────────────────────────────────────────────
def stage_a(tr_ld, val_ld, te_ld, device):
    print("\n=== Stage A: LR grid search ===", flush=True)
    best_lrs = {}
    seed_a = SEEDS[0]
    for method in METHODS:
        best_val = -1.0; best_lr = LR_GRID[0]
        for lr in LR_GRID:
            print(f"  [Stage A] {method} lr={lr} seed={seed_a}", flush=True)
            result = train_one(method, lr, seed_a, N_EPOCHS_A,
                               tr_ld, val_ld, te_ld, device,
                               rho_geo=RHO, lambda_S=LS, K_geo=KG)
            val_acc = result['history'][-1]['val_acc']
            print(f"    → val={val_acc:.4f}", flush=True)
            if val_acc > best_val:
                best_val = val_acc; best_lr = lr
        best_lrs[method] = best_lr
        print(f"  [Stage A] BEST {method}: lr={best_lr} val={best_val:.4f}", flush=True)
    return best_lrs


# ─── Main run ─────────────────────────────────────────────────────────────
def run(out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_cifar10_hf(device)
    tr_ld, val_ld, te_ld = make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te, BATCH_SZ)

    print(f"\nTrain batches={len(tr_ld)}, Val batches={len(val_ld)}, Test batches={len(te_ld)}", flush=True)
    print(f"GPU: {device}   Methods: {METHODS}   Seeds: {SEEDS}   Epochs-B: {N_EPOCHS_B}", flush=True)

    # ─── Stage A ──────────────────────────────────────────────────────────
    best_lrs = stage_a(tr_ld, val_ld, te_ld, device)
    print(f"\n[Stage A done] best_lrs={best_lrs}", flush=True)
    with open(out_dir / 'best_lrs.json', 'w') as f:
        json.dump(best_lrs, f, indent=2)

    # ─── Results accumulators ──────────────────────────────────────────────
    all_results = {m: [] for m in METHODS}
    stage_b = {
        'task': 'lenet_cifar10', 'status': 'running',
        'methods': {m: [] for m in METHODS},
    }
    h1_persist = {'task': 'lenet_cifar10', 'methods': {}}
    g4_trace   = {'task': 'lenet_cifar10', 'methods': {}}

    sb_file = out_dir / 'stage_b_results.json'
    h1_file = out_dir / 'h1_pressure_persistence.json'
    g4_file = out_dir / 'g4_spectral_trace.json'

    def flush_all():
        try:
            for path, obj in [(sb_file, stage_b), (h1_file, h1_persist), (g4_file, g4_trace)]:
                with open(path, 'w') as f: json.dump(js(obj), f, indent=2)
        except Exception as e:
            print(f"[flush_all error] {e}", flush=True)

    atexit.register(flush_all)

    # ─── Stage B: 3 seeds × 4 methods ────────────────────────────────────
    print("\n=== Stage B: 3 seeds × 4 methods ===", flush=True)

    for method in METHODS:
        lr = best_lrs[method]
        print(f"\n--- Method: {method}  lr={lr} ---", flush=True)
        h1_persist['methods'][method] = {}
        g4_trace['methods'][method] = []

        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            result = train_one(method, lr, seed, N_EPOCHS_B,
                               tr_ld, val_ld, te_ld, device,
                               rho_geo=RHO, lambda_S=LS, K_geo=KG)
            all_results[method].append(result)

            # H1 accumulate
            for nm, h1s in result['h1_stats'].items():
                if nm not in h1_persist['methods'][method]:
                    h1_persist['methods'][method][nm] = []
                h1_persist['methods'][method][nm].append({'seed': seed, **h1s})

            # G4 trace (MF only)
            mode_str = method.split('-')[0]
            if mode_str == 'mf':
                g4_trace['methods'][method].append({
                    'seed': seed,
                    'per_epoch': [
                        {
                            'epoch': rec['epoch'],
                            'S_stats': rec.get('S_stats', {}),
                            'cos_P':   rec.get('cos_P', {}),
                            'gate':    rec.get('gate', {}),
                        }
                        for rec in result['per_epoch_trace']
                    ]
                })

            stage_b['methods'][method].append({
                'seed': seed, 'lr': lr,
                'final_test_acc': result['history'][-1]['test_acc'],
                'final_val_acc':  result['history'][-1]['val_acc'],
                'history_summary': [
                    {'epoch': h['epoch'], 'test_acc': h['test_acc']}
                    for h in result['history'] if h['epoch'] % 5 == 0 or h['epoch'] == N_EPOCHS_B
                ],
            })
            flush_all()

    # ─── G3 Analysis ──────────────────────────────────────────────────────
    print("\n=== G3 Analysis ===", flush=True)
    g3_results = {}
    for base_opt in ['sgd', 'adam']:
        mf_k, fs_k = f'mf-{base_opt}', f'fs-{base_opt}'
        if all_results[mf_k] and all_results[fs_k]:
            g3 = g3_paired_test(all_results[mf_k], all_results[fs_k])
            g3_results[base_opt] = g3
            print(f"  {mf_k} vs {fs_k}: diff={g3['mean_diff']:+.4f} SE={g3['se_diff']:.4f} "
                  f"t={g3['t_stat']:.2f} p={g3['p_val_1sided']:.3f} "
                  f"1SE={'YES' if g3['significant_1se'] else 'no'}", flush=True)

    # ─── H1 Summary ───────────────────────────────────────────────────────
    h1_summary = {}
    for method in METHODS:
        h1_summary[method] = {}
        for nm, seed_list in h1_persist['methods'].get(method, {}).items():
            means = [s['mean'] for s in seed_list if s.get('mean') is not None]
            if means:
                t_stat, p_2s = sp_stats.ttest_1samp(means, 0.0) if len(means) > 1 else (0.0, 1.0)
                p_1s = float(p_2s / 2 if t_stat > 0 else 1.0)
                h1_summary[method][nm] = {
                    'mean_across_seeds': float(np.mean(means)),
                    'std_across_seeds':  float(np.std(means)),
                    't_stat': float(t_stat), 'p_val_1sided': p_1s,
                    'per_seed': seed_list,
                }

    all_h1_mf_means = []
    for method in ['mf-sgd', 'mf-adam']:
        if method in h1_summary:
            for nm, stats in h1_summary[method].items():
                m = stats.get('mean_across_seeds')
                if m is not None: all_h1_mf_means.append(m)
    h1_max = max(all_h1_mf_means) if all_h1_mf_means else 0.0

    # ─── G4 Summary ───────────────────────────────────────────────────────
    g4_summary = {}
    for method in ['mf-sgd', 'mf-adam']:
        if method not in g4_trace['methods']: continue
        final_stats = {}
        for seed_rec in g4_trace['methods'][method]:
            if not seed_rec['per_epoch']: continue
            last_ep = seed_rec['per_epoch'][-1]
            for nm, ss in last_ep.get('S_stats', {}).items():
                if nm not in final_stats:
                    final_stats[nm] = {'lambda_ratio': [], 'lambda_max': [], 'lambda_min': []}
                final_stats[nm]['lambda_ratio'].append(ss.get('lambda_ratio', 0.0))
                final_stats[nm]['lambda_max'].append(ss.get('lambda_max', 0.0))
                final_stats[nm]['lambda_min'].append(ss.get('lambda_min', 0.0))
        g4_summary[method] = {
            nm: {
                'lambda_ratio_mean': float(np.mean(v['lambda_ratio'])),
                'lambda_max_mean':   float(np.mean(v['lambda_max'])),
                'lambda_min_mean':   float(np.mean(v['lambda_min'])),
            }
            for nm, v in final_stats.items()
        }

    g4_max_ratio = 0.0
    for method_stats in g4_summary.values():
        for nm, stats in method_stats.items():
            g4_max_ratio = max(g4_max_ratio, stats.get('lambda_ratio_mean', 0.0))

    # ─── Verdict ──────────────────────────────────────────────────────────
    g3_sgd  = g3_results.get('sgd',  {}).get('significant_1se', False)
    g3_adam = g3_results.get('adam', {}).get('significant_1se', False)
    h1_confirm = h1_max > 0.2
    g3_confirm = g3_sgd or g3_adam
    g4_confirm = g4_max_ratio > 1.5

    verdict = {
        'h1_max_cos': float(h1_max), 'h1_confirm': h1_confirm,
        'g3_confirm_sgd': g3_sgd, 'g3_confirm_adam': g3_adam,
        'g3_confirm': g3_confirm, 'g4_max_ratio': float(g4_max_ratio),
        'g4_confirm': g4_confirm,
        'double_confirm': h1_confirm and g3_confirm,
        'triple_confirm': h1_confirm and g3_confirm and g4_confirm,
    }

    print(f"\n=== VERDICT: LeNet CIFAR-10 ===", flush=True)
    print(f"  H1 max cos(P)={h1_max:.4f} ({'CONFIRM' if h1_confirm else 'fail'} thr=0.2)", flush=True)
    print(f"  G3 SGD={'CONFIRM' if g3_sgd else 'fail'}  Adam={'CONFIRM' if g3_adam else 'fail'}", flush=True)
    print(f"  G4 max λ_ratio={g4_max_ratio:.3f} ({'CONFIRM' if g4_confirm else 'fail'} thr=1.5)", flush=True)
    print(f"  Double (H1+G3): {'*** YES ***' if verdict['double_confirm'] else 'no'}", flush=True)
    print(f"  Triple (H1+G3+G4): {'*** YES ***' if verdict['triple_confirm'] else 'no'}", flush=True)

    stage_b['status'] = 'done'
    stage_b['g3'] = g3_results
    stage_b['h1_summary'] = h1_summary
    stage_b['g4_summary'] = g4_summary
    stage_b['verdict'] = verdict
    h1_persist['h1_summary'] = h1_summary
    g4_trace['g4_summary'] = g4_summary
    g4_trace['verdict'] = {'g4_max_ratio': float(g4_max_ratio), 'g4_confirm': g4_confirm}
    flush_all()

    # ─── REPORT.md ────────────────────────────────────────────────────────
    lines = [
        "# LeNet CIFAR-10 — Batch 10 Stream β Report\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}\n\n",
        "## Architecture\n",
        "- **Conv layers**: Free (standard nn.Conv2d)\n",
        "- **FC layers**: Stiefel-parameterized (fc1: 400→120, fc2: 120→84, fc3: 84→10)\n",
        "- **Goal**: Test G3 confirm in CNN architecture with fc-only Stiefel\n\n",
        "## Setup\n",
        f"- {METHODS}\n",
        f"- Seeds: {SEEDS}  Epochs: {N_EPOCHS_B}  Batch: {BATCH_SZ}\n",
        f"- MF hyper: rho_geo={RHO}, lambda_S={LS}, K_geo={KG}\n\n",
        "## Best LRs (Stage A)\n",
    ]
    for m, lr_v in best_lrs.items():
        lines.append(f"- {m}: {lr_v}\n")

    lines += ["\n## Final Test Accuracy\n\n| Method | S1 | S2 | S3 | Mean |\n|---|---|---|---|---|\n"]
    for method in METHODS:
        if not all_results[method]: continue
        accs = [r['history'][-1]['test_acc'] for r in all_results[method]]
        lines.append(f"| {method} | " + " | ".join(f"{a:.4f}" for a in accs) + f" | {np.mean(accs):.4f} |\n")

    lines += ["\n## G3: MF vs FS paired 1-sided t-test\n\n"]
    for base_opt in ['sgd', 'adam']:
        if base_opt not in g3_results: continue
        g = g3_results[base_opt]
        lines += [
            f"### {base_opt.upper()}\n",
            f"- Δ mean={g['mean_diff']:+.4f} ± {g['se_diff']:.4f} SE  t={g['t_stat']:.2f}  p={g['p_val_1sided']:.3f}\n",
            f"- **>1 SE: {'✅ YES' if g['significant_1se'] else '❌ no'}**  "
            f"p<0.05: {'✅ YES' if g['significant_p05'] else '❌ no'}\n\n",
        ]

    lines += ["\n## H1: cos(P_t, P_{t-1}) per FC layer (first 1/3 of training)\n\n"]
    for method in METHODS:
        lines.append(f"### {method}\n")
        for nm, stats in h1_summary.get(method, {}).items():
            m = stats.get('mean_across_seeds')
            p = stats.get('p_val_1sided')
            lines.append(f"- {nm}: mean={m:.4f}" + (f"  p={p:.3f}" if p else "") + "\n")
        lines.append("\n")

    lines += ["\n## G4: λ_max/λ_min(S) at end of training (FC layers)\n\n"]
    for method, method_stats in g4_summary.items():
        lines.append(f"### {method}\n")
        for nm, s in method_stats.items():
            lines.append(f"- {nm}: λ_ratio={s['lambda_ratio_mean']:.3f}  "
                         f"λ_max={s['lambda_max_mean']:.3f}  λ_min={s['lambda_min_mean']:.3f}\n")
        lines.append("\n")

    lines += [
        "\n## Verdict\n\n",
        f"- H1 max cos(P) = **{h1_max:.4f}** ({'✅ CONFIRM' if h1_confirm else '❌ fail'} thr=0.2)\n",
        f"- G3 SGD: {'✅ CONFIRM' if g3_sgd else '❌ fail'}\n",
        f"- G3 Adam: {'✅ CONFIRM' if g3_adam else '❌ fail'}\n",
        f"- G4 max λ_ratio: **{g4_max_ratio:.3f}** ({'✅ CONFIRM' if g4_confirm else '❌ fail'} thr=1.5)\n",
        f"- **H1+G3 Double Confirm: {'✅ YES' if verdict['double_confirm'] else '❌ NO'}**\n",
        f"- **H1+G3+G4 Triple Confirm: {'✅ YES' if verdict['triple_confirm'] else '❌ NO'}**\n",
    ]

    with open(out_dir / 'REPORT.md', 'w') as f: f.writelines(lines)
    print(f"\n[DONE] Outputs saved to {out_dir}", flush=True)
    return verdict


# ─── Entry ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    def _sig(sig, frame):
        print(f"\n[SIGNAL {sig}] flushing and exit...", flush=True); sys.exit(0)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[Batch10-β LeNet CIFAR-10] GPU=cuda:2 device={device}", flush=True)

    out_dir = BASE / "method_1" / "lenet_cifar10"

    try:
        verdict = run(out_dir, device)
        print(f"\n[DONE] double_confirm={verdict['double_confirm']} triple_confirm={verdict['triple_confirm']}", flush=True)
        print(f"[DONE] H1_max={verdict['h1_max_cos']:.4f} G3={verdict['g3_confirm']} G4_ratio={verdict['g4_max_ratio']:.3f}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
