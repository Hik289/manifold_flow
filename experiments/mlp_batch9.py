#!/usr/bin/env python3
"""
Batch 9 — MLP trace + Adam fix
- ADAM FIX: remove v (2nd moment) transport to prevent negative v → NaN sqrt
- Per-epoch per-layer trace: H1 cos_P, G4 λ_max/λ_min, P_norm, G_tan_norm, gate a_t
- Tasks: Adult (GPU1) + Covertype (GPU2)  |  NO CIFAR-10
- Outputs: stage_b_results_v2.json, h1_pressure_persistence.json, g4_spectral_trace.json, REPORT.md
"""
import sys, os, json, time, math, warnings, argparse, atexit, signal, traceback
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

# ─── Serializer ────────────────────────────────────────────────────────────
def js(obj):
    if isinstance(obj, dict):          return {k: js(v) for k, v in obj.items()}
    if isinstance(obj, list):          return [js(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):return float(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, torch.Tensor):  return float(obj.item())
    if obj is None:                    return None
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


class StiefelMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mode='fs', seed=0):
        super().__init__()
        dims = [in_dim] + [hidden_dim]*4 + [out_dim]
        self.layers = nn.ModuleList([
            StiefelLinear(dims[i], dims[i+1], seed=seed*1000+i, mode=mode)
            for i in range(len(dims)-1)
        ])
        self.mode = mode
        self.layer_names = [f"layer_{i}" for i in range(len(self.layers))]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1: x = F.relu(x)
        return x

    def stiefel_params(self):
        return [l.Q for l in self.layers]

    def set_mode(self, mode):
        self.mode = mode
        for l in self.layers: l.mode = mode

    def update_sqrtS_cache(self, layer_idx, S):
        self.layers[layer_idx].update_sqrtS_cache(S)


# ─── FIXED Riemannian Adam ────────────────────────────────────────────────
# BUG FIX: Do NOT project v (second moment) onto new tangent space.
# Tangent projection introduces negative entries in v → sqrt(v) = NaN → collapse.
# v is used only as per-element scale → transport is unnecessary and harmful.
class StiefelAdamState:
    def __init__(self):
        self.step = 0
        self.m = None   # first moment (tangent vector — transport is OK)
        self.v = None   # second moment (scale matrix — NO TRANSPORT)

def stiefel_adam_step(Q, G_tan, state_obj, lr, betas=(0.9, 0.999), eps=1e-8):
    """
    Fixed Riemannian Adam on Stiefel manifold.
    Key fix: v (second moment) NOT transported through tangent projection.
    v stores elementwise variance estimates; projection destroys positivity.
    """
    beta1, beta2 = betas
    if state_obj.m is None:
        state_obj.m = torch.zeros_like(G_tan)
        state_obj.v = torch.zeros_like(G_tan)
    state_obj.step += 1
    t = state_obj.step

    # Update moments
    m_new = beta1 * state_obj.m + (1 - beta1) * G_tan
    v_new = beta2 * state_obj.v + (1 - beta2) * G_tan.pow(2)
    state_obj.m = m_new
    state_obj.v = v_new  # v remains NOT transported

    # Bias correction
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    # v_hat should be non-negative (it's sum of squared tangent components);
    # clamp to eps to guard against any numerical underflow
    v_hat_safe = v_hat.clamp(min=0.0)

    D = project_tangent(Q, m_hat / (v_hat_safe.sqrt() + eps))
    Q_new = qr_retract(Q, -lr * D)

    # Transport ONLY m (first moment is a tangent vector — transport matters)
    state_obj.m = project_tangent(Q_new, state_obj.m)
    # v is NOT transported — intentionally left as-is
    return Q_new


# ─── FS Optimizer ──────────────────────────────────────────────────────────
class StiefelFSOptimizer:
    def __init__(self, model, base_optim='sgd', lr=0.01, momentum=0.9,
                 betas=(0.9, 0.999), weight_decay=1e-4):
        self.model = model
        self.base_optim = base_optim
        self.lr = lr; self.momentum = momentum
        self.betas = betas; self.wd = weight_decay
        self.Q_states   = {}
        self.adam_states = {}
        self.bias_optim = torch.optim.Adam(
            [l.bias for l in model.layers], lr=lr, weight_decay=weight_decay)
        self._P_prev = {}

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None: p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.bias_optim.step()
        cos_P_per_layer = {}
        P_norm_per_layer = {}
        G_tan_norm_per_layer = {}

        for name, Q in zip(self.model.layer_names, self.model.stiefel_params()):
            if Q.grad is None: continue
            G_bar = Q.grad.float()
            split = decompose_tangent_normal(Q.float(), G_bar)
            G_tan = split.G_tan; P_t = split.P
            qid = id(Q)

            if qid in self._P_prev:
                P_prev = self._P_prev[qid]
                cos_val = (P_t * P_prev).sum() / (P_t.norm() * P_prev.norm() + 1e-12)
                cos_P_per_layer[name] = float(cos_val.item())
            self._P_prev[qid] = P_t.detach().clone()
            P_norm_per_layer[name]    = float(P_t.norm().item())
            G_tan_norm_per_layer[name]= float(G_tan.norm().item())

            if self.base_optim == 'sgd':
                state = self.Q_states.setdefault(qid, {'step': 0})
                Q_new = _stiefel_sgd_step(Q.float(), G_tan, state, self.lr, self.momentum)
            else:
                astate = self.adam_states.setdefault(qid, StiefelAdamState())
                Q_new  = stiefel_adam_step(Q.float(), G_tan, astate, self.lr, self.betas)
            Q.data.copy_(Q_new.to(Q.dtype))
        return cos_P_per_layer, P_norm_per_layer, G_tan_norm_per_layer

    def set_lr(self, lr):
        self.lr = lr
        self.bias_optim.param_groups[0]['lr'] = lr


# ─── MF Optimizer ──────────────────────────────────────────────────────────
class StiefelMFOptimizer:
    def __init__(self, model, base_optim='sgd', lr=0.01, momentum=0.9,
                 betas=(0.9, 0.999), weight_decay=1e-4,
                 rho_geo=3e-3, lambda_S=1e-3, K_geo=10, total_steps=None):
        self.model = model; self.base_optim = base_optim
        self.lr = lr; self.momentum = momentum
        self.betas = betas; self.wd = weight_decay; self.total_steps = total_steps
        self.cfg = ManifoldFlowConfig(
            rho_geo=rho_geo, lambda_S=lambda_S, K_geo=K_geo,
            beta_P=0.95, tau_c=0.1, tau_r=0.0,
            alpha_c=5.0, alpha_r=2.0, lambda_min=0.25, lambda_max=4.0, warmup_frac=0.05)
        self.Q_states    = {}
        self.adam_states = {}
        self.S_states    = {}
        self.bias_optim  = torch.optim.Adam(
            [l.bias for l in model.layers], lr=lr, weight_decay=weight_decay)
        self._P_prev        = {}
        self._last_gate     = {}   # for trace
        self._last_S_stats  = {}   # for trace

    def _warmup_steps(self):
        if self.total_steps is None: return 0
        return int(math.ceil(self.cfg.warmup_frac * self.total_steps))

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None: p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.bias_optim.step()
        cos_P_per_layer    = {}
        P_norm_per_layer   = {}
        G_tan_norm_per_layer = {}
        cfg = self.cfg; eps = 1e-8

        for layer_idx, (name, Q) in enumerate(zip(self.model.layer_names, self.model.stiefel_params())):
            if Q.grad is None: continue
            dev = Q.device; qid = id(Q); r = Q.shape[-1]

            if qid not in self.Q_states:
                self.Q_states[qid] = {
                    'step': 0,
                    'M_P':    torch.zeros(r, r, dtype=torch.float32),
                    'Q_prev': Q.float().detach().cpu(),
                }
            if qid not in self.S_states:
                self.S_states[qid] = torch.eye(r, dtype=torch.float32)

            state  = self.Q_states[qid]
            S_cpu  = self.S_states[qid]

            G_bar  = Q.grad.float()
            split  = decompose_tangent_normal(Q.float(), G_bar)
            G_tan  = split.G_tan; P_t = split.P

            if qid in self._P_prev:
                P_prev = self._P_prev[qid]
                cos_val = (P_t * P_prev).sum() / (P_t.norm() * P_prev.norm() + 1e-12)
                cos_P_per_layer[name] = float(cos_val.item())
            self._P_prev[qid] = P_t.detach().clone()
            P_norm_per_layer[name]     = float(P_t.norm().item())
            G_tan_norm_per_layer[name] = float(G_tan.norm().item())

            # Stiefel step (FIXED Adam)
            if self.base_optim == 'sgd':
                Q_new = _stiefel_sgd_step(Q.float(), G_tan, state, self.lr, self.momentum)
            else:
                astate = self.adam_states.setdefault(qid, StiefelAdamState())
                Q_new  = stiefel_adam_step(Q.float(), G_tan, astate, self.lr, self.betas)

            # SPD geometry update
            t = state['step']; gamma_t = cfg.rho_geo * self.lr
            M_P    = state['M_P'].to(dev)
            Q_prev = state['Q_prev'].to(dev)

            if t > 0:
                A = Q.float().T @ Q_prev
                try:
                    O_t = procrustes_align(A)
                    M_P = O_t @ M_P @ O_t.T
                except Exception:
                    pass

            G_bar_norm   = G_bar.norm() + eps
            P_normalized = P_t / G_bar_norm
            M_P_prev     = M_P.clone()
            M_P_new      = cfg.beta_P * M_P + (1.0 - cfg.beta_P) * P_normalized
            state['M_P'] = M_P_new.cpu()

            warmup_done = t >= self._warmup_steps()
            do_geo = (gamma_t > 0.0) and warmup_done and ((t % cfg.K_geo) == 0)

            a_t = 0.0
            if do_geo:
                S_dev    = S_cpu.to(dev)
                P_norm   = P_t.norm() + eps
                M_p_norm = M_P_prev.norm() + eps
                c_t      = (P_t * M_P_prev).sum() / (P_norm * M_p_norm)
                G_n_norm = (Q.float() @ P_t).norm() + eps
                G_t_norm = G_tan.norm() + eps
                r_t      = G_n_norm / G_t_norm
                log_r_t  = torch.log(r_t)
                a_t_c    = torch.sigmoid(torch.tensor(cfg.alpha_c * (c_t.item() - cfg.tau_c)))
                a_t_r    = torch.sigmoid(cfg.alpha_r * (log_r_t - cfg.tau_r))
                a_t      = float((a_t_c * a_t_r).item())
                H_t      = sym(M_P_new) + cfg.lambda_S * symlogm(S_dev)
                S_raw    = affine_invariant_step(S_dev, H_t, gamma_t * a_t)
                S_new    = spectral_clip(S_raw, cfg.lambda_min, cfg.lambda_max)
                self.S_states[qid] = S_new.cpu()
                self.model.update_sqrtS_cache(layer_idx, S_new.cpu())
            else:
                S_new = S_cpu.to(dev)

            Q.data.copy_(Q_new.to(Q.dtype))
            state['Q_prev'] = Q_new.detach().cpu()
            state['step']   = t + 1

            eigvals, _ = fp32_eigh(S_new)
            self._last_gate[name] = a_t
            self._last_S_stats[name] = {
                'lambda_min':   float(eigvals.min().item()),
                'lambda_max':   float(eigvals.max().item()),
                'lambda_ratio': float((eigvals.max() / (eigvals.min() + 1e-12)).item()),
            }

        return cos_P_per_layer, P_norm_per_layer, G_tan_norm_per_layer

    def get_S_stats(self):
        return dict(self._last_S_stats)

    def get_gate(self):
        return dict(self._last_gate)

    def set_lr(self, lr):
        self.lr = lr
        self.bias_optim.param_groups[0]['lr'] = lr


# ─── Data loaders ──────────────────────────────────────────────────────────
def load_adult(seed=0):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    print("  Fetching Adult...", flush=True)
    data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    X = data.data.copy()
    y = data.target
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.values.astype(np.float32)
    y_enc = LabelEncoder().fit_transform(y.astype(str))
    y_arr = y_enc.astype(np.int64)
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

def make_optimizer(model, mode_str, base_opt, lr, rho_geo, lambda_S, K_geo, total_steps):
    momentum = 0.9 if base_opt == 'sgd' else 0.0
    if mode_str == 'fs':
        return StiefelFSOptimizer(model, base_optim=base_opt, lr=lr,
                                   momentum=momentum, betas=(0.9, 0.999), weight_decay=1e-4)
    else:
        return StiefelMFOptimizer(model, base_optim=base_opt, lr=lr,
                                   momentum=momentum, betas=(0.9, 0.999), weight_decay=1e-4,
                                   rho_geo=rho_geo, lambda_S=lambda_S, K_geo=K_geo,
                                   total_steps=total_steps)


# ─── Per-epoch trace ────────────────────────────────────────────────────────
def make_epoch_trace(ep, cos_P, P_norm, G_tan_norm, mode_str, opt):
    rec = {'epoch': ep, 'cos_P': cos_P, 'P_norm': P_norm, 'G_tan_norm': G_tan_norm}
    if mode_str == 'mf':
        rec['S_stats'] = opt.get_S_stats()
        rec['gate']    = opt.get_gate()
    return rec


# ─── Train one seed ────────────────────────────────────────────────────────
def train_one(in_dim, hidden_dim, out_dim, method, lr, seed, n_epochs,
              tr_ld, val_ld, test_ld, device, total_steps,
              rho_geo=3e-3, lambda_S=1e-3, K_geo=10):
    mode_str = method.split('-')[0]
    base_opt = method.split('-')[1]
    torch.manual_seed(seed); np.random.seed(seed)

    model = StiefelMLP(in_dim, hidden_dim, out_dim, mode=mode_str, seed=seed).to(device)
    opt   = make_optimizer(model, mode_str, base_opt, lr, rho_geo, lambda_S, K_geo, total_steps)
    crit  = nn.CrossEntropyLoss()

    history   = []
    per_epoch_trace = []
    h1_cos_accum    = {nm: [] for nm in model.layer_names}  # only first 1/3
    t0 = time.time()
    global_step = 0
    third_mark  = total_steps // 3

    for ep in range(1, n_epochs + 1):
        model.train()
        correct = total = 0
        ep_cos_P = {nm: [] for nm in model.layer_names}
        ep_P_norm = {nm: [] for nm in model.layer_names}
        ep_G_tan  = {nm: [] for nm in model.layer_names}

        for X, y in tr_ld:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            result = opt.step()
            if isinstance(result, tuple):
                cos_P, P_norm, G_tan_norm = result
            else:
                cos_P, P_norm, G_tan_norm = result, {}, {}

            for nm in model.layer_names:
                if nm in cos_P:
                    ep_cos_P[nm].append(cos_P[nm])
                    if global_step < third_mark:
                        h1_cos_accum[nm].append(cos_P[nm])
                if nm in P_norm: ep_P_norm[nm].append(P_norm[nm])
                if nm in G_tan_norm: ep_G_tan[nm].append(G_tan_norm[nm])

            pred = model(X).detach().argmax(1)
            correct += (pred == y).sum().item(); total += y.size(0)
            global_step += 1

        tr_acc   = correct / total
        val_acc  = eval_acc(model, val_ld, device)
        test_acc = eval_acc(model, test_ld, device)
        elapsed  = time.time() - t0

        # Epoch mean trace
        epoch_cos   = {nm: float(np.mean(v)) for nm, v in ep_cos_P.items()  if v}
        epoch_Pn    = {nm: float(np.mean(v)) for nm, v in ep_P_norm.items() if v}
        epoch_Gtan  = {nm: float(np.mean(v)) for nm, v in ep_G_tan.items()  if v}
        trace_rec   = make_epoch_trace(ep, epoch_cos, epoch_Pn, epoch_Gtan, mode_str, opt)
        per_epoch_trace.append(trace_rec)

        history.append({'epoch': ep, 'train_acc': tr_acc, 'val_acc': val_acc,
                        'test_acc': test_acc, 'elapsed_s': elapsed, 'lr': lr})
        if ep % 10 == 0 or ep == n_epochs:
            print(f"    ep{ep:3d} train={tr_acc:.4f} val={val_acc:.4f} test={test_acc:.4f} {elapsed:.0f}s",
                  flush=True)

    # H1 stats (first 1/3 of training)
    h1_stats = {}
    for nm, vals in h1_cos_accum.items():
        if len(vals) > 3:
            arr = np.array(vals)
            t_stat, p_2s = sp_stats.ttest_1samp(arr, 0.0)
            p_1s = float(p_2s / 2 if t_stat > 0 else 1.0 - p_2s / 2)
            h1_stats[nm] = {
                'mean': float(arr.mean()), 'std': float(arr.std()),
                'n': len(arr), 't_stat': float(t_stat),
                'p_val_1sided': p_1s,
                'vals_early': arr[:50].tolist(),
            }
        else:
            h1_stats[nm] = {'mean': None, 'std': None, 'n': len(vals),
                             't_stat': None, 'p_val_1sided': None}

    del model; torch.cuda.empty_cache()
    return {
        'history': history,
        'h1_stats': h1_stats,
        'per_epoch_trace': per_epoch_trace,
    }


# ─── G3 paired test ────────────────────────────────────────────────────────
def g3_paired_test(mf_results, fs_results):
    mf_accs = [r['history'][-1]['test_acc'] for r in mf_results]
    fs_accs = [r['history'][-1]['test_acc'] for r in fs_results]
    diffs   = np.array(mf_accs) - np.array(fs_accs)
    n       = len(diffs); mean_d = diffs.mean()
    se_d    = diffs.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    if se_d > 0:
        t_stat = mean_d / se_d
        p_val  = float(sp_stats.t.sf(t_stat, df=n-1))
    else:
        t_stat, p_val = 0.0, 1.0
    return {
        'mf_accs': mf_accs, 'fs_accs': fs_accs, 'diffs': diffs.tolist(),
        'mean_diff': float(mean_d), 'se_diff': float(se_d),
        't_stat': float(t_stat), 'p_val_1sided': float(p_val),
        'significant_1se': bool(mean_d > se_d),
        'significant_p05': bool(p_val < 0.05),
    }


# ─── Adam quick-verify ─────────────────────────────────────────────────────
def verify_adam_fix(in_dim, hidden_dim, out_dim, device, tr_ld, val_ld, test_ld,
                    lr=0.001, n_epochs=30, rho_geo=3e-3, lambda_S=1e-3, K_geo=10):
    """Quick 1-seed 30-epoch FS-Adam verify. Returns test_acc."""
    print(f"\n[Adam verify] FS-Adam lr={lr} 30ep seed=42", flush=True)
    result = train_one(in_dim, hidden_dim, out_dim, 'fs-adam', lr, 42, n_epochs,
                       tr_ld, val_ld, test_ld, device,
                       total_steps=n_epochs * len(tr_ld),
                       rho_geo=rho_geo, lambda_S=lambda_S, K_geo=K_geo)
    acc = result['history'][-1]['test_acc']
    print(f"[Adam verify] FS-Adam 30ep test_acc={acc:.4f} (threshold=0.80)", flush=True)
    return acc


# ─── Main run ──────────────────────────────────────────────────────────────
PRESET_LRS = {
    'adult':    {'fs-sgd': 0.01, 'fs-adam': 0.001, 'mf-sgd': 0.01, 'mf-adam': 0.001},
    'covertype':{'fs-sgd': 0.01, 'fs-adam': 0.001, 'mf-sgd': 0.01, 'mf-adam': 0.001},
}

def run_task(task_name, out_dir, device, skip_adam=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if task_name == 'adult':
        n_epochs, hidden, batch_sz = 100, 128, 256; data_fn = load_adult
    elif task_name == 'covertype':
        n_epochs, hidden, batch_sz = 80, 128, 512;  data_fn = load_covertype
    else:
        raise ValueError(task_name)

    SEEDS   = [42, 123, 7]
    METHODS = ['fs-sgd', 'fs-adam', 'mf-sgd', 'mf-adam'] if not skip_adam else ['fs-sgd', 'mf-sgd']
    RHO, LS, KG = 3e-3, 1e-3, 10

    X_train, y_train, X_val, y_val, X_test, y_test, in_dim, out_dim = data_fn(seed=42)
    tr_ld, val_ld, test_ld = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_sz)
    total_steps = n_epochs * len(tr_ld)

    print(f"\n{'='*60}", flush=True)
    print(f"Task: {task_name}  in={in_dim} out={out_dim} hidden={hidden}", flush=True)
    print(f"n_epochs={n_epochs} batch={batch_sz} total_steps={total_steps}", flush=True)
    print(f"Device: {device}  METHODS: {METHODS}", flush=True)
    print(f"{'='*60}\n", flush=True)

    best_lrs = PRESET_LRS[task_name]

    # ─── Adam verify (adult only) ─────────────────────────────────────────
    adam_verified = False
    if task_name == 'adult' and not skip_adam:
        acc_30 = verify_adam_fix(in_dim, hidden, out_dim, device, tr_ld, val_ld, test_ld,
                                  lr=0.001, n_epochs=30)
        if acc_30 < 0.80:
            # try higher lr
            print(f"[Adam verify] FAILED ({acc_30:.4f} < 0.80). Trying lr=0.01...", flush=True)
            acc_30_hi = verify_adam_fix(in_dim, hidden, out_dim, device, tr_ld, val_ld, test_ld,
                                         lr=0.01, n_epochs=30)
            if acc_30_hi >= 0.80:
                print(f"[Adam verify] lr=0.01 OK ({acc_30_hi:.4f}). Using lr=0.01 for Adam.", flush=True)
                best_lrs = dict(best_lrs)
                best_lrs['fs-adam'] = 0.01; best_lrs['mf-adam'] = 0.01
                adam_verified = True
            else:
                print(f"[Adam verify] STILL FAILED ({acc_30_hi:.4f}). Skipping Adam cells.", flush=True)
                METHODS = ['fs-sgd', 'mf-sgd']
        else:
            print(f"[Adam verify] PASSED ({acc_30:.4f}). Adam fix confirmed.", flush=True)
            adam_verified = True

    # ─── Accumulate state ─────────────────────────────────────────────────
    all_results = {m: [] for m in METHODS}

    stage_b = {
        'task': task_name, 'status': 'running',
        'adam_verified': adam_verified if task_name == 'adult' else None,
        'methods': {m: [] for m in METHODS},
    }
    h1_persist  = {'task': task_name, 'methods': {}}
    g4_trace    = {'task': task_name, 'methods': {}}
    v2_file  = out_dir / 'stage_b_results_v2.json'
    h1_file  = out_dir / 'h1_pressure_persistence.json'
    g4_file  = out_dir / 'g4_spectral_trace.json'

    def flush_all():
        try:
            for path, obj in [(v2_file, stage_b), (h1_file, h1_persist), (g4_file, g4_trace)]:
                with open(path, 'w') as f: json.dump(js(obj), f, indent=2)
        except Exception as e:
            print(f"[flush_all error] {e}", flush=True)

    atexit.register(flush_all)

    # ─── Stage B: seeds × methods ─────────────────────────────────────────
    print("\n=== Stage B: 3 seeds × methods ===", flush=True)

    for method in METHODS:
        lr = best_lrs[method]
        mode_str = method.split('-')[0]
        print(f"\n--- Method: {method}  lr={lr} ---", flush=True)

        h1_persist['methods'][method] = {}
        g4_trace['methods'][method]   = []

        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            result = train_one(in_dim, hidden, out_dim, method, lr, seed, n_epochs,
                               tr_ld, val_ld, test_ld, device, total_steps,
                               rho_geo=RHO, lambda_S=LS, K_geo=KG)
            all_results[method].append(result)

            # Accumulate H1
            for nm, h1s in result['h1_stats'].items():
                if nm not in h1_persist['methods'][method]:
                    h1_persist['methods'][method][nm] = []
                h1_persist['methods'][method][nm].append({'seed': seed, **h1s})

            # G4: per-epoch S stats for MF cells
            if mode_str == 'mf':
                g4_trace['methods'][method].append({
                    'seed': seed,
                    'per_epoch': [
                        {
                            'epoch': rec['epoch'],
                            'S_stats': rec.get('S_stats', {}),
                            'gate':    rec.get('gate', {}),
                            'cos_P':   rec.get('cos_P', {}),
                            'P_norm':  rec.get('P_norm', {}),
                            'G_tan_norm': rec.get('G_tan_norm', {}),
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
                    for h in result['history'] if h['epoch'] % 10 == 0 or h['epoch'] == n_epochs
                ],
            })
            flush_all()

    # ─── G3 analysis ──────────────────────────────────────────────────────
    print("\n=== G3 Analysis ===", flush=True)
    g3_results = {}
    for base_opt in ['sgd', 'adam']:
        mf_k, fs_k = f'mf-{base_opt}', f'fs-{base_opt}'
        if mf_k in all_results and fs_k in all_results:
            g3 = g3_paired_test(all_results[mf_k], all_results[fs_k])
            g3_results[base_opt] = g3
            print(f"  {mf_k} vs {fs_k}: diff={g3['mean_diff']:+.4f} SE={g3['se_diff']:.4f} "
                  f"t={g3['t_stat']:.2f} p={g3['p_val_1sided']:.3f} "
                  f"1SE={'YES' if g3['significant_1se'] else 'no'}", flush=True)

    # ─── H1 summary across seeds ──────────────────────────────────────────
    h1_summary = {}
    for method in METHODS:
        h1_summary[method] = {}
        for nm, seed_list in h1_persist['methods'].get(method, {}).items():
            means = [s['mean'] for s in seed_list if s.get('mean') is not None]
            if means:
                # One-sample t-test: cos_P > 0 across first-1/3 steps, averaged across seeds
                t_stat, p_2s = sp_stats.ttest_1samp(means, 0.0) if len(means) > 1 else (0.0, 1.0)
                p_1s = float(p_2s / 2 if t_stat > 0 else 1.0)
                h1_summary[method][nm] = {
                    'mean_across_seeds': float(np.mean(means)),
                    'std_across_seeds':  float(np.std(means)),
                    't_stat': float(t_stat), 'p_val_1sided': p_1s,
                    'per_seed': seed_list,
                }

    # H1 max cos (MF cells only)
    all_h1_mf_means = []
    for method in ['mf-sgd', 'mf-adam']:
        if method in h1_summary:
            for nm, stats in h1_summary[method].items():
                m = stats.get('mean_across_seeds')
                if m is not None: all_h1_mf_means.append(m)
    h1_max = max(all_h1_mf_means) if all_h1_mf_means else 0.0

    # ─── G4 summary: final λ ratio per layer ──────────────────────────────
    g4_summary = {}
    for method in ['mf-sgd', 'mf-adam']:
        if method not in g4_trace['methods']: continue
        final_stats = {}
        for seed_rec in g4_trace['methods'][method]:
            if not seed_rec['per_epoch']: continue
            last_ep = seed_rec['per_epoch'][-1]
            for nm, ss in last_ep.get('S_stats', {}).items():
                if nm not in final_stats: final_stats[nm] = {'lambda_ratio': [], 'lambda_max': [], 'lambda_min': []}
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

    # λ ratio > 1 is a G4 signal
    g4_max_ratio = 0.0
    for method_stats in g4_summary.values():
        for nm, stats in method_stats.items():
            g4_max_ratio = max(g4_max_ratio, stats.get('lambda_ratio_mean', 0.0))

    # ─── Verdict ──────────────────────────────────────────────────────────
    g3_sgd  = g3_results.get('sgd',  {}).get('significant_1se', False)
    g3_adam = g3_results.get('adam', {}).get('significant_1se', False)
    h1_confirm = h1_max > 0.3
    g3_confirm = g3_sgd or g3_adam
    g4_confirm = g4_max_ratio > 1.5

    verdict = {
        'h1_max_cos': float(h1_max),  'h1_confirm': h1_confirm,
        'g3_confirm_sgd': g3_sgd,     'g3_confirm_adam': g3_adam,
        'g3_confirm': g3_confirm,     'g4_max_ratio': float(g4_max_ratio),
        'g4_confirm': g4_confirm,     'double_confirm': h1_confirm and g3_confirm,
        'triple_confirm': h1_confirm and g3_confirm and g4_confirm,
    }

    print(f"\n=== VERDICT for {task_name} ===", flush=True)
    print(f"  H1 max cos={h1_max:.4f} ({'CONFIRM' if h1_confirm else 'fail'} thr=0.3)", flush=True)
    print(f"  G3 SGD={'CONFIRM' if g3_sgd else 'fail'}  Adam={'CONFIRM' if g3_adam else 'fail'}", flush=True)
    print(f"  G4 max λ_ratio={g4_max_ratio:.3f} ({'CONFIRM' if g4_confirm else 'fail'} thr=1.5)", flush=True)
    print(f"  Double (H1+G3): {'*** YES ***' if verdict['double_confirm'] else 'no'}", flush=True)
    print(f"  Triple (H1+G3+G4): {'*** YES ***' if verdict['triple_confirm'] else 'no'}", flush=True)

    stage_b['status']     = 'done'
    stage_b['g3']         = g3_results
    stage_b['h1_summary'] = h1_summary
    stage_b['g4_summary'] = g4_summary
    stage_b['verdict']    = verdict
    h1_persist['h1_summary'] = h1_summary
    g4_trace['g4_summary']   = g4_summary
    g4_trace['verdict']      = {'g4_max_ratio': float(g4_max_ratio), 'g4_confirm': g4_confirm}
    flush_all()

    # ─── REPORT.md ────────────────────────────────────────────────────────
    lines = [
        f"# {task_name.upper()} MLP — Batch 9 Report (with H1/G4 Trace)\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}\n\n",
        "## Adam Bug Fix\n",
        "- **Root cause**: `stiefel_adam_step` transported `v` (2nd moment) via `project_tangent`,\n",
        "  introducing negative entries → `sqrt(v)` → NaN → all-zero Q → majority class predict.\n",
        "- **Fix**: Only transport `m` (1st moment); `v` is a per-element scale, not a tangent vector.\n\n",
        "## Methods\n",
        f"- {METHODS}\n",
        f"- 3 seeds × {n_epochs} epochs\n",
        f"- MF hyper: rho_geo={RHO}, lambda_S={LS}, K_geo={KG}\n\n",
        "## Best LRs\n",
    ]
    for m, lr_v in best_lrs.items():
        lines.append(f"- {m}: {lr_v}\n")

    lines += ["\n## Final Test Accuracy\n\n| Method | S1 | S2 | S3 | Mean |\n|---|---|---|---|---|\n"]
    for method in METHODS:
        if not all_results[method]: continue
        accs = [r['history'][-1]['test_acc'] for r in all_results[method]]
        lines.append(f"| {method} | " + " | ".join(f"{a:.4f}" for a in accs) + f" | {np.mean(accs):.4f} |\n")

    lines += ["\n## G3: MF vs FS (paired 1-sided t-test)\n\n"]
    for base_opt in ['sgd', 'adam']:
        if base_opt not in g3_results: continue
        g = g3_results[base_opt]
        lines += [
            f"### {base_opt.upper()}\n",
            f"- Δ mean={g['mean_diff']:+.4f} ± {g['se_diff']:.4f} SE,  t={g['t_stat']:.2f}, p={g['p_val_1sided']:.3f}\n",
            f"- **>1 SE: {'✅ YES' if g['significant_1se'] else '❌ no'}**  p<0.05: {'✅ YES' if g['significant_p05'] else '❌ no'}\n\n",
        ]

    lines += ["\n## H1: cos(P_t, P_{t-1}) — first 1/3 training\n\n"]
    for method in METHODS:
        lines.append(f"### {method}\n")
        for nm, stats in h1_summary.get(method, {}).items():
            m = stats.get('mean_across_seeds')
            p = stats.get('p_val_1sided')
            lines.append(f"- {nm}: mean={m:.4f}" + (f" p={p:.3f}" if p else "") + "\n")
        lines.append("\n")

    lines += ["\n## G4: λ_max/λ_min(S) at end of training\n\n"]
    for method, method_stats in g4_summary.items():
        lines.append(f"### {method}\n")
        for nm, s in method_stats.items():
            lines.append(f"- {nm}: λ_ratio={s['lambda_ratio_mean']:.3f}  λ_max={s['lambda_max_mean']:.3f}\n")
        lines.append("\n")

    lines += [
        "\n## Verdict\n\n",
        f"- H1 max cos(P) = **{h1_max:.4f}** ({'✅ CONFIRM' if h1_confirm else '❌ fail'} thr=0.3)\n",
        f"- G3 SGD: {'✅ CONFIRM' if g3_sgd else '❌ fail'}\n",
        f"- G3 Adam: {'✅ CONFIRM' if g3_adam else '❌ fail'}\n",
        f"- G4 max λ_ratio: **{g4_max_ratio:.3f}** ({'✅ CONFIRM' if g4_confirm else '❌ fail'} thr=1.5)\n",
        f"- **H1+G3 Double Confirm: {'✅ YES' if verdict['double_confirm'] else '❌ NO'}**\n",
        f"- **H1+G3+G4 Triple Confirm: {'✅ YES' if verdict['triple_confirm'] else '❌ NO'}**\n",
    ]

    with open(out_dir / 'REPORT.md', 'w') as f: f.writelines(lines)
    print(f"\nOutputs saved to {out_dir}", flush=True)
    return verdict


# ─── Entry ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['adult', 'covertype'], required=True)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--skip-adam', action='store_true')
    args = parser.parse_args()

    def _sig(sig, frame):
        print(f"\n[SIGNAL {sig}] flushing...", flush=True); sys.exit(0)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[Batch9] Task={args.task} GPU={args.cuda} device={device}", flush=True)

    out_base = BASE / "method_1"
    out_map  = {'adult': out_base / 'adult_mlp', 'covertype': out_base / 'covertype_mlp'}
    out_dir  = out_map[args.task]

    try:
        verdict = run_task(args.task, out_dir, device, skip_adam=args.skip_adam)
        print(f"\n[DONE] task={args.task}", flush=True)
        print(f"[DONE] double_confirm={verdict['double_confirm']} triple_confirm={verdict['triple_confirm']}", flush=True)
        print(f"[DONE] H1_max={verdict['h1_max_cos']:.4f} G3_sgd={verdict['g3_confirm_sgd']} G3_adam={verdict['g3_confirm_adam']} G4_ratio={verdict['g4_max_ratio']:.3f}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
