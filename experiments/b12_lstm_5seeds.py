#!/usr/bin/env python3
"""
Batch 12 — LSTM/WikiText-2 hidden→vocab projection 5-seed confirm
4 cells: FS-SGD, FS-Adam, MF-SGD, MF-Adam
5 seeds: 42 (reuse), 123, 7, 2024, 2025 (4 new)
GPU 1 = SGD cells, GPU 2 = Adam cells (parallel via subprocess)
Mechanism trace per epoch: ‖P_t‖_F, cos(P_t, P_{t-1}), λ_max/λ_min(S_t)
Output: experiments/method_1/lstm_wt2_proj/
Task 2 (A6 random pressure): ablation/a6_random_pressure/lstm_wt2/
"""
import sys, os, json, time, math, warnings, atexit, signal, traceback, argparse, subprocess
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")

BASE = Path("./experiments")
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))

OUT_DIR   = Path("./experiments/method_1/lstm_wt2_proj")
ABL_DIR   = Path("./experiments/ablation/a6_random_pressure/lstm_wt2")
HF_CACHE  = "./datasets/hf_cache"

SEEDS     = [42, 123, 7, 2024, 2025]
N_EPOCHS  = 8
BATCH_SZ  = 32
SEQ_LEN   = 35
EMBED_DIM = 128
HIDDEN_DIM = 128
VOCAB_SIZE_TARGET = 10000
# MF hyper
RHO_GEO   = 0.01
LAMBDA_S  = 0.001
K_GEO     = 10
# LR
LR_ADAM   = 0.003
LR_SGD    = 0.01

# ─── Serializer ─────────────────────────────────────────────────────────────
def js(obj):
    if isinstance(obj, dict):           return {k: js(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [js(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return float(obj.item()) if obj.numel()==1 else obj.tolist()
    except ImportError:
        pass
    if obj is None: return None
    return obj

def flush_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(js(data), f, indent=2)
    os.replace(tmp, str(path))

def set_seed(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─── QR init ────────────────────────────────────────────────────────────────
def _qr_init(n, r, seed=None):
    import torch
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    A = torch.randn(n, r, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.float()

# ─── StiefelLinear ───────────────────────────────────────────────────────────
def build_stiefel_linear(in_dim, out_dim, seed, mode, device):
    """Returns (module, Q_param)"""
    import torch, torch.nn as nn, torch.nn.functional as F
    from manifoldflow.spd_ops import matrix_sqrt, sym

    class StiefelLinear(nn.Module):
        def __init__(self):
            super().__init__()
            if out_dim <= in_dim:
                n, r = in_dim, out_dim; self.transpose = True
            else:
                n, r = out_dim, in_dim; self.transpose = False
            self.n, self.r = n, r
            self.Q    = nn.Parameter(_qr_init(n, r, seed))
            self._sqrtS_cache = torch.eye(r, device=device)
            self.bias = nn.Parameter(torch.zeros(out_dim, device=device))
            self._mode = mode

        def forward(self, x):
            if self._mode == 'fs':
                W_base = self.Q
            else:
                sqrtS = self._sqrtS_cache.to(self.Q.device, self.Q.dtype)
                W_base = self.Q @ sqrtS
            W = W_base.T if self.transpose else W_base
            return F.linear(x, W, self.bias)

        def update_sqrtS(self, S):
            import torch
            with torch.no_grad():
                self._sqrtS_cache = matrix_sqrt(sym(S)).detach()
    return StiefelLinear()

# ─── LSTM Model ─────────────────────────────────────────────────────────────
def make_lstm_model(vocab_size, embed_dim, hidden_dim, mode, seed, device):
    import torch, torch.nn as nn, torch.nn.functional as F
    from manifoldflow.spd_ops import matrix_sqrt, sym

    class LSTMLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=False)
            # Stiefel only on hidden→vocab projection
            # F.linear needs W of shape (out, in) = (vocab_size, hidden_dim)
            # Q must be tall (rows >= cols); no transpose needed when vocab>hidden
            if vocab_size <= hidden_dim:  # out <= in → Q=(hidden_dim, vocab_size), W=Q.T
                n, r = hidden_dim, vocab_size; self.proj_transpose = True
            else:                          # out > in → Q=(vocab_size, hidden_dim), W=Q
                n, r = vocab_size, hidden_dim; self.proj_transpose = False
            self.proj_Q    = nn.Parameter(_qr_init(n, r, seed))
            self._sqrtS    = torch.eye(r, device=device)
            self.proj_bias = nn.Parameter(torch.zeros(vocab_size, device=device))
            nn.init.uniform_(self.embed.weight, -0.1, 0.1)

        def _proj_forward(self, x):
            if mode == 'fs':
                W_base = self.proj_Q
            else:
                sqrtS = self._sqrtS.to(self.proj_Q.device, self.proj_Q.dtype)
                W_base = self.proj_Q @ sqrtS
            W = W_base.T if self.proj_transpose else W_base
            return F.linear(x, W, self.proj_bias)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            logits = self._proj_forward(out.view(-1, out.size(-1)))
            return logits, hidden

        def update_sqrtS(self, S):
            with torch.no_grad():
                self._sqrtS = matrix_sqrt(sym(S)).detach()

    m = LSTMLMModel().to(device)
    return m

# ─── Optimizer builders ─────────────────────────────────────────────────────
def build_optimizer(model, optim_type, mode, lr, total_steps, device):
    from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig, ManifoldFlowOptimizer

    stiefel_params = [p for n,p in model.named_parameters() if 'proj_Q' in n and p.requires_grad]
    other_params   = [p for n,p in model.named_parameters() if 'proj_Q' not in n and p.requires_grad]

    if mode == 'fs':
        cfg = ManifoldFlowConfig(rho_geo=0.0, beta_P=0.95, lambda_S=LAMBDA_S, K_geo=K_GEO)
    else:
        cfg = ManifoldFlowConfig(rho_geo=RHO_GEO, beta_P=0.95, lambda_S=LAMBDA_S, K_geo=K_GEO)

    if optim_type == 'adam':
        opt_mf = ManifoldFlowOptimizer(stiefel_params, base_optim='adam', lr=lr,
                                        betas=(0.9,0.999), mf_config=cfg, total_steps=total_steps)
        opt_base = __import__('torch').optim.Adam(other_params, lr=lr) if other_params else None
    else:  # sgd
        opt_mf = ManifoldFlowOptimizer(stiefel_params, base_optim='sgd', lr=lr,
                                        momentum=0.9, mf_config=cfg, total_steps=total_steps)
        opt_base = __import__('torch').optim.SGD(other_params, lr=lr, momentum=0.9) if other_params else None
    return opt_mf, opt_base

# ─── Data loading ────────────────────────────────────────────────────────────
_DATASET_CACHE = {}

def load_wt2_data(device):
    """Load WikiText-2, return (train_data, val_data, vocab_size, w2i)"""
    import torch
    if 'wt2' in _DATASET_CACHE:
        return _DATASET_CACHE['wt2']

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=HF_CACHE)
    train_text = "\n".join(ds["train"]["text"])
    val_text   = "\n".join(ds["validation"]["text"])

    words   = train_text.split()
    counter = Counter(words)
    vocab_words = ["<unk>", "<eos>"] + [w for w,_ in counter.most_common(VOCAB_SIZE_TARGET-2)]
    w2i = {w:i for i,w in enumerate(vocab_words)}
    vocab_size = len(vocab_words)

    def text_to_ids(text):
        return torch.tensor([w2i.get(w,0) for w in text.split()], dtype=torch.long)

    def batchify(data, bsz, dev):
        nb = data.size(0) // bsz
        return data[:nb*bsz].view(bsz, -1).t().contiguous().to(dev)

    train_ids = text_to_ids(train_text)
    val_ids   = text_to_ids(val_text)
    train_data = batchify(train_ids, BATCH_SZ, device)
    val_data   = batchify(val_ids,   1, device)

    result = (train_data, val_data, vocab_size, w2i)
    _DATASET_CACHE['wt2'] = result
    print(f"  WikiText-2 loaded: vocab={vocab_size}, train_tokens={train_ids.numel()}")
    return result

def get_batch(source, i):
    import torch
    sl = min(SEQ_LEN, len(source) - 1 - i)
    x  = source[i:i+sl]
    y  = source[i+1:i+1+sl].reshape(-1)
    return x, y

def eval_ppl(model, val_data):
    import torch, torch.nn.functional as F
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for i in range(0, val_data.size(0)-1, SEQ_LEN):
            x, y = get_batch(val_data, i)
            logits, _ = model(x)
            total_loss   += F.cross_entropy(logits, y, reduction='sum').item()
            total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)

# ─── Mechanism trace helpers ──────────────────────────────────────────────────
def get_trace_from_opt(model, opt_mf, prev_mp=None):
    """
    Extract per-epoch mechanism trace for the projection layer.
    Returns (trace_dict, current_M_P_for_next_epoch)
    """
    import torch
    Q = model.proj_Q
    state = opt_mf.state.get(Q, {})
    plog  = opt_mf._pressure_log.get(id(Q), {})

    P_norm  = plog.get("P_norm", float('nan'))
    lam_min = plog.get("lambda_min", float('nan'))
    lam_max = plog.get("lambda_max", float('nan'))

    # cos(P_t, P_{t-1}): approximate via M_P EMA direction
    M_P_curr = state.get("M_P", None)
    cos_val  = float('nan')
    if M_P_curr is not None and prev_mp is not None:
        with torch.no_grad():
            a = M_P_curr.float().flatten()
            b = prev_mp.to(a.device).float().flatten()  # ensure same device
            denom = a.norm() * b.norm()
            if denom > 1e-10:
                cos_val = (a @ b / denom).item()

    curr_mp = M_P_curr.clone().detach().cpu() if M_P_curr is not None else None
    trace = {
        "P_norm_frob":  P_norm,
        "cos_Pt_Ptm1":  cos_val,
        "lambda_min_S": lam_min,
        "lambda_max_S": lam_max,
    }
    return trace, curr_mp

# ─── Single cell run ─────────────────────────────────────────────────────────
def run_cell(optim_type, mode, seed, device, train_data, val_data, vocab_size,
             a6_random_pressure=False):
    """
    Run one (optim_type, mode, seed) combination.
    Returns result dict with PPL curve + mechanism trace.
    """
    import torch, torch.nn.functional as F

    cell_name = f"{'MF' if mode=='mf' else 'FS'}-{optim_type.upper()}"
    if a6_random_pressure:
        cell_name = "A6-random"
    print(f"\n  [{cell_name} seed={seed}] starting ...")
    set_seed(seed)

    lr = LR_ADAM if optim_type == 'adam' else LR_SGD
    n_batches = (train_data.size(0) - 1) // SEQ_LEN
    total_steps = N_EPOCHS * n_batches

    model = make_lstm_model(vocab_size, EMBED_DIM, HIDDEN_DIM, mode, seed, device)
    opt_mf, opt_base = build_optimizer(model, optim_type, mode, lr, total_steps, device)

    # For A6 random pressure: monkey-patch opt_mf.step to replace P_t with random
    if a6_random_pressure and mode == 'mf':
        _orig_step = opt_mf.step.__func__
        import types
        from manifoldflow.spd_ops import sym
        from manifoldflow.tangent import decompose_tangent_normal
        from manifoldflow.retraction import qr_retract, procrustes_align
        from manifoldflow.spd_ops import symlogm, affine_invariant_step, spectral_clip, fp32_eigh

        def _a6_step(self, closure=None):
            """Same as MF step but replace P_t with random symmetric matrix of equal norm."""
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            cfg = self.mf_config
            eps = 1e-8
            for group in self.param_groups:
                lr_g = group["lr"]
                momentum = group["momentum"]
                gamma_t = cfg.rho_geo * lr_g
                for Q in group["params"]:
                    if Q.grad is None: continue
                    G_bar = Q.grad.to(Q.dtype)
                    state = self.state[Q]
                    if len(state) == 0:
                        state["step"] = 0
                        r = Q.shape[-1]
                        state["S"] = torch.eye(r, dtype=Q.dtype, device=Q.device)
                        state["M_P"] = torch.zeros(r, r, dtype=Q.dtype, device=Q.device)
                        state["Q_prev"] = Q.clone()
                    t = state["step"]
                    S = state["S"]
                    M_P = state["M_P"]
                    Q_prev = state["Q_prev"]
                    split = decompose_tangent_normal(Q, G_bar)
                    G_tan = split.G_tan
                    P_t_real = split.P
                    P_norm_real = P_t_real.norm() + eps

                    # A6: replace P_t with random symmetric matrix of equal norm
                    r_size = P_t_real.shape[0]
                    R = torch.randn(r_size, r_size, device=Q.device, dtype=Q.dtype)
                    R_sym = sym(R)
                    P_t = R_sym / (R_sym.norm() + eps) * P_norm_real

                    from manifoldflow.manifoldflow_optimizer import _stiefel_sgd_step
                    Q_new = _stiefel_sgd_step(Q, G_tan, state, lr_g, momentum)

                    if t > 0:
                        A = Q.T @ Q_prev
                        O_t = procrustes_align(A)
                        M_P_aligned = O_t @ M_P @ O_t.T
                    else:
                        M_P_aligned = M_P

                    G_bar_norm = G_bar.norm() + eps
                    P_normalized = P_t / G_bar_norm
                    M_P_prev = M_P_aligned.clone()
                    M_P_new = cfg.beta_P * M_P_aligned + (1.0 - cfg.beta_P) * P_normalized
                    state["M_P"] = M_P_new

                    warmup_done = t >= self._warmup_steps()
                    do_geo = (gamma_t > 0.0) and warmup_done and (t % cfg.K_geo == 0)

                    if do_geo:
                        P_norm = P_t.norm() + eps
                        M_prev_norm = M_P_prev.norm() + eps
                        c_t = (P_t * M_P_prev).sum() / (P_norm * M_prev_norm)
                        G_nor_norm = (Q @ P_t).norm() + eps
                        G_tan_norm = G_tan.norm() + eps
                        r_t = G_nor_norm / G_tan_norm
                        log_r_t = torch.log(r_t)
                        a_t_c = torch.sigmoid(torch.tensor(cfg.alpha_c * (c_t.item() - cfg.tau_c), dtype=Q.dtype, device=Q.device))
                        a_t_r = torch.sigmoid(cfg.alpha_r * (log_r_t - cfg.tau_r))
                        a_t = (a_t_c * a_t_r).item()
                        H_t = sym(M_P_new) + cfg.lambda_S * symlogm(S)
                        S_raw = affine_invariant_step(S, H_t, gamma_t * a_t)
                        state["S"] = spectral_clip(S_raw, cfg.lambda_min, cfg.lambda_max)

                    Q.data.copy_(Q_new)
                    state["Q_prev"] = Q_new.clone()
                    state["step"] = t + 1

                    if self.log_pressure:
                        eigvals, _ = fp32_eigh(state["S"])
                        self._pressure_log[id(Q)] = {
                            "P_norm": P_t_real.norm().item(),  # real P_norm for trace
                            "grad_tan_norm": G_tan.norm().item(),
                            "lambda_min": eigvals.min().item(),
                            "lambda_max": eigvals.max().item(),
                            "step": t,
                        }
            return loss

        opt_mf.step = types.MethodType(_a6_step, opt_mf)

    best_ppl  = float('inf')
    epoch_ppls = []
    mechanism_trace = []
    prev_mp = None

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for i in range(0, train_data.size(0)-1, SEQ_LEN):
            x, y = get_batch(train_data, i)
            opt_mf.zero_grad()
            if opt_base is not None:
                opt_base.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt_mf.step()
            if opt_base is not None:
                opt_base.step()
            if mode == 'mf':
                Q = model.proj_Q
                if Q in opt_mf.state and 'S' in opt_mf.state[Q]:
                    model.update_sqrtS(opt_mf.state[Q]['S'])

        ppl = eval_ppl(model, val_data)
        epoch_ppls.append(ppl)
        if ppl < best_ppl:
            best_ppl = ppl

        # Mechanism trace
        if mode == 'mf':
            trace, prev_mp = get_trace_from_opt(model, opt_mf, prev_mp)
            mechanism_trace.append(trace)

        print(f"    [{cell_name} seed={seed}] ep{epoch+1}/{N_EPOCHS} ppl={ppl:.2f} best={best_ppl:.2f}")

    elapsed = time.time() - t0
    result = {
        "optim":   optim_type,
        "mode":    mode,
        "seed":    seed,
        "a6":      a6_random_pressure,
        "best_ppl":    best_ppl,
        "epoch_ppls":  epoch_ppls,
        "elapsed":     elapsed,
    }
    if mechanism_trace:
        result["mechanism_trace"] = mechanism_trace
    return result

# ─── Worker process ──────────────────────────────────────────────────────────
def worker_main(optim_type, gpu_idx):
    """Run all seeds for a given optim_type on a specific GPU."""
    import torch
    device = torch.device(f"cuda:{gpu_idx}")
    print(f"\n{'='*60}")
    print(f"WORKER: optim={optim_type.upper()}, GPU={gpu_idx}")
    print(f"{'='*60}")

    # Load data once
    train_data, val_data, vocab_size, w2i = load_wt2_data(device)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing partial results to resume from where we left off
    partial_path = out_dir / f"partial_{optim_type}.json"
    if partial_path.exists():
        try:
            with open(partial_path) as f:
                worker_results = json.load(f)
            print(f"  Loaded existing partial: {partial_path}")
        except Exception:
            worker_results = {}
    else:
        worker_results = {}

    _atexit_data = {"path": partial_path, "data": worker_results}
    def _atexit_fn():
        flush_json(_atexit_data["path"], _atexit_data["data"])
        print(f"[atexit] flushed {_atexit_data['path']}")
    atexit.register(_atexit_fn)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))

    for mode in ['fs', 'mf']:
        cell_key = f"{'MF' if mode=='mf' else 'FS'}-{optim_type.upper()}"
        if cell_key not in worker_results:
            worker_results[cell_key] = {}
        for seed in SEEDS:
            # Skip if already done
            existing = worker_results.get(cell_key, {}).get(str(seed), {})
            if isinstance(existing, dict) and "best_ppl" in existing:
                print(f"  [{cell_key} seed={seed}] SKIP (already done PPL={existing['best_ppl']:.2f})")
                continue
            try:
                res = run_cell(optim_type, mode, seed, device, train_data, val_data, vocab_size)
                worker_results[cell_key][str(seed)] = res
                _atexit_data["data"] = worker_results
                flush_json(partial_path, worker_results)
                print(f"  [{cell_key} seed={seed}] DONE PPL={res['best_ppl']:.2f}")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [{cell_key} seed={seed}] FAILED: {e}\n{tb}")
                worker_results[cell_key][str(seed)] = {"status":"failed","error":str(e)}
                _atexit_data["data"] = worker_results
                flush_json(partial_path, worker_results)

    flush_json(partial_path, worker_results)
    print(f"\n[WORKER {optim_type}] Done. Results: {partial_path}")

# ─── Task 2: A6 ablation ─────────────────────────────────────────────────────
def run_a6_ablation():
    """A6 random pressure control: 3 seeds, GPU 2, MF-Adam only."""
    import torch
    device = torch.device("cuda:2")
    print(f"\n{'='*60}")
    print("TASK 2: A6 Random Pressure Ablation (LSTM MF-Adam vs A6)")
    print(f"{'='*60}")

    ABL_DIR.mkdir(parents=True, exist_ok=True)
    train_data, val_data, vocab_size, _ = load_wt2_data(device)

    a6_results  = {}
    mfa_results = {}
    seeds_a6 = [42, 123, 7]
    partial = ABL_DIR / "results.json"

    _abl_data = {"a6": a6_results, "mf_adam": mfa_results}
    def _atexit_abl():
        flush_json(partial, _abl_data)
    atexit.register(_atexit_abl)

    for seed in seeds_a6:
        # MF-Adam baseline
        try:
            res_mf = run_cell('adam', 'mf', seed, device, train_data, val_data, vocab_size,
                               a6_random_pressure=False)
            mfa_results[str(seed)] = res_mf
            _abl_data["mf_adam"] = mfa_results
            flush_json(partial, _abl_data)
            print(f"  [MF-Adam seed={seed}] DONE PPL={res_mf['best_ppl']:.2f}")
        except Exception as e:
            print(f"  [MF-Adam seed={seed}] FAILED: {e}")
            mfa_results[str(seed)] = {"status":"failed","error":str(e)}

        # A6 random pressure
        try:
            res_a6 = run_cell('adam', 'mf', seed, device, train_data, val_data, vocab_size,
                               a6_random_pressure=True)
            a6_results[str(seed)] = res_a6
            _abl_data["a6"] = a6_results
            flush_json(partial, _abl_data)
            print(f"  [A6-random seed={seed}] DONE PPL={res_a6['best_ppl']:.2f}")
        except Exception as e:
            print(f"  [A6-random seed={seed}] FAILED: {e}")
            a6_results[str(seed)] = {"status":"failed","error":str(e)}

    # Compute summary
    mf_ppls = [v["best_ppl"] for v in mfa_results.values() if "best_ppl" in v]
    a6_ppls = [v["best_ppl"] for v in a6_results.values()  if "best_ppl" in v]
    summary = {
        "mf_adam_mean": float(np.mean(mf_ppls)) if mf_ppls else None,
        "mf_adam_std":  float(np.std(mf_ppls))  if mf_ppls else None,
        "a6_mean":      float(np.mean(a6_ppls))  if a6_ppls else None,
        "a6_std":       float(np.std(a6_ppls))   if a6_ppls else None,
        "delta_mf_vs_a6": float(np.mean(a6_ppls) - np.mean(mf_ppls)) if (mf_ppls and a6_ppls) else None,
    }
    final = {"mf_adam": mfa_results, "a6": a6_results, "summary": summary}
    flush_json(ABL_DIR / "results.json", final)
    print(f"\n  A6 summary: MF-Adam={summary['mf_adam_mean']:.2f}±{summary['mf_adam_std']:.2f} PPL")
    print(f"              A6-rand={summary['a6_mean']:.2f}±{summary['a6_std']:.2f} PPL")
    print(f"              Δ(A6-MF)={summary['delta_mf_vs_a6']:+.2f} PPL (positive = A6 worse = mechanism exists)")
    return final

# ─── Aggregate + report ──────────────────────────────────────────────────────
def aggregate_and_report(sgd_path, adam_path):
    """Merge partial results, compute mean±std, write final JSON + REPORT.md"""
    sgd_res  = json.load(open(sgd_path))  if sgd_path.exists()  else {}
    adam_res = json.load(open(adam_path)) if adam_path.exists() else {}
    all_cells = {**sgd_res, **adam_res}

    # Cells expected: FS-SGD, MF-SGD, FS-ADAM, MF-ADAM
    summary = {}
    for cell, seed_dict in all_cells.items():
        ppls = [v["best_ppl"] for v in seed_dict.values() if isinstance(v, dict) and "best_ppl" in v]
        if not ppls:
            summary[cell] = {"mean": None, "std": None, "n": 0}
            continue
        summary[cell] = {
            "mean": float(np.mean(ppls)),
            "std":  float(np.std(ppls)),
            "n":    len(ppls),
            "ppls": ppls,
        }

    # Mechanism trace summary (MF cells only)
    mf_trace_summary = {}
    for cell_key in ["MF-ADAM", "MF-SGD"]:
        cell = all_cells.get(cell_key, {})
        all_traces = []
        for seed_v in cell.values():
            if isinstance(seed_v, dict) and "mechanism_trace" in seed_v:
                all_traces.extend(seed_v["mechanism_trace"])
        if all_traces:
            P_norms  = [t["P_norm_frob"]  for t in all_traces if not math.isnan(t.get("P_norm_frob", float('nan')))]
            cos_vals = [t["cos_Pt_Ptm1"]  for t in all_traces if not math.isnan(t.get("cos_Pt_Ptm1", float('nan')))]
            lam_mins = [t["lambda_min_S"] for t in all_traces if not math.isnan(t.get("lambda_min_S", float('nan')))]
            lam_maxs = [t["lambda_max_S"] for t in all_traces if not math.isnan(t.get("lambda_max_S", float('nan')))]
            mf_trace_summary[cell_key] = {
                "P_norm_mean":   float(np.mean(P_norms))  if P_norms  else None,
                "cos_mean":      float(np.mean(cos_vals)) if cos_vals else None,
                "lambda_min_range": [float(np.min(lam_mins)), float(np.max(lam_mins))] if lam_mins else None,
                "lambda_max_range": [float(np.min(lam_maxs)), float(np.max(lam_maxs))] if lam_maxs else None,
            }

    # Delta MF-FS for each optim
    delta_adam = delta_sgd = None
    fs_adam = summary.get("FS-ADAM", {}).get("mean")
    mf_adam = summary.get("MF-ADAM", {}).get("mean")
    fs_sgd  = summary.get("FS-SGD",  {}).get("mean")
    mf_sgd  = summary.get("MF-SGD",  {}).get("mean")
    if fs_adam is not None and mf_adam is not None:
        delta_adam = fs_adam - mf_adam  # positive = MF better (lower PPL)
    if fs_sgd is not None and mf_sgd is not None:
        delta_sgd = fs_sgd - mf_sgd

    final = {
        "task":       "lstm_wt2_proj",
        "stage":      "b",
        "seeds":      SEEDS,
        "n_epochs":   N_EPOCHS,
        "cells":      all_cells,
        "summary":    summary,
        "delta_adam_fs_minus_mf": delta_adam,
        "delta_sgd_fs_minus_mf":  delta_sgd,
        "mechanism_trace": mf_trace_summary,
        "g3_confirmed_adam": delta_adam is not None and delta_adam > 5.0,
        "g3_confirmed_sgd":  delta_sgd  is not None and delta_sgd  > 5.0,
        "g6_cross_optim":    delta_adam is not None and delta_sgd is not None and delta_adam > 0 and delta_sgd > 0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flush_json(OUT_DIR / "stage_b_results_5seeds.json", final)

    # Write REPORT.md
    lines = [
        "# Batch 12 — LSTM/WikiText-2 Projection 5-Seed Confirm",
        "",
        "## Summary: 4 Cells × 5 Seeds",
        "",
        "| Cell | Mean PPL | Std PPL | N |",
        "|------|----------|---------|---|",
    ]
    for cell in ["FS-SGD", "MF-SGD", "FS-ADAM", "MF-ADAM"]:
        s = summary.get(cell, {})
        mean = f"{s['mean']:.2f}" if s.get("mean") else "N/A"
        std  = f"{s['std']:.2f}"  if s.get("std")  else "N/A"
        n    = s.get("n", 0)
        lines.append(f"| {cell} | {mean} | {std} | {n} |")

    lines += [
        "",
        "## G2/G3 Verdict: MF vs FS Delta",
        "",
        f"- **Adam**: Δ(FS-MF) = {delta_adam:.2f} PPL (positive = MF better)" if delta_adam else "- **Adam**: N/A",
        f"- **SGD**:  Δ(FS-MF) = {delta_sgd:.2f} PPL"  if delta_sgd  else "- **SGD**: N/A",
        f"- **G3 confirmed (Adam)**: {final['g3_confirmed_adam']}  (threshold: Δ>5 PPL)",
        f"- **G3 confirmed (SGD)**:  {final['g3_confirmed_sgd']}",
        f"- **G6 cross-optim**: {final['g6_cross_optim']} (both optims positive)",
        "",
    ]

    if mf_trace_summary:
        lines += ["## H1/G4 Mechanism Trace (MF cells, all seeds all epochs)", ""]
        for cell, tr in mf_trace_summary.items():
            lines.append(f"### {cell}")
            lines.append(f"- ‖P_t‖_F mean: {tr['P_norm_mean']:.4f}" if tr.get("P_norm_mean") else "- ‖P_t‖_F: N/A")
            if tr.get("cos_mean"):
                cos = tr["cos_mean"]
                h1_strong = abs(cos) > 0.7
                lines.append(f"- cos(P_t, P_{{t-1}}) mean: {cos:.4f} — H1 {'STRONG ✓' if h1_strong else 'weak'}")
            if tr.get("lambda_min_range"):
                lines.append(f"- λ_min(S) range: [{tr['lambda_min_range'][0]:.4f}, {tr['lambda_min_range'][1]:.4f}]")
                lines.append(f"- λ_max(S) range: [{tr['lambda_max_range'][0]:.4f}, {tr['lambda_max_range'][1]:.4f}]")
            lines.append("")

    lines += [
        "## Recommendation",
        "",
    ]
    if final["g3_confirmed_adam"] and final["g6_cross_optim"]:
        rec = "**STRONG SIGNAL CONFIRMED**: G3 (5-seed Δ>5 PPL Adam) + G6 (cross-optim) both hold. Proceed to paper writing."
    elif final["g3_confirmed_adam"]:
        rec = "**G3 CONFIRMED on Adam**: 5-seed Δ>5 PPL. SGD signal weaker, but Adam result is strong."
    elif delta_adam and delta_adam > 0:
        rec = f"Positive signal (Δ={delta_adam:.2f}) but below G3 threshold. Investigate."
    else:
        rec = "Signal not confirmed at 5-seed level. Review individual seeds."
    lines.append(rec)

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\n[AGGREGATE] Results: {OUT_DIR / 'stage_b_results_5seeds.json'}")
    print(f"[AGGREGATE] Report:  {OUT_DIR / 'REPORT.md'}")
    return final

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--optim-type", choices=["sgd", "adam"])
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--a6-only", action="store_true")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.optim_type, args.gpu)
        return

    if args.a6_only:
        run_a6_ablation()
        return

    # Master: launch two workers in parallel
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Batch 12 — LSTM/WikiText-2 5-seed confirm")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {OUT_DIR}")
    print("="*70)

    script = Path(__file__).resolve()
    env = os.environ.copy()

    t_start = time.time()

    # Launch SGD worker (GPU 1)
    proc_sgd = subprocess.Popen(
        [sys.executable, str(script), "--worker", "--optim-type", "sgd", "--gpu", "1"],
        env={**env, "CUDA_VISIBLE_DEVICES": "0,1,2"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    # Launch Adam worker (GPU 2)
    proc_adam = subprocess.Popen(
        [sys.executable, str(script), "--worker", "--optim-type", "adam", "--gpu", "2"],
        env={**env, "CUDA_VISIBLE_DEVICES": "0,1,2"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def drain(label, proc, logfile):
        with open(logfile, "w") as f:
            for line in proc.stdout:
                f.write(line)
                print(f"[{label}] {line}", end="")

    import threading
    sgd_log  = str(OUT_DIR / "worker_sgd.log")
    adam_log = str(OUT_DIR / "worker_adam.log")
    t_sgd  = threading.Thread(target=drain, args=("SGD",  proc_sgd,  sgd_log),  daemon=True)
    t_adam = threading.Thread(target=drain, args=("ADAM", proc_adam, adam_log), daemon=True)
    t_sgd.start(); t_adam.start()

    rc_sgd  = proc_sgd.wait()
    rc_adam = proc_adam.wait()
    t_sgd.join(); t_adam.join()

    elapsed = time.time() - t_start
    print(f"\n[MASTER] Both workers done in {elapsed/60:.1f} min. rc_sgd={rc_sgd}, rc_adam={rc_adam}")

    # Aggregate
    sgd_path  = OUT_DIR / "partial_sgd.json"
    adam_path = OUT_DIR / "partial_adam.json"
    final = aggregate_and_report(sgd_path, adam_path)

    # Task 2: A6 ablation if G3 confirmed
    delta_adam = final.get("delta_adam_fs_minus_mf")
    if delta_adam is not None and delta_adam > 5.0:
        print(f"\n[TASK 2] G3 confirmed (Δ_adam={delta_adam:.2f} > 5). Running A6 ablation...")
        try:
            a6_res = run_a6_ablation()
            print(f"[TASK 2] Done.")
        except Exception as e:
            print(f"[TASK 2] FAILED: {e}\n{traceback.format_exc()}")
    else:
        print(f"\n[TASK 2] Skipping A6 (Δ_adam={delta_adam}; need >5 PPL for Task 2)")

    # Print summary table
    print("\n" + "="*70)
    print("BATCH 12 FINAL SUMMARY")
    print("="*70)
    for cell in ["FS-SGD", "MF-SGD", "FS-ADAM", "MF-ADAM"]:
        s = final["summary"].get(cell, {})
        mean_v = s.get("mean")
        std_v  = s.get("std")
        n_v    = s.get("n", 0)
        if mean_v is not None:
            print(f"  {cell:10s}: {mean_v:.2f} ± {std_v:.2f} PPL  (n={n_v})")
        else:
            print(f"  {cell:10s}: N/A  (n={n_v})")
    print(f"\n  Δ(FS-MF) Adam = {final.get('delta_adam_fs_minus_mf', 'N/A'):.2f} PPL")
    print(f"  Δ(FS-MF) SGD  = {final.get('delta_sgd_fs_minus_mf',  'N/A'):.2f} PPL")
    print(f"\n  G3 Adam: {final['g3_confirmed_adam']}")
    print(f"  G3 SGD:  {final['g3_confirmed_sgd']}")
    print(f"  G6 cross-optim: {final['g6_cross_optim']}")
    print(f"\nResults: {OUT_DIR / 'stage_b_results_5seeds.json'}")

if __name__ == "__main__":
    main()
