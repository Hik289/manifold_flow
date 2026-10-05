#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

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
RHO_GEO   = 0.01
LAMBDA_S  = 0.001
K_GEO     = 10
LR_ADAM   = 0.003
LR_SGD    = 0.01

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

def _qr_init(n, r, seed=None):
    import torch
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    A = torch.randn(n, r, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.float()

def build_stiefel_linear(in_dim, out_dim, seed, mode, device):

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

def make_lstm_model(vocab_size, embed_dim, hidden_dim, mode, seed, device):
    import torch, torch.nn as nn, torch.nn.functional as F
    from manifoldflow.spd_ops import matrix_sqrt, sym

    class LSTMLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=False)
            if vocab_size <= hidden_dim:
                n, r = hidden_dim, vocab_size; self.proj_transpose = True
            else:
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
    else:
        opt_mf = ManifoldFlowOptimizer(stiefel_params, base_optim='sgd', lr=lr,
                                        momentum=0.9, mf_config=cfg, total_steps=total_steps)
        opt_base = __import__('torch').optim.SGD(other_params, lr=lr, momentum=0.9) if other_params else None
    return opt_mf, opt_base

_DATASET_CACHE = {}

def load_wt2_data(device):

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

def get_trace_from_opt(model, opt_mf, prev_mp=None):

    import torch
    Q = model.proj_Q
    state = opt_mf.state.get(Q, {})
    plog  = opt_mf._pressure_log.get(id(Q), {})

    P_norm  = plog.get("P_norm", float('nan'))
    lam_min = plog.get("lambda_min", float('nan'))
    lam_max = plog.get("lambda_max", float('nan'))

    M_P_curr = state.get("M_P", None)
    cos_val  = float('nan')
    if M_P_curr is not None and prev_mp is not None:
        with torch.no_grad():
            a = M_P_curr.float().flatten()
            b = prev_mp.to(a.device).float().flatten()
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

def run_cell(optim_type, mode, seed, device, train_data, val_data, vocab_size,
             a6_random_pressure=False):

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

    if a6_random_pressure and mode == 'mf':
        _orig_step = opt_mf.step.__func__
        import types
        from manifoldflow.spd_ops import sym
        from manifoldflow.tangent import decompose_tangent_normal
        from manifoldflow.retraction import qr_retract, procrustes_align
        from manifoldflow.spd_ops import symlogm, affine_invariant_step, spectral_clip, fp32_eigh

        def _a6_step(self, closure=None):

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
                            "P_norm": P_t_real.norm().item(),
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

def worker_main(optim_type, gpu_idx):

    import torch
    device = torch.device(f"cuda:{gpu_idx}")
    print(f"\n{'='*60}")
    print(f"WORKER: optim={optim_type.upper()}, GPU={gpu_idx}")
    print(f"{'='*60}")

    train_data, val_data, vocab_size, w2i = load_wt2_data(device)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

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

def run_a6_ablation():

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

def aggregate_and_report(sgd_path, adam_path):

    sgd_res  = json.load(open(sgd_path))  if sgd_path.exists()  else {}
    adam_res = json.load(open(adam_path)) if adam_path.exists() else {}
    all_cells = {**sgd_res, **adam_res}

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

    delta_adam = delta_sgd = None
    fs_adam = summary.get("FS-ADAM", {}).get("mean")
    mf_adam = summary.get("MF-ADAM", {}).get("mean")
    fs_sgd  = summary.get("FS-SGD",  {}).get("mean")
    mf_sgd  = summary.get("MF-SGD",  {}).get("mean")
    if fs_adam is not None and mf_adam is not None:
        delta_adam = fs_adam - mf_adam
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

def legacy_main():
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Batch 12 — LSTM/WikiText-2 5-seed confirm")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {OUT_DIR}")
    print("="*70)

    script = Path(__file__).resolve()
    env = os.environ.copy()

    t_start = time.time()

    proc_sgd = subprocess.Popen(
        [sys.executable, str(script), "--legacy", "--worker", "--optim-type", "sgd", "--gpu", "1"],
        env={**env, "CUDA_VISIBLE_DEVICES": "0,1,2"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    proc_adam = subprocess.Popen(
        [sys.executable, str(script), "--legacy", "--worker", "--optim-type", "adam", "--gpu", "2"],
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

    sgd_path  = OUT_DIR / "partial_sgd.json"
    adam_path = OUT_DIR / "partial_adam.json"
    final = aggregate_and_report(sgd_path, adam_path)

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

import argparse
import array
import hashlib
import itertools
import math
import os
import platform
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from manifoldflow.parametrization import PlateauStopper, SpectrumLinear, make_optimizers, resolve_device, set_seed as _current_set_seed, write_json
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig
DATASET_CONFIGS = {'wikitext2': 'wikitext-2-raw-v1', 'wikitext103': 'wikitext-103-raw-v1'}

def current_dataset_name(value: str) -> str:
    normalized = value.lower().replace('-', '').replace('_', '')
    aliases = {'wt2': 'wikitext2', 'wt103': 'wikitext103'}
    normalized = aliases.get(normalized, normalized)
    if normalized not in DATASET_CONFIGS:
        raise argparse.ArgumentTypeError('choose wikitext2 or wikitext103')
    return normalized

def current_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--architectures', '--architecture', nargs='+', type=str.lower, choices=('lstm', 'gru', 'transformer'), default=['lstm'])
    result.add_argument('--datasets', '--dataset', nargs='+', type=current_dataset_name, default=['wikitext2'])
    result.add_argument('--hidden-sizes', '--hidden-size', nargs='+', type=int, choices=(128, 256), default=[128])
    result.add_argument('--optimizers', '--optimizer', nargs='+', type=str.lower, choices=('adam', 'sgd'), default=['adam', 'sgd'])
    result.add_argument('--methods', nargs='+', type=str.lower, choices=('dense', 'fs', 'scalar', 'diagonal', 'mf'), default=['fs', 'mf'])
    result.add_argument('--spectrum-suite', action='store_true')
    result.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7, 2024, 2025])
    result.add_argument('--max-epochs', '--epochs', type=int, default=1000)
    result.add_argument('--min-epochs', type=int, default=300)
    result.add_argument('--patience', type=int, default=100)
    result.add_argument('--min-delta', type=float, default=0.0)
    result.add_argument('--batch-size', type=int, default=32)
    result.add_argument('--eval-batch-size', type=int, default=1)
    result.add_argument('--seq-len', type=int, default=35)
    result.add_argument('--embedding-size', type=int, default=128)
    result.add_argument('--num-layers', type=int, default=1)
    result.add_argument('--dropout', type=float)
    result.add_argument('--transformer-layers', type=int, default=2)
    result.add_argument('--transformer-heads', type=int, default=4)
    result.add_argument('--ffn-dim', type=int, default=256)
    result.add_argument('--transformer-dropout', type=float)
    result.add_argument('--hidden-state', choices=('detach', 'reset'), default='detach')
    result.add_argument('--line-eos', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--max-vocab-size', type=int, default=10000)
    result.add_argument('--lr', type=float)
    result.add_argument('--lr-adam', '--adam-lr', type=float, default=0.003)
    result.add_argument('--lr-sgd', '--sgd-lr', type=float, default=0.01)
    result.add_argument('--momentum', type=float, default=0.9)
    result.add_argument('--weight-decay', type=float, default=0.0)
    result.add_argument('--gradient-clip', type=float, default=0.5)
    result.add_argument('--alpha', type=float, default=0.25)
    result.add_argument('--beta', type=float, default=4.0)
    result.add_argument('--rho-geo', type=float, default=0.01)
    result.add_argument('--beta-p', type=float, default=0.95)
    result.add_argument('--lambda-s', type=float, default=0.001)
    result.add_argument('--k-geo', type=int, default=10)
    result.add_argument('--warmup-frac', type=float, default=0.05)
    result.add_argument('--tau-c', type=float, default=0.1)
    result.add_argument('--tau-r', type=float, default=0.0)
    result.add_argument('--alpha-c', type=float, default=5.0)
    result.add_argument('--alpha-r', type=float, default=2.0)
    result.add_argument('--data-dir', type=Path)
    result.add_argument('--hf-cache', '--cache-dir', type=Path)
    result.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/sequence_models')
    result.add_argument('--device', default='auto')
    result.add_argument('--num-threads', type=int)
    result.add_argument('--allow-nondeterministic', action='store_true')
    result.add_argument('--save-models', action='store_true')
    return result

def current_validate_args(args: argparse.Namespace) -> None:
    for name in ('max_epochs', 'min_epochs', 'patience', 'batch_size', 'eval_batch_size', 'seq_len', 'embedding_size', 'num_layers', 'k_geo', 'transformer_layers', 'transformer_heads', 'ffn_dim'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.max_epochs < args.min_epochs:
        raise ValueError('max_epochs must be at least min_epochs')
    if args.min_delta < 0 or not math.isfinite(args.min_delta):
        raise ValueError('min_delta must be finite and nonnegative')
    if not 2 <= args.max_vocab_size <= 10000:
        raise ValueError('max_vocab_size must be between 2 and 10000')
    for name in ('dropout', 'transformer_dropout'):
        value = getattr(args, name)
        if value is not None and (not 0 <= value < 1):
            raise ValueError(f'{name} must be in [0, 1)')
    if 'transformer' in args.architectures and any((size % args.transformer_heads != 0 for size in args.hidden_sizes)):
        raise ValueError('Transformer hidden sizes must be divisible by transformer_heads')
    if not 0 < args.alpha <= 1 <= args.beta or not math.isfinite(args.beta):
        raise ValueError('spectral bounds must satisfy 0 < alpha <= 1 <= beta')
    for name in ('lr_adam', 'lr_sgd', 'gradient_clip'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('lr must be finite and positive')
    if not 0 <= args.momentum < 1 or not 0 <= args.beta_p < 1:
        raise ValueError('momentum and beta_p must be in [0, 1)')
    for name in ('weight_decay', 'rho_geo', 'lambda_s'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 <= args.warmup_frac <= 1:
        raise ValueError('warmup_frac must be in [0, 1]')
    for name in ('tau_c', 'tau_r', 'alpha_c', 'alpha_r'):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f'{name} must be finite')
    if args.num_threads is not None and args.num_threads < 1:
        raise ValueError('num_threads must be positive')
    if any((seed < 0 or seed >= 2 ** 32 for seed in args.seeds)):
        raise ValueError('seeds must be in [0, 2**32)')
    for name in ('architectures', 'datasets', 'hidden_sizes', 'optimizers', 'methods', 'seeds'):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f'{name} must not contain duplicates')
    if args.spectrum_suite:
        args.methods = ['dense', 'fs', 'scalar', 'diagonal', 'mf']
    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()
        if len(args.datasets) > 1 and (args.data_dir / 'train.txt').exists():
            raise ValueError('multiple datasets require data_dir/wikitext2 and data_dir/wikitext103')
    if args.hf_cache is not None:
        args.hf_cache = args.hf_cache.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

def current_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def current_local_records(path: Path) -> Iterable[str]:
    with path.open('r', encoding='utf-8') as handle:
        yield from handle

def current_tokens(records: Iterable[str], line_eos: bool=True) -> Iterable[str]:
    for record in records:
        if not isinstance(record, str):
            raise ValueError('all corpus records must contain text strings')
        for line in record.splitlines() or ['']:
            yield from line.split()
            if line_eos:
                yield '<eos>'

def current_load_corpus(name: str, args: argparse.Namespace) -> Dict[str, Any]:
    if args.data_dir is not None:
        directory = args.data_dir
        if not (directory / 'train.txt').is_file():
            directory = directory / name
        paths = {'train': directory / 'train.txt', 'validation': directory / 'valid.txt', 'test': directory / 'test.txt'}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError('missing local corpus files: ' + ', '.join(missing))
        records = {split: lambda path=path: current_local_records(path) for (split, path) in paths.items()}
        source = {'kind': 'local_utf8_text', 'dataset_label': name, 'files': {split: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': current_file_hash(path)} for (split, path) in paths.items()}}
    else:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        try:
            import datasets
            from datasets import DownloadConfig, load_dataset
        except ImportError as error:
            raise RuntimeError('install datasets or provide --data-dir with local text files') from error
        datasets.config.HF_DATASETS_OFFLINE = True
        from huggingface_hub import constants as hub_constants
        hub_constants.HF_HUB_OFFLINE = True
        try:
            dataset = load_dataset('wikitext', DATASET_CONFIGS[name], cache_dir=str(args.hf_cache) if args.hf_cache is not None else None, download_config=DownloadConfig(local_files_only=True), download_mode='reuse_dataset_if_exists')
        except Exception as error:
            raise RuntimeError(f'{name} is unavailable in the offline datasets cache; provide --data-dir or point --hf-cache at an existing cache') from error
        for split in ('train', 'validation', 'test'):
            if split not in dataset or 'text' not in dataset[split].column_names:
                raise ValueError(f'cached {name} requires a text column in {split}')
        records = {split: lambda split=split: (row['text'] for row in dataset[split]) for split in ('train', 'validation', 'test')}
        source = {'kind': 'huggingface_cached_dataset', 'offline_only': True, 'dataset': 'wikitext', 'dataset_config': DATASET_CONFIGS[name], 'datasets_version': datasets.__version__, 'cache_dir': str(args.hf_cache) if args.hf_cache is not None else None, 'splits': {split: {'rows': len(dataset[split]), 'fingerprint': getattr(dataset[split], '_fingerprint', None), 'cache_files': dataset[split].cache_files} for split in records}}
    counts = Counter(current_tokens(records['train'](), args.line_eos))
    ordinary = sorted((token for token in counts if token not in {'<unk>', '<eos>'}), key=lambda token: (-counts[token], token))
    vocabulary = ['<unk>', '<eos>'] + ordinary[:args.max_vocab_size - 2]
    indices = {token: index for (index, token) in enumerate(vocabulary)}
    encoded = {}
    split_metadata = {}
    for (split, factory) in records.items():
        buffer = array.array('q')
        unknown = 0
        for token in current_tokens(factory(), args.line_eos):
            index = indices.get(token, 0)
            buffer.append(index)
            unknown += int(index == 0)
        if not buffer:
            raise ValueError(f'{split} has no tokens')
        encoded[split] = torch.frombuffer(buffer, dtype=torch.int64)
        split_metadata[split] = {'tokens': len(buffer), 'unknown_id_tokens': unknown, 'unknown_fraction': unknown / len(buffer)}
    vocabulary_hash = hashlib.sha256('\n'.join(vocabulary).encode('utf-8')).hexdigest()
    metadata = {'source': source, 'tokenization': {'scheme': 'case_sensitive_whitespace', 'encoding': 'utf-8', 'line_boundary': 'one <eos> per source line including blank lines' if args.line_eos else 'whitespace only; no explicit line-boundary token', 'vocabulary_split': 'train', 'frequency_ties': 'lexicographic', 'unknown_token': '<unk>', 'unknown_id': 0, 'eos_id': 1}, 'vocabulary': {'size': len(vocabulary), 'maximum_size': args.max_vocab_size, 'sha256': vocabulary_hash, 'tokens': vocabulary}, 'splits': split_metadata}
    return {'encoded': encoded, 'metadata': metadata}

def current_batchify(stream: torch.Tensor, batch_size: int) -> torch.Tensor:
    columns = stream.numel() // batch_size
    if columns < 2:
        raise ValueError(f'at least {2 * batch_size} tokens are required for batch_size={batch_size}')
    return stream[:columns * batch_size].view(batch_size, columns).transpose(0, 1)

def current_get_batch(source: torch.Tensor, offset: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    length = min(seq_len, source.size(0) - 1 - offset)
    inputs = source[offset:offset + length].to(device)
    targets = source[offset + 1:offset + length + 1].reshape(-1).to(device)
    return (inputs, targets)

def current_detach_hidden(hidden: Any) -> Any:
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple((component.detach() for component in hidden))
    return hidden.detach()

class RecurrentLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, architecture: str, hidden_size: int, method: str, args: argparse.Namespace):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, args.embedding_size)
        recurrent = nn.LSTM if architecture == 'lstm' else nn.GRU
        dropout = args.dropout if args.dropout is not None else 0.0
        self.recurrent = recurrent(args.embedding_size, hidden_size, num_layers=args.num_layers, dropout=dropout if args.num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.projection = SpectrumLinear(hidden_size, vocab_size, method, alpha=args.alpha, beta=args.beta)

    def forward(self, inputs: torch.Tensor, hidden: Any=None) -> Tuple[torch.Tensor, Any]:
        embedding = self.dropout(self.embedding(inputs))
        (output, hidden) = self.recurrent(embedding, hidden)
        logits = self.projection(self.dropout(output).reshape(-1, output.size(-1)))
        return (logits, hidden)

class TransformerBlock(nn.Module):

    def __init__(self, hidden_size: int, method: str, dropout: float, args: argparse.Namespace):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, args.transformer_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn1 = SpectrumLinear(hidden_size, args.ffn_dim, method, alpha=args.alpha, beta=args.beta)
        self.ffn2 = SpectrumLinear(args.ffn_dim, hidden_size, method, alpha=args.alpha, beta=args.beta)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        (attention, _) = self.attention(inputs, inputs, inputs, attn_mask=mask, need_weights=False)
        hidden = self.norm1(inputs + self.dropout(attention))
        feed_forward = self.ffn2(self.dropout(F.gelu(self.ffn1(hidden))))
        return self.norm2(hidden + self.dropout(feed_forward))

class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, method: str, args: argparse.Namespace):
        super().__init__()
        dropout = args.transformer_dropout
        if dropout is None:
            dropout = args.dropout if args.dropout is not None else 0.1
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(max(512, args.seq_len), hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, method, dropout, args) for _ in range(args.transformer_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, hidden: Any=None) -> Tuple[torch.Tensor, Any]:
        positions = torch.arange(inputs.size(0), device=inputs.device).unsqueeze(1)
        output = self.dropout(self.embedding(inputs) + self.position(positions))
        mask = torch.triu(torch.ones(inputs.size(0), inputs.size(0), dtype=torch.bool, device=inputs.device), diagonal=1)
        for block in self.blocks:
            output = block(output, mask)
        logits = self.projection(self.norm(output)).reshape(-1, self.projection.out_features)
        return (logits, None)

def current_perplexity(loss: float) -> Optional[float]:
    if not math.isfinite(loss) or loss > math.log(sys.float_info.max):
        return None
    return math.exp(loss)

def current_evaluate(model: nn.Module, source: torch.Tensor, seq_len: int, device: torch.device, hidden_state: str='detach') -> Dict[str, Any]:
    model.eval()
    hidden = None
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for offset in range(0, source.size(0) - 1, seq_len):
            (inputs, targets) = current_get_batch(source, offset, seq_len, device)
            (logits, hidden) = model(inputs, current_detach_hidden(hidden) if hidden_state == 'detach' else None)
            loss = F.cross_entropy(logits, targets, reduction='sum').item()
            if not math.isfinite(loss):
                raise FloatingPointError('nonfinite evaluation cross entropy')
            total_loss += loss
            total_tokens += targets.numel()
    average = total_loss / total_tokens
    return {'loss': average, 'ppl': current_perplexity(average), 'tokens': total_tokens}

def current_synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()

def current_hardware(device: torch.device) -> Dict[str, Any]:
    information = {'device': str(device), 'device_type': device.type, 'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor() or platform.machine(), 'python_version': platform.python_version(), 'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(), 'torch_num_threads': torch.get_num_threads(), 'dtype': 'float32', 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        information.update({'accelerator_name': properties.name, 'accelerator_total_memory_bytes': properties.total_memory, 'compute_capability': [properties.major, properties.minor]})
    elif device.type == 'mps':
        information['accelerator_name'] = 'Apple Metal Performance Shaders'
    return information

def current_tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode('ascii'))
    digest.update(str(cpu.dtype).encode('ascii'))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()

def current_initialization_metadata(model: nn.Module) -> Dict[str, Any]:
    layers = [(name, layer) for (name, layer) in model.named_modules() if isinstance(layer, SpectrumLinear)]
    excluded = {id(parameter) for (_, layer) in layers for parameter in layer.parameters()}
    ordinary = {}
    for (name, parameter) in model.named_parameters():
        if id(parameter) not in excluded:
            ordinary[name] = current_tensor_hash(parameter)
    return {'ordinary_parameter_sha256': ordinary, 'spectrum_layers': {name: {'realized_weight_sha256': current_tensor_hash(layer.effective_weight()), 'bias_sha256': current_tensor_hash(layer.bias) if layer.bias is not None else None} for (name, layer) in layers}}

def current_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for (key, value) in vars(args).items()}

def current_manifold_config(args: argparse.Namespace) -> ManifoldFlowConfig:
    return ManifoldFlowConfig(rho_geo=args.rho_geo, beta_P=args.beta_p, lambda_S=args.lambda_s, K_geo=args.k_geo, tau_c=args.tau_c, tau_r=args.tau_r, alpha_c=args.alpha_c, alpha_r=args.alpha_r, lambda_min=args.alpha, lambda_max=args.beta, warmup_frac=args.warmup_frac)

def current_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for (name, tensor) in model.state_dict().items()}

def current_train_run(args: argparse.Namespace, corpus: Dict[str, Any], architecture: str, dataset: str, hidden_size: int, optimizer: str, method: str, seed: int, device: torch.device, output_path: Path) -> Dict[str, Any]:
    _current_set_seed(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    vocab_size = corpus['metadata']['vocabulary']['size']
    model = (TransformerLanguageModel(vocab_size, hidden_size, method, args) if architecture == 'transformer' else RecurrentLanguageModel(vocab_size, architecture, hidden_size, method, args)).to(device)
    initial = current_initialization_metadata(model)
    train_data = current_batchify(corpus['encoded']['train'], args.batch_size)
    validation_data = current_batchify(corpus['encoded']['validation'], args.eval_batch_size)
    test_data = current_batchify(corpus['encoded']['test'], args.eval_batch_size)
    steps_per_epoch = math.ceil((train_data.size(0) - 1) / args.seq_len)
    learning_rate = args.lr if args.lr is not None else args.lr_adam if optimizer == 'adam' else args.lr_sgd
    mf_config = current_manifold_config(args)
    bundle = make_optimizers(model, optimizer, learning_rate, steps_per_epoch * args.max_epochs, mf_config, momentum=args.momentum, weight_decay=args.weight_decay)
    stopper = PlateauStopper(args.min_epochs, args.patience, args.min_delta, mode='min')
    run_config = current_configuration(args)
    run_config.update({'architecture': architecture, 'dataset': dataset, 'hidden_size': hidden_size, 'optimizer': optimizer, 'method': method, 'seed': seed, 'learning_rate': learning_rate, 'manifold_flow': asdict(mf_config), 'steps_per_epoch': steps_per_epoch, 'maximum_optimizer_steps': steps_per_epoch * args.max_epochs, 'learning_rate_schedule': 'constant', 'constrained_layers': [name for (name, layer) in model.named_modules() if isinstance(layer, SpectrumLinear)], 'shuffle': False, 'effective_dropout': model.dropout.p, 'effective_embedding_size': hidden_size if architecture == 'transformer' else args.embedding_size, 'hidden_state': args.hidden_state if architecture != 'transformer' else 'none; causal attention within each fixed segment', 'checkpoint_selection': 'minimum validation cross entropy', 'early_stopping_metric': 'validation_cross_entropy', 'test_evaluation': 'once after restoring the best validation checkpoint'})
    result = {'status': 'running', 'protocol': 'new_sequence_runner', 'config': run_config, 'data': corpus['metadata'], 'code_source': {'sequence_experiment': {'path': str(Path(__file__).resolve()), 'sha256': current_file_hash(Path(__file__))}, 'parametrization': {'path': str(ROOT / 'src/manifoldflow/parametrization.py'), 'sha256': current_file_hash(ROOT / 'src/manifoldflow/parametrization.py')}, 'manifoldflow_optimizer': {'path': str(ROOT / 'src/manifoldflow/manifoldflow_optimizer.py'), 'sha256': current_file_hash(ROOT / 'src/manifoldflow/manifoldflow_optimizer.py')}}, 'actual_hardware': current_hardware(device), 'initialization': initial, 'history': [], 'best_epoch': None, 'best_val_ppl': None, 'final_val_ppl': None, 'test_ppl': None, 'batchification': {split: {'batch_size': data.size(1), 'retained_tokens': data.numel(), 'discarded_remainder_tokens': corpus['encoded'][split].numel() - data.numel(), 'prediction_tokens_per_pass': (data.size(0) - 1) * data.size(1)} for (split, data) in (('train', train_data), ('validation', validation_data), ('test', test_data))}}
    write_json(output_path, result)
    _current_set_seed(seed)
    best_state = None
    best_loss = math.inf
    started = time.perf_counter()
    try:
        for epoch in range(1, args.max_epochs + 1):
            current_synchronize(device)
            epoch_started = time.perf_counter()
            model.train()
            hidden = None
            total_loss = 0.0
            total_tokens = 0
            last_gradient_norm = None
            for offset in range(0, train_data.size(0) - 1, args.seq_len):
                (inputs, targets) = current_get_batch(train_data, offset, args.seq_len, device)
                hidden = current_detach_hidden(hidden) if args.hidden_state == 'detach' else None
                bundle.zero_grad()
                (logits, hidden) = model(inputs, hidden)
                loss = F.cross_entropy(logits, targets)
                if not math.isfinite(loss.item()):
                    raise FloatingPointError(f'nonfinite training cross entropy in epoch {epoch}')
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                last_gradient_norm = float(gradient_norm.item())
                bundle.step()
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
            validation = current_evaluate(model, validation_data, args.seq_len, device, args.hidden_state)
            stop = stopper.update(validation['loss'], epoch)
            if validation['loss'] < best_loss:
                best_loss = validation['loss']
                best_state = current_snapshot(model)
                result.update({'best_epoch': epoch, 'best_val_loss': best_loss, 'best_val_ppl': validation['ppl']})
            current_synchronize(device)
            training_loss = total_loss / total_tokens
            entry = {'epoch': epoch, 'train_loss': training_loss, 'train_ppl': current_perplexity(training_loss), 'train_prediction_tokens': total_tokens, 'val_loss': validation['loss'], 'val_ppl': validation['ppl'], 'validation_prediction_tokens': validation['tokens'], 'optimizer_steps': epoch * steps_per_epoch, 'last_unclipped_gradient_norm': last_gradient_norm, 'epoch_seconds': time.perf_counter() - epoch_started, 'diagnostics': bundle.diagnostics()}
            result['history'].append(entry)
            result.update({'completed_epochs': epoch, 'final_epoch': epoch, 'final_val_loss': validation['loss'], 'final_val_ppl': validation['ppl'], 'elapsed_seconds': time.perf_counter() - started})
            write_json(output_path, result)
            print(f"{architecture} {dataset} h={hidden_size} {optimizer} {method} seed={seed} epoch={epoch} validation_ppl={validation['ppl']}", flush=True)
            if stop:
                result['stop_reason'] = 'validation_plateau'
                break
        if best_state is None:
            raise RuntimeError('no finite validation checkpoint was obtained')
        if args.save_models:
            checkpoint_path = output_path.with_suffix('.final.pt')
            torch.save({'state_dict': current_snapshot(model), 'epoch': result['final_epoch'], 'validation_loss': result['final_val_loss'], 'config': run_config}, checkpoint_path)
            result['final_checkpoint_path'] = str(checkpoint_path)
        model.load_state_dict(best_state)
        selected_diagnostics = bundle.diagnostics(include_gradient_metrics=False)
        test = current_evaluate(model, test_data, args.seq_len, device, args.hidden_state)
        current_synchronize(device)
        result.update({'status': 'completed', 'test_epoch': result['best_epoch'], 'test_loss': test['loss'], 'test_ppl': test['ppl'], 'test_prediction_tokens': test['tokens'], 'test_evaluations': 1, 'best_checkpoint_diagnostics': selected_diagnostics, 'elapsed_seconds': time.perf_counter() - started})
        result.setdefault('stop_reason', 'maximum_epochs')
        if args.save_models:
            checkpoint_path = output_path.with_suffix('.best.pt')
            torch.save({'state_dict': best_state, 'epoch': result['best_epoch'], 'validation_loss': result['best_val_loss'], 'config': run_config}, checkpoint_path)
            result['best_checkpoint_path'] = str(checkpoint_path)
        if device.type == 'cuda':
            result['actual_hardware']['peak_allocated_memory_bytes'] = torch.cuda.max_memory_allocated(device)
            result['actual_hardware']['peak_reserved_memory_bytes'] = torch.cuda.max_memory_reserved(device)
        write_json(output_path, result)
    except BaseException as error:
        result.update({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'error': f'{type(error).__name__}: {error}', 'elapsed_seconds': time.perf_counter() - started})
        write_json(output_path, result)
        raise
    return {'path': str(output_path), 'architecture': architecture, 'dataset': dataset, 'hidden_size': hidden_size, 'optimizer': optimizer, 'method': method, 'seed': seed, 'status': result['status'], 'best_epoch': result['best_epoch'], 'final_epoch': result['final_epoch'], 'best_val_ppl': result['best_val_ppl'], 'final_val_ppl': result['final_val_ppl'], 'test_ppl': result['test_ppl'], 'stop_reason': result['stop_reason']}

def current_main() -> None:
    args = current_parser().parse_args()
    current_validate_args(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(not args.allow_nondeterministic)
    torch.backends.cudnn.deterministic = not args.allow_nondeterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    device = torch.device(resolve_device(args.device))
    corpus_data = {name: current_load_corpus(name, args) for name in args.datasets}
    session_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:12]
    args.session_dir = args.output_dir / session_name
    manifest_path = args.session_dir / 'summary.json'
    manifest = {'status': 'running', 'protocol': 'new_sequence_runner', 'config': current_configuration(args), 'actual_hardware': current_hardware(device), 'datasets': {name: corpus['metadata'] for (name, corpus) in corpus_data.items()}, 'runs': [], 'planned_runs': len(args.architectures) * len(args.datasets) * len(args.hidden_sizes) * len(args.optimizers) * len(args.methods) * len(args.seeds)}
    write_json(manifest_path, manifest)
    try:
        for (architecture, name, hidden_size, optimizer, seed, method) in itertools.product(args.architectures, args.datasets, args.hidden_sizes, args.optimizers, args.seeds, args.methods):
            output = args.session_dir / architecture / name / f'h{hidden_size}' / optimizer / f'{method}_seed{seed}.json'
            outcome = current_train_run(args, corpus_data[name], architecture, name, hidden_size, optimizer, method, seed, device, output)
            manifest['runs'].append(outcome)
            write_json(manifest_path, manifest)
        manifest['status'] = 'completed'
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'error': f'{type(error).__name__}: {error}'})
        write_json(manifest_path, manifest)
        raise

def main():
    if "--legacy" in sys.argv[1:]:
        sys.argv.remove("--legacy")
        legacy_main()
    else:
        current_main()


if __name__ == "__main__":
    main()
