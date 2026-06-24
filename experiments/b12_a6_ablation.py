#!/usr/bin/env python3
"""
Batch 12 Task 2 — A6 Random Pressure Ablation on LSTM/WikiText-2
3 seeds (42, 123, 7), GPU 1 only
Compare A6 (random P_t) vs MF-Adam (already done in Task 1)
"""
import sys, os, json, time, math, warnings, atexit, signal, traceback
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")

BASE = Path("./experiments")
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))

OUT_DIR  = Path("./experiments/ablation/a6_random_pressure/lstm_wt2")
HF_CACHE = "./datasets/hf_cache"

# Match Task 1 hypers exactly
SEEDS     = [42, 123, 7]
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

import torch
DEVICE = torch.device("cuda:1")

def js(obj):
    if isinstance(obj, dict):           return {k: js(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [js(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return float(obj.item()) if obj.numel()==1 else obj.tolist()
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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _qr_init(n, r, seed=None):
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    A = torch.randn(n, r, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.float()

# ─── Load data ───────────────────────────────────────────────────────────────
def load_data():
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=HF_CACHE)
    train_text = "\n".join(ds["train"]["text"])
    val_text   = "\n".join(ds["validation"]["text"])
    words = train_text.split()
    counter = Counter(words)
    vocab_words = ["<unk>", "<eos>"] + [w for w,_ in counter.most_common(VOCAB_SIZE_TARGET-2)]
    w2i = {w:i for i,w in enumerate(vocab_words)}
    vocab_size = len(vocab_words)

    def text_to_ids(text):
        return torch.tensor([w2i.get(w,0) for w in text.split()], dtype=torch.long)

    def batchify(data, bsz):
        nb = data.size(0) // bsz
        return data[:nb*bsz].view(bsz, -1).t().contiguous().to(DEVICE)

    train_data = batchify(text_to_ids(train_text), BATCH_SZ)
    val_data   = batchify(text_to_ids(val_text), 1)
    print(f"WikiText-2: vocab={vocab_size}, train_tokens={text_to_ids(train_text).numel()}")
    return train_data, val_data, vocab_size

def get_batch(source, i):
    sl = min(SEQ_LEN, len(source) - 1 - i)
    return source[i:i+sl], source[i+1:i+1+sl].reshape(-1)

# ─── A6 optimizer with random pressure ───────────────────────────────────────
def build_a6_optimizer(model, lr, total_steps):
    """Build optimizer that replaces P_t with random symmetric matrix of equal norm."""
    import types
    from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig, ManifoldFlowOptimizer
    from manifoldflow.spd_ops import sym, symlogm, affine_invariant_step, spectral_clip, fp32_eigh
    from manifoldflow.retraction import qr_retract, procrustes_align
    from manifoldflow.tangent import decompose_tangent_normal, project_tangent
    from manifoldflow.manifoldflow_optimizer import _stiefel_sgd_step

    stiefel_params = [p for n,p in model.named_parameters() if 'proj_Q' in n and p.requires_grad]
    other_params   = [p for n,p in model.named_parameters() if 'proj_Q' not in n and p.requires_grad]

    cfg = ManifoldFlowConfig(rho_geo=RHO_GEO, beta_P=0.95, lambda_S=LAMBDA_S, K_geo=K_GEO)
    opt_mf = ManifoldFlowOptimizer(stiefel_params, base_optim='adam', lr=lr,
                                    betas=(0.9,0.999), mf_config=cfg, total_steps=total_steps)
    opt_base = torch.optim.Adam(other_params, lr=lr) if other_params else None

    def _a6_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        cfg_inner = self.mf_config
        eps = 1e-8
        for group in self.param_groups:
            lr_g = group["lr"]
            momentum = group["momentum"]
            gamma_t = cfg_inner.rho_geo * lr_g
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
                    state["m"] = torch.zeros_like(Q)
                    state["v"] = torch.zeros_like(Q)
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
                M_P_new = cfg_inner.beta_P * M_P_aligned + (1.0 - cfg_inner.beta_P) * P_normalized
                state["M_P"] = M_P_new

                warmup_done = t >= self._warmup_steps()
                do_geo = (gamma_t > 0.0) and warmup_done and (t % cfg_inner.K_geo == 0)

                if do_geo:
                    P_norm = P_t.norm() + eps
                    M_prev_norm = M_P_prev.norm() + eps
                    c_t = (P_t * M_P_prev).sum() / (P_norm * M_prev_norm)
                    G_nor_norm = (Q @ P_t).norm() + eps
                    G_tan_norm = G_tan.norm() + eps
                    r_t = G_nor_norm / G_tan_norm
                    log_r_t = torch.log(r_t)
                    a_t_c = torch.sigmoid(torch.tensor(cfg_inner.alpha_c * (c_t.item() - cfg_inner.tau_c), dtype=Q.dtype, device=Q.device))
                    a_t_r = torch.sigmoid(cfg_inner.alpha_r * (log_r_t - cfg_inner.tau_r))
                    a_t = (a_t_c * a_t_r).item()
                    H_t = sym(M_P_new) + cfg_inner.lambda_S * symlogm(S)
                    S_raw = affine_invariant_step(S, H_t, gamma_t * a_t)
                    state["S"] = spectral_clip(S_raw, cfg_inner.lambda_min, cfg_inner.lambda_max)

                Q.data.copy_(Q_new)
                state["Q_prev"] = Q_new.clone()
                state["step"] = t + 1

                if self.log_pressure:
                    eigvals, _ = fp32_eigh(state["S"])
                    self._pressure_log[id(Q)] = {
                        "P_norm": P_t_real.norm().item(),
                        "lambda_min": eigvals.min().item(),
                        "lambda_max": eigvals.max().item(),
                        "step": t,
                    }
        return loss

    opt_mf.step = types.MethodType(_a6_step, opt_mf)
    return opt_mf, opt_base

# ─── Model ───────────────────────────────────────────────────────────────────
def make_lstm_model(vocab_size, seed):
    import torch.nn as nn, torch.nn.functional as F
    from manifoldflow.spd_ops import matrix_sqrt, sym

    class LSTMLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, EMBED_DIM)
            self.lstm  = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=False)
            # Projection: hidden→vocab (out>in → n=vocab, r=hidden, no transpose)
            self.proj_Q    = nn.Parameter(_qr_init(vocab_size, HIDDEN_DIM, seed))
            self._sqrtS    = torch.eye(HIDDEN_DIM, device=DEVICE)
            self.proj_bias = nn.Parameter(torch.zeros(vocab_size, device=DEVICE))
            nn.init.uniform_(self.embed.weight, -0.1, 0.1)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            # W = proj_Q @ sqrtS (no transpose since vocab>hidden)
            sqrtS = self._sqrtS.to(self.proj_Q.device, self.proj_Q.dtype)
            W = self.proj_Q @ sqrtS
            logits = F.linear(out.view(-1, out.size(-1)), W, self.proj_bias)
            return logits, hidden

        def update_sqrtS(self, S):
            with torch.no_grad():
                self._sqrtS = matrix_sqrt(sym(S)).detach()

    return LSTMLMModel().to(DEVICE)

def eval_ppl(model, val_data):
    import torch.nn.functional as F
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for i in range(0, val_data.size(0)-1, SEQ_LEN):
            sl = min(SEQ_LEN, val_data.size(0) - 1 - i)
            x  = val_data[i:i+sl]
            y  = val_data[i+1:i+1+sl].reshape(-1)
            logits, _ = model(x)
            total_loss   += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
            total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)

# ─── Run A6 for one seed ──────────────────────────────────────────────────────
def run_a6_seed(seed, train_data, val_data, vocab_size):
    import torch.nn.functional as F
    set_seed(seed)
    n_batches   = (train_data.size(0) - 1) // SEQ_LEN
    total_steps = N_EPOCHS * n_batches
    model = make_lstm_model(vocab_size, seed)
    opt_mf, opt_base = build_a6_optimizer(model, LR_ADAM, total_steps)

    best_ppl = float('inf')
    epoch_ppls = []
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for i in range(0, train_data.size(0)-1, SEQ_LEN):
            sl = min(SEQ_LEN, train_data.size(0) - 1 - i)
            x  = train_data[i:i+sl]
            y  = train_data[i+1:i+1+sl].reshape(-1)
            opt_mf.zero_grad()
            if opt_base: opt_base.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt_mf.step()
            if opt_base: opt_base.step()
            # Update sqrtS from optimizer state
            Q = model.proj_Q
            if Q in opt_mf.state and 'S' in opt_mf.state[Q]:
                model.update_sqrtS(opt_mf.state[Q]['S'])

        ppl = eval_ppl(model, val_data)
        epoch_ppls.append(ppl)
        if ppl < best_ppl:
            best_ppl = ppl
        print(f"  [A6 seed={seed}] ep{epoch+1}/{N_EPOCHS} ppl={ppl:.2f}")

    elapsed = time.time() - t0
    # Free memory
    del model, opt_mf, opt_base
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return {
        "seed":       seed,
        "method":     "a6_random_pressure",
        "best_ppl":   best_ppl,
        "epoch_ppls": epoch_ppls,
        "elapsed":    elapsed,
    }

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"A6 Ablation — GPU: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        try:
            existing = json.load(open(results_path))
        except Exception:
            existing = {}
    else:
        existing = {}

    # A6 results dict
    a6_results = existing.get("a6", {})
    _atexit_data = {"a6": a6_results}

    def _atexit_fn():
        flush_json(OUT_DIR / "results.json", _atexit_data)
        print(f"[atexit] flushed results")
    atexit.register(_atexit_fn)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))

    print(f"\nLoading WikiText-2...")
    train_data, val_data, vocab_size = load_data()

    for seed in SEEDS:
        seed_key = str(seed)
        existing_a6 = a6_results.get(seed_key, {})
        if isinstance(existing_a6, dict) and "best_ppl" in existing_a6:
            print(f"  [A6 seed={seed}] SKIP (already done PPL={existing_a6['best_ppl']:.2f})")
            continue

        print(f"\n  [A6 seed={seed}] Running random pressure ablation...")
        try:
            res = run_a6_seed(seed, train_data, val_data, vocab_size)
            a6_results[seed_key] = res
            _atexit_data["a6"] = a6_results
            flush_json(results_path, _atexit_data)
            print(f"  [A6 seed={seed}] DONE PPL={res['best_ppl']:.2f}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [A6 seed={seed}] FAILED: {e}\n{tb}")
            a6_results[seed_key] = {"status": "failed", "error": str(e)}
            _atexit_data["a6"] = a6_results
            flush_json(results_path, _atexit_data)

    # Compute summary
    a6_ppls  = [v["best_ppl"] for v in a6_results.values() if isinstance(v,dict) and "best_ppl" in v]

    # Load MF-Adam results from Task 1 for comparison
    task1_path = Path("./experiments/method_1/lstm_wt2_proj/partial_adam.json")
    mf_ppls = []
    if task1_path.exists():
        t1 = json.load(open(task1_path))
        mf_cell = t1.get("MF-ADAM", {})
        # Use same seeds as A6
        for seed in SEEDS:
            v = mf_cell.get(str(seed), {})
            if isinstance(v, dict) and "best_ppl" in v:
                mf_ppls.append(v["best_ppl"])

    summary = {
        "seeds_used":     SEEDS,
        "a6_ppls":        a6_ppls,
        "mf_adam_ppls":   mf_ppls,
        "a6_mean":        float(np.mean(a6_ppls))  if a6_ppls  else None,
        "a6_std":         float(np.std(a6_ppls))   if a6_ppls  else None,
        "mf_adam_mean":   float(np.mean(mf_ppls))  if mf_ppls  else None,
        "mf_adam_std":    float(np.std(mf_ppls))   if mf_ppls  else None,
        "delta_a6_vs_mf": (float(np.mean(a6_ppls) - np.mean(mf_ppls)) if (a6_ppls and mf_ppls) else None),
    }

    final = {"a6": a6_results, "summary": summary}
    flush_json(results_path, final)

    print(f"\n{'='*60}")
    print("A6 ABLATION SUMMARY")
    print(f"{'='*60}")
    if summary["a6_mean"]:
        print(f"  A6-random:  {summary['a6_mean']:.2f} ± {summary['a6_std']:.2f} PPL (n={len(a6_ppls)})")
    if summary["mf_adam_mean"]:
        print(f"  MF-Adam:    {summary['mf_adam_mean']:.2f} ± {summary['mf_adam_std']:.2f} PPL (n={len(mf_ppls)})")
    if summary["delta_a6_vs_mf"] is not None:
        d = summary["delta_a6_vs_mf"]
        verdict = "MECHANISM EXISTS (A6 worse than MF)" if d > 2.0 else ("MECHANISM MARGINAL" if d > 0 else "A6 ≈ MF (pressure-only)")
        print(f"  Δ(A6 - MF): {d:+.2f} PPL → {verdict}")
    print(f"\n  Results: {results_path}")
