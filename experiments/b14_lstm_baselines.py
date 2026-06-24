#!/usr/bin/env python3
"""
Batch 14 — LSTM/WikiText-2 baselines for FATAL reviewer gap
B1: W = Q @ diag(exp(log_s)),  Q ∈ Stiefel (FS optimizer), s > 0
B2: Unconstrained nn.Linear(hidden, vocab) — no Stiefel
B3: Intrinsic Muon — tangent-only Stiefel (no SPD learning)

3 seeds: [42, 123, 7]  ×  3 baselines  ×  Adam lr=0.003
GPU: 2 only
Output: experiments/comparison/lstm_baselines/{B1_qdiag,B2_dense,B3_intrinsic_muon}/results.json
"""
import sys, os, json, time, math, warnings, atexit, signal, traceback
from pathlib import Path
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")

BASE = Path("./experiments")
SRC  = BASE / "src"
sys.path.insert(0, str(SRC))

OUT_BASE  = BASE / "comparison" / "lstm_baselines"
HF_CACHE  = "./datasets/hf_cache"
MF_REF    = BASE / "method_1" / "lstm_wt2_proj" / "stage_b_results_5seeds.json"

SEEDS      = [42, 123, 7]
N_EPOCHS   = 8
BATCH_SZ   = 32
SEQ_LEN    = 35
EMBED_DIM  = 128
HIDDEN_DIM = 128
VOCAB_SIZE_TARGET = 10000
LR         = 0.003

# MF-Adam reference PPLs (seeds 42, 123, 7) from stage_b_results_5seeds.json
MF_ADAM_PPLS = {42: 273.6116260829776, 123: 276.1922541917133, 7: 281.0335268554727}
MF_ADAM_MEAN = float(np.mean(list(MF_ADAM_PPLS.values())))  # 276.946...

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
            return float(obj.item()) if obj.numel() == 1 else obj.tolist()
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


# ─── Newton-Schulz orthogonalization (for Intrinsic Muon) ──────────────────
def newton_schulz_5(G, steps=5, eps=1e-7):
    import torch
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if X.size(-2) > X.size(-1):
        X = X.transpose(-1, -2)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-1, -2)
    return X.to(G.dtype)


# ─── Stiefel tangent helpers ─────────────────────────────────────────────────
def project_tangent(Q, Z):
    """Project Z onto tangent space of Stiefel at Q: Z - Q sym(Q.T Z)"""
    import torch
    S = Q.T @ Z
    return Z - Q @ (0.5 * (S + S.T))


def qr_retract(Q, Z):
    """Retraction: QR decomp of Q + Z"""
    import torch
    M = Q + Z
    Qn, R = torch.linalg.qr(M)
    # Fix signs so diagonal of R is positive
    d = torch.sign(torch.diagonal(R))
    d[d == 0] = 1.0
    return Qn * d.unsqueeze(0)


# ─── IntrinsicMuon Optimizer (standalone, param-level) ──────────────────────
class IntrinsicMuonOptimizer:
    """Stiefel Riemannian Muon: tangent-grad → NS5 → momentum → QR retract.
    No S_t learning. Mirrors mlp_batch10_im.py but operates on arbitrary params.
    """
    def __init__(self, stiefel_params, lr=3e-2, momentum=0.95, ns_steps=5):
        self.params   = list(stiefel_params)
        self.lr       = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self._m = {}  # momentum buffer

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        import torch
        with torch.no_grad():
            for Q in self.params:
                if Q.grad is None:
                    continue
                qid = id(Q)
                G_bar = Q.grad.float()
                # Project to tangent
                G_tan = project_tangent(Q.float(), G_bar)
                # Newton-Schulz
                D_ns  = newton_schulz_5(G_tan, steps=self.ns_steps)
                D_ns  = D_ns * G_tan.norm()  # rescale by grad norm
                # Momentum
                if qid not in self._m:
                    self._m[qid] = torch.zeros_like(D_ns)
                m = self._m[qid]
                m_new = self.momentum * m + D_ns
                m_proj = project_tangent(Q.float(), m_new)
                self._m[qid] = m_proj
                # QR retraction
                Q_new = qr_retract(Q.float(), -self.lr * m_proj)
                # Transport momentum
                self._m[qid] = project_tangent(Q_new, m_proj)
                Q.data.copy_(Q_new.to(Q.dtype))


# ─── Model definitions ──────────────────────────────────────────────────────
def make_b1_model(vocab_size, embed_dim, hidden_dim, seed, device):
    """B1: W = Q @ diag(exp(log_s)), Q on Stiefel (via FS MF optimizer)."""
    import torch, torch.nn as nn, torch.nn.functional as F

    class B1LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=False)
            # vocab_size > hidden_dim → Q: (vocab_size, hidden_dim), no transpose
            n, r = vocab_size, hidden_dim
            self.proj_Q     = nn.Parameter(_qr_init(n, r, seed))
            self.proj_log_s = nn.Parameter(torch.zeros(r, device=device))
            self.proj_bias  = nn.Parameter(torch.zeros(vocab_size, device=device))
            nn.init.uniform_(self.embed.weight, -0.1, 0.1)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            s     = torch.exp(self.proj_log_s)
            W     = self.proj_Q * s.unsqueeze(0)   # (vocab, hidden) * (hidden,) → (vocab, hidden)
            logits = F.linear(out.view(-1, out.size(-1)), W, self.proj_bias)
            return logits, hidden

    return B1LSTMModel().to(device)


def make_b2_model(vocab_size, embed_dim, hidden_dim, seed, device):
    """B2: standard nn.Linear, no Stiefel."""
    import torch, torch.nn as nn, torch.nn.functional as F

    class B2LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=False)
            self.proj  = nn.Linear(hidden_dim, vocab_size)
            nn.init.uniform_(self.embed.weight, -0.1, 0.1)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            logits = self.proj(out.view(-1, out.size(-1)))
            return logits, hidden

    return B2LSTMModel().to(device)


def make_b3_model(vocab_size, embed_dim, hidden_dim, seed, device):
    """B3: FS-like LSTM (Q only, no S), optimized with IntrinsicMuon."""
    import torch, torch.nn as nn, torch.nn.functional as F

    class B3LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=False)
            n, r = vocab_size, hidden_dim
            self.proj_Q    = nn.Parameter(_qr_init(n, r, seed))
            self.proj_bias = nn.Parameter(torch.zeros(vocab_size, device=device))
            nn.init.uniform_(self.embed.weight, -0.1, 0.1)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            logits = F.linear(out.view(-1, out.size(-1)), self.proj_Q, self.proj_bias)
            return logits, hidden

    return B3LSTMModel().to(device)


# ─── Data loading ────────────────────────────────────────────────────────────
_DATASET_CACHE = {}

def load_wt2_data(device):
    import torch
    if 'wt2' in _DATASET_CACHE:
        td, vd, vs, w2i = _DATASET_CACHE['wt2']
        return td.to(device), vd.to(device), vs, w2i

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=HF_CACHE)
    train_text = "\n".join(ds["train"]["text"])
    val_text   = "\n".join(ds["validation"]["text"])

    words   = train_text.split()
    counter = Counter(words)
    vocab_words = ["<unk>", "<eos>"] + [w for w, _ in counter.most_common(VOCAB_SIZE_TARGET - 2)]
    w2i = {w: i for i, w in enumerate(vocab_words)}
    vocab_size = len(vocab_words)

    def text_to_ids(text):
        return torch.tensor([w2i.get(w, 0) for w in text.split()], dtype=torch.long)

    def batchify(data, bsz):
        nb = data.size(0) // bsz
        return data[:nb * bsz].view(bsz, -1).t().contiguous()

    train_ids  = text_to_ids(train_text)
    val_ids    = text_to_ids(val_text)
    train_data = batchify(train_ids, BATCH_SZ)
    val_data   = batchify(val_ids, 1)
    _DATASET_CACHE['wt2'] = (train_data, val_data, vocab_size, w2i)
    print(f"  WikiText-2 loaded: vocab={vocab_size}, train_tokens={train_ids.numel()}", flush=True)
    return train_data.to(device), val_data.to(device), vocab_size, w2i


def get_batch(source, i):
    import torch
    sl = min(SEQ_LEN, len(source) - 1 - i)
    x  = source[i:i + sl]
    y  = source[i + 1:i + 1 + sl].reshape(-1)
    return x, y


def eval_ppl(model, val_data):
    import torch, torch.nn.functional as F
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, SEQ_LEN):
            x, y = get_batch(val_data, i)
            logits, _ = model(x)
            total_loss   += F.cross_entropy(logits, y, reduction='sum').item()
            total_tokens += y.numel()
    return math.exp(total_loss / total_tokens)


# ─── Runner ──────────────────────────────────────────────────────────────────
def run_b1(seed, device, train_data, val_data, vocab_size):
    import torch
    from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig, ManifoldFlowOptimizer
    set_seed(seed)
    model = make_b1_model(vocab_size, EMBED_DIM, HIDDEN_DIM, seed, device)

    n_batches   = (train_data.size(0) - 1) // SEQ_LEN
    total_steps = N_EPOCHS * n_batches

    # Q: Stiefel via FS (rho_geo=0 = no SPD step, just manifold retraction)
    cfg = ManifoldFlowConfig(rho_geo=0.0, beta_P=0.95, lambda_S=0.001, K_geo=10)
    q_params    = [model.proj_Q]
    other_params = [p for n, p in model.named_parameters() if 'proj_Q' not in n]

    opt_mf   = ManifoldFlowOptimizer(q_params, base_optim='adam', lr=LR,
                                      betas=(0.9, 0.999), mf_config=cfg, total_steps=total_steps)
    opt_adam = torch.optim.Adam(other_params, lr=LR)

    best_ppl = float('inf')
    epoch_ppls = []
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for i in range(0, train_data.size(0) - 1, SEQ_LEN):
            x, y = get_batch(train_data, i)
            opt_mf.zero_grad(); opt_adam.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt_mf.step(); opt_adam.step()
        ppl = eval_ppl(model, val_data)
        epoch_ppls.append(ppl)
        if ppl < best_ppl: best_ppl = ppl
        print(f"    [B1_qdiag s={seed}] ep{epoch+1}/{N_EPOCHS} ppl={ppl:.2f} best={best_ppl:.2f}", flush=True)

    return {"baseline": "B1_qdiag", "seed": seed, "best_ppl": best_ppl,
            "epoch_ppls": epoch_ppls, "elapsed": time.time() - t0}


def run_b2(seed, device, train_data, val_data, vocab_size):
    import torch
    set_seed(seed)
    model = make_b2_model(vocab_size, EMBED_DIM, HIDDEN_DIM, seed, device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_ppl = float('inf')
    epoch_ppls = []
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for i in range(0, train_data.size(0) - 1, SEQ_LEN):
            x, y = get_batch(train_data, i)
            opt.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        ppl = eval_ppl(model, val_data)
        epoch_ppls.append(ppl)
        if ppl < best_ppl: best_ppl = ppl
        print(f"    [B2_dense  s={seed}] ep{epoch+1}/{N_EPOCHS} ppl={ppl:.2f} best={best_ppl:.2f}", flush=True)

    return {"baseline": "B2_dense", "seed": seed, "best_ppl": best_ppl,
            "epoch_ppls": epoch_ppls, "elapsed": time.time() - t0}


def run_b3(seed, device, train_data, val_data, vocab_size):
    import torch
    set_seed(seed)
    model = make_b3_model(vocab_size, EMBED_DIM, HIDDEN_DIM, seed, device)

    q_params    = [model.proj_Q]
    other_params = [p for n, p in model.named_parameters() if 'proj_Q' not in n]

    opt_im   = IntrinsicMuonOptimizer(q_params, lr=LR, momentum=0.95, ns_steps=5)
    opt_adam = torch.optim.Adam(other_params, lr=LR)

    best_ppl = float('inf')
    epoch_ppls = []
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        for i in range(0, train_data.size(0) - 1, SEQ_LEN):
            x, y = get_batch(train_data, i)
            opt_im.zero_grad(); opt_adam.zero_grad()
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt_im.step(); opt_adam.step()
        ppl = eval_ppl(model, val_data)
        epoch_ppls.append(ppl)
        if ppl < best_ppl: best_ppl = ppl
        print(f"    [B3_im     s={seed}] ep{epoch+1}/{N_EPOCHS} ppl={ppl:.2f} best={best_ppl:.2f}", flush=True)

    return {"baseline": "B3_intrinsic_muon", "seed": seed, "best_ppl": best_ppl,
            "epoch_ppls": epoch_ppls, "elapsed": time.time() - t0}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    import torch
    device = torch.device("cuda:0")  # GPU 2 (via CUDA_VISIBLE_DEVICES=2)
    print(f"Device: {device}  ({torch.cuda.get_device_name(0)})", flush=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load data once
    train_data, val_data, vocab_size, _ = load_wt2_data(device)

    # Global state for atexit
    all_results = {"B1_qdiag": {}, "B2_dense": {}, "B3_intrinsic_muon": {}}
    partial_path = OUT_BASE / "partial_all.json"

    def _atexit():
        flush_json(partial_path, all_results)
        print(f"[atexit] flushed {partial_path}", flush=True)

    atexit.register(_atexit)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))

    # Load existing partial
    if partial_path.exists():
        try:
            with open(partial_path) as f:
                all_results.update(json.load(f))
            print(f"Loaded partial: {partial_path}", flush=True)
        except Exception as e:
            print(f"Warning: could not load partial ({e}), starting fresh", flush=True)

    BASELINES = [
        ("B1_qdiag",          run_b1),
        ("B2_dense",           run_b2),
        ("B3_intrinsic_muon",  run_b3),
    ]

    for bname, run_fn in BASELINES:
        print(f"\n{'='*60}", flush=True)
        print(f"BASELINE: {bname}", flush=True)
        print(f"{'='*60}", flush=True)
        for seed in SEEDS:
            skey = str(seed)
            if isinstance(all_results[bname].get(skey, {}), dict) and \
               "best_ppl" in all_results[bname].get(skey, {}):
                ppl = all_results[bname][skey]["best_ppl"]
                print(f"  [{bname} seed={seed}] SKIP (cached PPL={ppl:.2f})", flush=True)
                continue
            try:
                result = run_fn(seed, device, train_data, val_data, vocab_size)
                all_results[bname][skey] = result
                flush_json(partial_path, all_results)
                print(f"  [{bname} seed={seed}] DONE best_ppl={result['best_ppl']:.2f}", flush=True)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  [{bname} seed={seed}] FAILED: {e}\n{tb}", flush=True)
                all_results[bname][skey] = {"status": "failed", "error": str(e)}
                flush_json(partial_path, all_results)

    # ─── Write per-baseline results.json ─────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    summary = {}
    for bname in ["B1_qdiag", "B2_dense", "B3_intrinsic_muon"]:
        out_dir = OUT_BASE / bname
        out_dir.mkdir(parents=True, exist_ok=True)

        seed_data = all_results.get(bname, {})
        ppls = {}
        for seed in SEEDS:
            v = seed_data.get(str(seed), {})
            if isinstance(v, dict) and "best_ppl" in v:
                ppls[seed] = v["best_ppl"]

        if len(ppls) >= 2:
            ppl_vals = list(ppls.values())
            mean_ppl = float(np.mean(ppl_vals))
            std_ppl  = float(np.std(ppl_vals))
            delta_vs_mf = mean_ppl - MF_ADAM_MEAN   # positive = baseline worse (MF better)
        else:
            mean_ppl = std_ppl = delta_vs_mf = None
            ppl_vals = []

        # Paired diffs
        paired_diffs = {}
        for seed, ppl in ppls.items():
            mf_ppl = MF_ADAM_PPLS.get(seed, None)
            if mf_ppl is not None:
                paired_diffs[seed] = ppl - mf_ppl  # positive = baseline worse

        per_seed = {}
        for seed in SEEDS:
            v = seed_data.get(str(seed), {})
            if isinstance(v, dict) and "best_ppl" in v:
                per_seed[seed] = {
                    "best_ppl": v["best_ppl"],
                    "epoch_ppls": v.get("epoch_ppls", []),
                    "elapsed": v.get("elapsed", 0),
                    "mf_adam_ppl": MF_ADAM_PPLS.get(seed),
                    "delta_vs_mf_adam": paired_diffs.get(seed),
                }

        result_obj = {
            "baseline": bname,
            "seeds": SEEDS,
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "per_seed": per_seed,
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "mf_adam_reference_mean": MF_ADAM_MEAN,
            "delta_vs_mf_adam_mean": delta_vs_mf,  # positive = MF better (lower PPL)
            "ppls": ppl_vals,
        }

        flush_json(out_dir / "results.json", result_obj)
        summary[bname] = result_obj

        ppl_str = f"{mean_ppl:.2f}±{std_ppl:.2f}" if mean_ppl else "N/A"
        delta_str = f"Δ={delta_vs_mf:+.2f}" if delta_vs_mf else "N/A"
        print(f"  {bname:25s}: PPL={ppl_str}  {delta_str} (pos=MF better)", flush=True)

    # ─── Verdict ──────────────────────────────────────────────────────────────
    print(f"\nMF-Adam reference: {MF_ADAM_MEAN:.2f} PPL (seeds 42/123/7)", flush=True)
    print(f"Verdict threshold: PPL > {MF_ADAM_MEAN + 5:.1f} = MF dominates\n", flush=True)

    verdicts = {}
    for bname, res in summary.items():
        d = res["delta_vs_mf_adam_mean"]
        if d is None:
            verdicts[bname] = "INCOMPLETE"
        elif d > 5.0:
            verdicts[bname] = f"MF dominates (Δ={d:+.2f}, baseline clearly worse)"
        elif d > 1.0:
            verdicts[bname] = f"MF marginal (Δ={d:+.2f}, borderline)"
        elif d > -1.0:
            verdicts[bname] = f"TIED (Δ={d:+.2f}, within 1 PPL — reviewer concern valid)"
        else:
            verdicts[bname] = f"MF LOST (Δ={d:+.2f}, baseline better)"

    for bname, v in verdicts.items():
        print(f"  {bname}: {v}", flush=True)

    # Overall
    deltas = [summary[b]["delta_vs_mf_adam_mean"] for b in summary
              if summary[b]["delta_vs_mf_adam_mean"] is not None]
    if deltas and all(d > 5 for d in deltas):
        overall = "✅ MF DOMINATES ALL — paper unblocked"
    elif deltas and any(d < 1 for d in deltas):
        overall = "⚠️  AT LEAST ONE BASELINE CLOSE TO MF — reviewer concern real"
    elif deltas and all(d > 0 for d in deltas):
        overall = "✅ MF better than all baselines but margin is moderate"
    else:
        overall = "❌ CHECK INDIVIDUAL RESULTS"

    print(f"\nOVERALL: {overall}", flush=True)

    # Final combined summary
    flush_json(OUT_BASE / "summary.json", {
        "baselines": summary,
        "verdicts": verdicts,
        "overall": overall,
        "mf_adam_reference": {"mean": MF_ADAM_MEAN, "per_seed": MF_ADAM_PPLS},
    })
    print(f"\nAll outputs in: {OUT_BASE}", flush=True)


if __name__ == "__main__":
    main()
