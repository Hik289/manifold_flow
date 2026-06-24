#!/usr/bin/env python3
"""
Batch 10 Stream β — Mini-Transformer FFN on WikiText-2

Architecture: 2-block Transformer (d_model=128, n_head=4, n_layer=2)
- FFN layers: StiefelLinear(128→256) + GELU + StiefelLinear(256→128)
- Attention QKV: FREE (standard nn.Linear)
- Embedding/Unembedding: FREE

Metric: Perplexity (PPL) — lower is better.
G3 paired diff: FS - MF (positive = MF is better).

Outputs: method_1/mini_transformer_wikitext/{stage_b_results.json,
          h1_pressure_persistence.json, g4_spectral_trace.json, REPORT.md}
"""
import sys, os, json, time, math, warnings, atexit, signal, traceback, re
from pathlib import Path
from collections import Counter
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
METHODS  = ['fs-sgd', 'fs-adam', 'mf-sgd', 'mf-adam']
SEEDS    = [42, 123, 2024]
N_EPOCHS_A = 5    # Stage A (quick tuning)
N_EPOCHS_B = 12   # Stage B
BATCH_SZ = 32
SEQ_LEN  = 128
VOCAB_SIZE_MAX = 10000  # top-K words
D_MODEL  = 128
N_HEAD   = 4
N_LAYER  = 2
FFN_DIM  = 256
LR_GRID  = [1e-3, 3e-3, 5e-3]
RHO = 1e-2; LS = 1e-3; KG = 10

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


# ─── Riemannian Adam ─────────────────────────────────────────────────────
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
    D = project_tangent(Q, m_hat / (v_hat.clamp(min=0.0).sqrt() + eps))
    Q_new = qr_retract(Q, -lr * D)
    state_obj.m = project_tangent(Q_new, state_obj.m)
    return Q_new


# ─── G3 paired test (PPL: lower is better → FS - MF, positive = MF better) ──
def g3_paired_test_ppl(fs_results, mf_results):
    """FS - MF: positive means MF has lower PPL (better)."""
    fs_ppls = [r['history'][-1]['test_ppl'] for r in fs_results]
    mf_ppls = [r['history'][-1]['test_ppl'] for r in mf_results]
    diffs = [f - m for f, m in zip(fs_ppls, mf_ppls)]  # positive = MF better
    mean_d = float(np.mean(diffs))
    se_d   = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    t_stat, p_2s = sp_stats.ttest_rel(fs_ppls, mf_ppls) if len(diffs) > 1 else (0.0, 1.0)
    p_1s = float(p_2s / 2 if t_stat > 0 else 1.0)
    return {
        'fs_ppls': fs_ppls, 'mf_ppls': mf_ppls, 'diffs_fs_minus_mf': diffs,
        'mean_diff': mean_d, 'se_diff': se_d,
        't_stat': float(t_stat), 'p_val_1sided': p_1s,
        'significant_1se': abs(mean_d) > se_d and mean_d > 0,
        'significant_p05': p_1s < 0.05,
    }


# ─── WikiText-2 data loading ──────────────────────────────────────────────
def load_wikitext2(vocab_size=VOCAB_SIZE_MAX):
    print("[Data] Loading WikiText-2 via HuggingFace datasets...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    print(f"[Data] Loaded: {list(ds.keys())}", flush=True)

    def get_text(split):
        texts = [row['text'] for row in ds[split] if row['text'].strip()]
        return ' '.join(texts)

    train_text = get_text('train')
    val_text   = get_text('validation')
    test_text  = get_text('test')

    # Build vocab: word-level, top vocab_size words
    words = re.findall(r'\w+|[^\w\s]', train_text.lower())
    counter = Counter(words)
    vocab_words = ['<pad>', '<unk>', '<eos>'] + [w for w, _ in counter.most_common(vocab_size - 3)]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    actual_vocab_size = len(vocab)
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']

    print(f"[Data] Vocab size: {actual_vocab_size}", flush=True)

    def tokenize(text):
        toks = re.findall(r'\w+|[^\w\s]', text.lower())
        return [vocab.get(t, unk_id) for t in toks] + [eos_id]

    train_ids = tokenize(train_text)
    val_ids   = tokenize(val_text)
    test_ids  = tokenize(test_text)

    print(f"[Data] Tokens: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)
    return train_ids, val_ids, test_ids, actual_vocab_size


def make_batch_loader(ids, seq_len, batch_sz):
    """Create (inputs, targets) pairs from token id list."""
    ids_t = torch.tensor(ids, dtype=torch.long)
    # Trim to multiple of seq_len * batch_sz
    n = len(ids_t) - 1
    n = (n // (seq_len * batch_sz)) * (seq_len * batch_sz)
    x = ids_t[:n].view(-1, seq_len)
    y = ids_t[1:n+1].view(-1, seq_len)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_sz, shuffle=True, drop_last=True)


# ─── FFN block with Stiefel linear layers ────────────────────────────────
class StiefelFFN(nn.Module):
    def __init__(self, d_model, ffn_dim, mode='fs', seed=0, block_idx=0):
        super().__init__()
        self.fc1 = StiefelLinear(d_model, ffn_dim, seed=seed*100+block_idx*10+1, mode=mode)
        self.fc2 = StiefelLinear(ffn_dim, d_model, seed=seed*100+block_idx*10+2, mode=mode)
        self.mode = mode

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def stiefel_params(self):
        return [self.fc1.Q, self.fc2.Q]

    def stiefel_layers(self):
        return [self.fc1, self.fc2]

    def set_mode(self, mode):
        self.mode = mode
        self.fc1.mode = mode
        self.fc2.mode = mode


# ─── Mini-Transformer Block ───────────────────────────────────────────────
class MiniTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_dim, mode='fs', seed=0, block_idx=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = StiefelFFN(d_model, ffn_dim, mode=mode, seed=seed, block_idx=block_idx)
        self.mode  = mode

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

    def stiefel_params(self):
        return self.ffn.stiefel_params()

    def stiefel_layers(self):
        return self.ffn.stiefel_layers()

    def stiefel_layer_names(self, prefix=''):
        return [f"{prefix}ffn_fc1", f"{prefix}ffn_fc2"]

    def set_mode(self, mode):
        self.mode = mode
        self.ffn.set_mode(mode)


# ─── Full Mini-Transformer ────────────────────────────────────────────────
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=2, ffn_dim=256,
                 mode='fs', seed=0):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Embedding(512, d_model)   # positional embedding
        self.blocks  = nn.ModuleList([
            MiniTransformerBlock(d_model, n_head, ffn_dim, mode=mode,
                                 seed=seed, block_idx=i)
            for i in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model
        self.mode = mode

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.embed(x) + self.pos_enc(pos)
        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        for block in self.blocks:
            h = block(h, attn_mask=causal_mask)
        h   = self.norm(h)
        out = self.head(h)
        return out

    def stiefel_params(self):
        params = []
        for block in self.blocks:
            params.extend(block.stiefel_params())
        return params

    def stiefel_layers(self):
        layers = []
        for block in self.blocks:
            layers.extend(block.stiefel_layers())
        return layers

    def stiefel_layer_names(self):
        names = []
        for i, block in enumerate(self.blocks):
            names.extend(block.stiefel_layer_names(prefix=f"b{i}_"))
        return names

    def free_params(self):
        stiefel_ids = {id(p) for p in self.stiefel_params()}
        return [p for p in self.parameters() if id(p) not in stiefel_ids]

    def set_mode(self, mode):
        self.mode = mode
        for block in self.blocks:
            block.set_mode(mode)


# ─── MF S state ──────────────────────────────────────────────────────────
class MFSState:
    def __init__(self, r):
        self.S = torch.eye(r)

def mf_update_S(S, P_t, rho_geo, lambda_S, lambda_min, lambda_max, device):
    S = S.to(device); P_t = P_t.to(device)
    PPt = P_t @ P_t.t()
    grad_S = sym(PPt)
    S_new = affine_invariant_step(S, grad_S, rho_geo)
    S_new = spectral_clip(S_new, lambda_min, lambda_max)
    return S_new.cpu()


# ─── Evaluate PPL ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_ppl(model, loader, device):
    model.eval()
    total_loss = 0.0; total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)   # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T), reduction='sum')
        total_loss   += loss.item()
        total_tokens += B * T
    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return math.exp(min(avg_loss, 100.0))  # clamp for safety


# ─── Train one method ─────────────────────────────────────────────────────
def train_one(method, lr, seed, n_epochs, tr_ld, val_ld, te_ld, device, vocab_size,
              rho_geo=1e-2, lambda_S=1e-3, K_geo=10):
    mode_str = method.split('-')[0]
    base_opt = method.split('-')[1]

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MiniTransformer(vocab_size, D_MODEL, N_HEAD, N_LAYER, FFN_DIM,
                            mode=mode_str, seed=seed).to(device)

    # Free params → regular optimizer
    free_params = model.free_params()
    if base_opt == 'sgd':
        reg_opt = torch.optim.SGD(free_params, lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        reg_opt = torch.optim.Adam(free_params, lr=lr, weight_decay=1e-4)

    # Stiefel states
    stiefel_q_states = {}
    stiefel_adam_st  = {}
    mf_S_states      = {}
    mf_cfg = ManifoldFlowConfig(rho_geo=rho_geo, lambda_S=lambda_S, K_geo=K_geo)

    history = []
    per_epoch_trace = []
    _P_prev = {}
    h1_cos_accumulator = {nm: [] for nm in model.stiefel_layer_names()}
    _step_count = 0

    total_steps  = n_epochs * len(tr_ld)
    warmup_steps = max(1, int(0.05 * total_steps))

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_cos_P = {nm: [] for nm in model.stiefel_layer_names()}
        epoch_loss  = 0.0; n_batches = 0

        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            _step_count += 1

            reg_opt.zero_grad()
            for p in model.stiefel_params():
                if p.grad is not None: p.grad.zero_()

            B, T = x.shape
            logits = model(x)   # (B, T, V)
            loss = F.cross_entropy(logits.view(B*T, -1), y.view(B*T))
            loss.backward()

            # Gradient clip for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── Free param step ──
            reg_opt.step()

            # ── Stiefel Q step ──
            fc_names  = model.stiefel_layer_names()
            fc_layers = model.stiefel_layers()

            for nm, layer in zip(fc_names, fc_layers):
                Q = layer.Q
                if Q.grad is None: continue
                G_bar = Q.grad.float()
                split = decompose_tangent_normal(Q.detach().float(), G_bar)
                G_tan = split.G_tan; P_t = split.P
                qid = id(Q)

                # H1 cos
                if qid in _P_prev:
                    P_prev = _P_prev[qid]
                    cos_val = float(((P_t * P_prev).sum() / (P_t.norm() * P_prev.norm() + 1e-12)).item())
                    epoch_cos_P[nm].append(cos_val)
                _P_prev[qid] = P_t.detach().clone()

                # MF S update
                if mode_str == 'mf' and _step_count >= warmup_steps and _step_count % K_geo == 0:
                    if qid not in mf_S_states:
                        mf_S_states[qid] = MFSState(layer.r)
                    S_state = mf_S_states[qid]
                    S_state.S = mf_update_S(
                        S_state.S, P_t, rho_geo, lambda_S,
                        mf_cfg.lambda_min, mf_cfg.lambda_max, device)
                    layer.update_sqrtS_cache(S_state.S)

                # MF effective gradient
                if mode_str == 'mf' and qid in mf_S_states:
                    S = mf_S_states[qid].S.to(device, G_tan.dtype)
                    try:
                        S_inv = torch.linalg.inv(S)
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
                    Q_new  = stiefel_adam_step(Q.detach().float(), G_tan_eff, astate, lr)
                Q.data.copy_(Q_new.to(Q.dtype))

            epoch_loss += loss.item(); n_batches += 1

        # ── End of epoch ──
        val_ppl  = evaluate_ppl(model, val_ld, device)
        test_ppl = evaluate_ppl(model, te_ld, device)
        avg_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')

        cos_P_epoch = {}
        for nm in model.stiefel_layer_names():
            vals = epoch_cos_P[nm]
            if vals:
                mean_cos = float(np.mean(vals))
                cos_P_epoch[nm] = mean_cos
                h1_cos_accumulator[nm].append(mean_cos)

        S_stats_epoch = {}
        if mode_str == 'mf':
            for nm, layer in zip(model.stiefel_layer_names(), model.stiefel_layers()):
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
                        'lambda_max': lam_max, 'lambda_min': lam_min, 'lambda_ratio': ratio,
                    }

        rec = {
            'epoch': epoch, 'train_loss': avg_loss,
            'val_ppl': val_ppl, 'test_ppl': test_ppl,
            'cos_P': cos_P_epoch, 'S_stats': S_stats_epoch,
        }
        per_epoch_trace.append(rec)
        history.append({'epoch': epoch, 'val_ppl': val_ppl, 'test_ppl': test_ppl,
                        'train_loss': avg_loss})

        if epoch % 3 == 0 or epoch == n_epochs:
            print(f"    [{method} seed={seed}] ep={epoch}/{n_epochs} "
                  f"loss={avg_loss:.3f} val_ppl={val_ppl:.1f} test_ppl={test_ppl:.1f} "
                  f"cos_P={[f'{v:.3f}' for v in cos_P_epoch.values()]}", flush=True)

    # H1 stats: first 1/3
    cutoff = n_epochs // 3
    h1_stats = {}
    for nm in model.stiefel_layer_names():
        early_vals = h1_cos_accumulator[nm][:cutoff]
        if early_vals:
            h1_stats[nm] = {
                'mean': float(np.mean(early_vals)),
                'std':  float(np.std(early_vals)),
                'n':    len(early_vals),
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


# ─── Stage A: LR grid ─────────────────────────────────────────────────────
def stage_a(tr_ld, val_ld, te_ld, device, vocab_size):
    print("\n=== Stage A: LR grid search ===", flush=True)
    best_lrs = {}
    seed_a = SEEDS[0]
    for method in METHODS:
        best_ppl = float('inf'); best_lr = LR_GRID[0]
        for lr in LR_GRID:
            print(f"  [Stage A] {method} lr={lr} seed={seed_a}", flush=True)
            result = train_one(method, lr, seed_a, N_EPOCHS_A,
                               tr_ld, val_ld, te_ld, device, vocab_size,
                               rho_geo=RHO, lambda_S=LS, K_geo=KG)
            val_ppl = result['history'][-1]['val_ppl']
            print(f"    → val_ppl={val_ppl:.1f}", flush=True)
            if val_ppl < best_ppl:
                best_ppl = val_ppl; best_lr = lr
        best_lrs[method] = best_lr
        print(f"  [Stage A] BEST {method}: lr={best_lr} val_ppl={best_ppl:.1f}", flush=True)
    return best_lrs


# ─── Main run ─────────────────────────────────────────────────────────────
def run(out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_ids, val_ids, test_ids, vocab_size = load_wikitext2()
    tr_ld  = make_batch_loader(train_ids, SEQ_LEN, BATCH_SZ)
    val_ld = make_batch_loader(val_ids,   SEQ_LEN, BATCH_SZ)
    te_ld  = make_batch_loader(test_ids,  SEQ_LEN, BATCH_SZ)

    print(f"\nTrain batches={len(tr_ld)} Val batches={len(val_ld)} Test batches={len(te_ld)}", flush=True)
    print(f"Vocab={vocab_size} d_model={D_MODEL} n_head={N_HEAD} n_layer={N_LAYER} ffn_dim={FFN_DIM}", flush=True)
    print(f"GPU: {device}   Methods: {METHODS}   Seeds: {SEEDS}   Epochs-B: {N_EPOCHS_B}", flush=True)

    # Stage A
    best_lrs = stage_a(tr_ld, val_ld, te_ld, device, vocab_size)
    print(f"\n[Stage A done] best_lrs={best_lrs}", flush=True)
    with open(out_dir / 'best_lrs.json', 'w') as f:
        json.dump(best_lrs, f, indent=2)

    # Accumulators
    all_results = {m: [] for m in METHODS}
    stage_b = {
        'task': 'mini_transformer_wikitext', 'status': 'running',
        'methods': {m: [] for m in METHODS},
    }
    h1_persist = {'task': 'mini_transformer_wikitext', 'methods': {}}
    g4_trace   = {'task': 'mini_transformer_wikitext', 'methods': {}}

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

    # Stage B
    print("\n=== Stage B: 3 seeds × 4 methods ===", flush=True)

    for method in METHODS:
        lr = best_lrs[method]
        print(f"\n--- Method: {method}  lr={lr} ---", flush=True)
        h1_persist['methods'][method] = {}
        g4_trace['methods'][method] = []

        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            result = train_one(method, lr, seed, N_EPOCHS_B,
                               tr_ld, val_ld, te_ld, device, vocab_size,
                               rho_geo=RHO, lambda_S=LS, K_geo=KG)
            all_results[method].append(result)

            # H1
            for nm, h1s in result['h1_stats'].items():
                if nm not in h1_persist['methods'][method]:
                    h1_persist['methods'][method][nm] = []
                h1_persist['methods'][method][nm].append({'seed': seed, **h1s})

            # G4 (MF only)
            mode_str = method.split('-')[0]
            if mode_str == 'mf':
                g4_trace['methods'][method].append({
                    'seed': seed,
                    'per_epoch': [
                        {
                            'epoch': rec['epoch'],
                            'S_stats': rec.get('S_stats', {}),
                            'cos_P':   rec.get('cos_P', {}),
                        }
                        for rec in result['per_epoch_trace']
                    ]
                })

            stage_b['methods'][method].append({
                'seed': seed, 'lr': lr,
                'final_test_ppl': result['history'][-1]['test_ppl'],
                'final_val_ppl':  result['history'][-1]['val_ppl'],
                'history_summary': [
                    {'epoch': h['epoch'], 'test_ppl': h['test_ppl'],
                     'train_loss': h['train_loss']}
                    for h in result['history']
                ],
            })
            flush_all()

    # G3 Analysis (PPL: FS - MF, positive = MF better)
    print("\n=== G3 Analysis (PPL) ===", flush=True)
    g3_results = {}
    for base_opt in ['sgd', 'adam']:
        mf_k, fs_k = f'mf-{base_opt}', f'fs-{base_opt}'
        if all_results[mf_k] and all_results[fs_k]:
            g3 = g3_paired_test_ppl(all_results[fs_k], all_results[mf_k])
            g3_results[base_opt] = g3
            print(f"  {fs_k} vs {mf_k}: FS-MF diff={g3['mean_diff']:+.2f} SE={g3['se_diff']:.2f} "
                  f"t={g3['t_stat']:.2f} p={g3['p_val_1sided']:.3f} "
                  f"1SE={'YES' if g3['significant_1se'] else 'no'}", flush=True)

    # H1 Summary
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

    # G4 Summary
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

    # Verdict
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

    print(f"\n=== VERDICT: Mini-Transformer WikiText-2 ===", flush=True)
    print(f"  H1 max cos(P)={h1_max:.4f} ({'CONFIRM' if h1_confirm else 'fail'} thr=0.2)", flush=True)
    print(f"  G3 SGD={'CONFIRM' if g3_sgd else 'fail'}  Adam={'CONFIRM' if g3_adam else 'fail'}", flush=True)
    print(f"  G4 max λ_ratio={g4_max_ratio:.3f} ({'CONFIRM' if g4_confirm else 'fail'} thr=1.5)", flush=True)
    print(f"  Double: {'*** YES ***' if verdict['double_confirm'] else 'no'}", flush=True)
    print(f"  Triple: {'*** YES ***' if verdict['triple_confirm'] else 'no'}", flush=True)

    stage_b['status'] = 'done'
    stage_b['g3'] = g3_results
    stage_b['h1_summary'] = h1_summary
    stage_b['g4_summary'] = g4_summary
    stage_b['verdict'] = verdict
    h1_persist['h1_summary'] = h1_summary
    g4_trace['g4_summary'] = g4_summary
    g4_trace['verdict'] = {'g4_max_ratio': float(g4_max_ratio), 'g4_confirm': g4_confirm}
    flush_all()

    # REPORT.md
    lines = [
        "# Mini-Transformer WikiText-2 — Batch 10 Stream β Report\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}\n\n",
        "## Architecture\n",
        f"- d_model={D_MODEL}, n_head={N_HEAD}, n_layer={N_LAYER}, ffn_dim={FFN_DIM}\n",
        "- **FFN layers**: Stiefel-parameterized (fc1: 128→256, fc2: 256→128 per block)\n",
        "- **Attention QKV**: FREE (standard nn.MultiheadAttention)\n",
        "- **Embedding/head**: FREE\n",
        f"- vocab_size={vocab_size}, seq_len={SEQ_LEN}, batch={BATCH_SZ}\n\n",
        "## Setup\n",
        f"- {METHODS}\n",
        f"- Seeds: {SEEDS}  Epochs: {N_EPOCHS_B}\n",
        f"- MF hyper: rho_geo={RHO}, lambda_S={LS}, K_geo={KG}\n\n",
        "## Best LRs (Stage A)\n",
    ]
    for m, lr_v in best_lrs.items():
        lines.append(f"- {m}: {lr_v}\n")

    lines += ["\n## Final Test PPL (lower is better)\n\n| Method | S1 | S2 | S3 | Mean |\n|---|---|---|---|---|\n"]
    for method in METHODS:
        if not all_results[method]: continue
        ppls = [r['history'][-1]['test_ppl'] for r in all_results[method]]
        lines.append(f"| {method} | " + " | ".join(f"{p:.1f}" for p in ppls) + f" | {np.mean(ppls):.1f} |\n")

    lines += ["\n## G3: FS - MF PPL diff (positive = MF better)\n\n"]
    for base_opt in ['sgd', 'adam']:
        if base_opt not in g3_results: continue
        g = g3_results[base_opt]
        lines += [
            f"### {base_opt.upper()}\n",
            f"- FS-MF Δ mean={g['mean_diff']:+.2f} ± {g['se_diff']:.2f} SE  t={g['t_stat']:.2f}  p={g['p_val_1sided']:.3f}\n",
            f"- **>1 SE: {'✅ YES' if g['significant_1se'] else '❌ no'}**  "
            f"p<0.05: {'✅ YES' if g['significant_p05'] else '❌ no'}\n\n",
        ]

    lines += ["\n## H1: cos(P_t, P_{t-1}) per FFN layer (first 1/3)\n\n"]
    for method in METHODS:
        lines.append(f"### {method}\n")
        for nm, stats in h1_summary.get(method, {}).items():
            m = stats.get('mean_across_seeds')
            p = stats.get('p_val_1sided')
            lines.append(f"- {nm}: mean={m:.4f}" + (f"  p={p:.3f}" if p else "") + "\n")
        lines.append("\n")

    lines += ["\n## G4: λ_max/λ_min(S) at end of training (FFN layers)\n\n"]
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
        print(f"\n[SIGNAL {sig}] flushing...", flush=True); sys.exit(0)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[Batch10-β Transformer WikiText-2] GPU=cuda:2 device={device}", flush=True)

    out_dir = BASE / "method_1" / "mini_transformer_wikitext"

    try:
        verdict = run(out_dir, device)
        print(f"\n[DONE] double_confirm={verdict['double_confirm']} triple_confirm={verdict['triple_confirm']}", flush=True)
        print(f"[DONE] H1_max={verdict['h1_max_cos']:.4f} G3={verdict['g3_confirm']} G4_ratio={verdict['g4_max_ratio']:.3f}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
