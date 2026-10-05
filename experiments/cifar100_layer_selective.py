#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import sys, os, json, time, math, io, atexit, signal, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

BASE     = Path("./experiments")
SRC      = BASE / "src"
DATA_ROOT = Path("./datasets/image/cifar100")
OUT_DIR  = BASE / "method_1" / "cifar100_r18_layerselective"
sys.path.insert(0, str(SRC))

from manifoldflow.spd_ops import sym, fp32_eigh, symlogm, affine_invariant_step, spectral_clip, matrix_sqrt
from manifoldflow.retraction import qr_retract, procrustes_align
from manifoldflow.tangent import decompose_tangent_normal, project_tangent
from manifoldflow.manifoldflow_optimizer import _stiefel_sgd_step, ManifoldFlowConfig


def js(obj):
    if isinstance(obj, dict):  return {k: js(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [js(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, torch.Tensor):   return float(obj.item())
    return obj

_results_b1 = {"status": "running", "epochs": []}
_results_b2 = {"status": "running", "epochs": []}

def flush_all():
    try:
        (OUT_DIR / "B1_fc_only.json").write_text(json.dumps(js(_results_b1), indent=2))
        (OUT_DIR / "B2_layer4_only.json").write_text(json.dumps(js(_results_b2), indent=2))
        print("[FLUSH] B1+B2 written.", flush=True)
    except Exception as e:
        print(f"[FLUSH ERROR] {e}", flush=True)


def sig_handler(signum, frame):
    print(f"\n[SIGNAL {signum}] Flushing...", flush=True)
    flush_all()
    sys.exit(0)



class CIFAR100Parquet(Dataset):
    MEAN = (0.5071, 0.4865, 0.4409)
    STD  = (0.2673, 0.2564, 0.2762)

    def __init__(self, path, train):
        import torchvision.transforms as T
        import pyarrow.parquet as pq
        from PIL import Image as PILImage
        print(f'  Decoding {path.name} ...', flush=True)
        t0 = time.time()
        tbl = pq.read_table(str(path))
        d = tbl.to_pydict()
        raw = d['img']
        n = len(raw)
        imgs = np.empty((n, 32, 32, 3), dtype=np.uint8)
        for i, item in enumerate(raw):
            img = PILImage.open(io.BytesIO(item['bytes'])).convert('RGB')
            imgs[i] = np.asarray(img, dtype=np.uint8)
        self.imgs   = imgs
        self.labels = np.array(d['fine_label'], dtype=np.int64)
        print(f'  Done ({time.time()-t0:.1f}s, {n} images)', flush=True)
        if train:
            self.tfm = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                   T.ToTensor(), T.Normalize(self.MEAN, self.STD)])
        else:
            self.tfm = T.Compose([T.ToTensor(), T.Normalize(self.MEAN, self.STD)])

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        from PIL import Image as PILImage
        return self.tfm(PILImage.fromarray(self.imgs[i])), int(self.labels[i])

def get_loaders(batch_size=128, num_workers=4):
    train_ds = CIFAR100Parquet(DATA_ROOT / 'train.parquet', train=True)
    test_ds  = CIFAR100Parquet(DATA_ROOT / 'test.parquet',  train=False)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    test_ld  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_ld, test_ld


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


class StandardR18(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_planes, planes, stride=s))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


class StiefelLinear(nn.Module):

    def __init__(self, in_features, out_features, mode='mf'):
        super().__init__()
        self.mode = mode
        n, r = in_features, out_features
        Q_init = torch.linalg.qr(torch.randn(n, r))[0]
        self.Q = nn.Parameter(Q_init.float())
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._sqrtS = torch.eye(r)

    def forward(self, x):
        if self.mode == 'mf' and self._sqrtS is not None:
            W = self.Q @ self._sqrtS.to(self.Q.device, self.Q.dtype)
        else:
            W = self.Q
        return F.linear(x, W.T, self.bias)


class StiefelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, mode='mf'):
        super().__init__()
        self.mode = mode
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        n_sp = in_channels * kernel_size * kernel_size
        if n_sp >= out_channels:
            n, r = n_sp, out_channels
            self.is_transpose = True
        else:
            n, r = out_channels, n_sp
            self.is_transpose = False
        self.n, self.r = n, r
        Q_init = torch.linalg.qr(torch.randn(n, r))[0]
        self.Q = nn.Parameter(Q_init.float())
        self._sqrtS = torch.eye(r)

    def _get_weight(self):
        if self.mode == 'mf' and self._sqrtS is not None:
            W_flat = self.Q @ self._sqrtS.to(self.Q.device, self.Q.dtype)
        else:
            W_flat = self.Q
        if self.is_transpose:
            W = W_flat.T.contiguous()
        else:
            W = W_flat
        return W.view(self.out_channels, self.in_channels,
                      self.kernel_size, self.kernel_size)

    def forward(self, x):
        return F.conv2d(x, self._get_weight(), bias=None,
                        stride=self.stride, padding=self.padding)


class HybridR18(nn.Module):

    def __init__(self, mode='fc_only', stiefel_mode='mf', num_classes=100):
        super().__init__()
        self.hybrid_mode = mode
        self.stiefel_mode = stiefel_mode

        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_std_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_std_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_std_layer(128, 256, 2, stride=2)

        if mode == 'layer4_only':
            self.layer4 = self._make_stiefel_layer(256, 512, 2, stride=2)
            self.fc = nn.Linear(512, num_classes)
        elif mode == 'fc_only':
            self.layer4 = self._make_std_layer(256, 512, 2, stride=2)
            self.fc = StiefelLinear(512, num_classes, mode=stiefel_mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self._stiefel_modules = []

    def _make_std_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(in_planes, planes, stride=s))
            in_planes = planes
        return nn.Sequential(*layers)

    def _make_stiefel_layer(self, in_planes, planes, num_blocks, stride):

        blocks = []
        b0 = self._make_stiefel_block(in_planes, planes, stride=stride)
        blocks.append(b0)
        for _ in range(1, num_blocks):
            blocks.append(self._make_stiefel_block(planes, planes, stride=1))
        return nn.ModuleList(blocks)

    def _make_stiefel_block(self, in_planes, planes, stride=1):

        class StiefelBasicBlock(nn.Module):
            def __init__(sb, in_p, p, s):
                super().__init__()
                sm = self.stiefel_mode
                sb.conv1 = StiefelConv2d(in_p, p, 3, stride=s, padding=1, mode=sm)
                sb.bn1   = nn.BatchNorm2d(p)
                sb.conv2 = StiefelConv2d(p, p, 3, stride=1, padding=1, mode=sm)
                sb.bn2   = nn.BatchNorm2d(p)
                sb.downsample = None
                if s != 1 or in_p != p:
                    sb.downsample = nn.Sequential(
                        StiefelConv2d(in_p, p, 1, stride=s, padding=0, mode=sm),
                        nn.BatchNorm2d(p)
                    )
            def forward(sb, x):
                identity = x
                out = F.relu(sb.bn1(sb.conv1(x)))
                out = sb.bn2(sb.conv2(out))
                if sb.downsample is not None:
                    identity = sb.downsample(x)
                return F.relu(out + identity)
        return StiefelBasicBlock(in_planes, planes, stride)

    def get_stiefel_params(self):

        params = []
        for module in self.modules():
            if isinstance(module, (StiefelConv2d, StiefelLinear)):
                params.append(module.Q)
        return params

    def get_stiefel_modules(self):

        modules = []
        for name, module in self.named_modules():
            if isinstance(module, (StiefelConv2d, StiefelLinear)):
                modules.append((name, module))
        return modules

    def get_other_params(self):
        stiefel_ids = {id(p) for p in self.get_stiefel_params()}
        return [p for p in self.parameters() if id(p) not in stiefel_ids]

    def update_sqrtS_cache(self, optimizer):

        if self.stiefel_mode != 'mf': return
        for module in self.modules():
            if isinstance(module, (StiefelConv2d, StiefelLinear)):
                Q = module.Q
                st = optimizer.state.get(Q, {})
                if 'S' in st:
                    S = st['S']
                    sqrtS = matrix_sqrt(sym(S)).to(Q.dtype)
                    module._sqrtS = sqrtS

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        if isinstance(self.layer4, nn.ModuleList):
            for block in self.layer4:
                h = block(h)
        else:
            h = self.layer4(h)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        if isinstance(self.fc, StiefelLinear):
            return self.fc(h)
        else:
            return self.fc(h)


class StiefelSGD(torch.optim.Optimizer):


    def __init__(self, params, lr=0.1, momentum=0.9, mf_config=None,
                 total_steps=None, mode='mf'):
        if mf_config is None: mf_config = ManifoldFlowConfig()
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.cfg = mf_config
        self.total_steps = total_steps
        self.mode = mode
        self._plog = {}
        self._S_updated = set()

    def _warmup_steps(self):
        if not self.total_steps: return 0
        return int(math.ceil(self.cfg.warmup_frac * self.total_steps))

    @torch.no_grad()
    def step(self, closure=None):
        cfg = self.cfg
        eps = 1e-8
        self._S_updated.clear()
        for g in self.param_groups:
            lr, mo = g['lr'], g['momentum']
            gamma_t = cfg.rho_geo * lr
            for Q in g['params']:
                if Q.grad is None: continue
                G_bar = Q.grad.to(Q.dtype)
                st = self.state[Q]
                if not st:
                    st['step'] = 0
                    r = Q.shape[-1]
                    st['S']       = torch.eye(r, dtype=torch.float32, device=Q.device)
                    st['M_P']     = torch.zeros(r, r, dtype=Q.dtype, device=Q.device)
                    st['Q_prev']  = Q.clone()
                    st['P_prev']  = None

                t, S, M_P, Q_prev = st['step'], st['S'], st['M_P'], st['Q_prev']
                split = decompose_tangent_normal(Q, G_bar)
                G_tan, P_t = split.G_tan, split.P

                Q_new = _stiefel_sgd_step(Q, G_tan, st, lr, mo)

                do_geo = False
                a_t, c_t = 0.0, 0.0
                if self.mode == 'mf':
                    warmup_done = t >= self._warmup_steps()
                    do_geo = (gamma_t > 0.0) and warmup_done and (t % cfg.K_geo == 0)
                    if t > 0 and do_geo:
                        A = Q.T @ Q_prev
                        O_t = procrustes_align(A)
                        M_P = O_t @ M_P @ O_t.T
                    G_bar_norm = G_bar.norm() + eps
                    M_P_prev = M_P.clone()
                    M_P_new = cfg.beta_P * M_P + (1.0 - cfg.beta_P) * (P_t / G_bar_norm)
                    st['M_P'] = M_P_new
                    if do_geo:
                        Pn = P_t.norm() + eps
                        Mn = M_P_prev.norm() + eps
                        c_t_v = (P_t * M_P_prev).sum() / (Pn * Mn)
                        c_t = float(c_t_v)
                        r_t = (Q @ P_t).norm() / (G_tan.norm() + eps)
                        a_c = torch.sigmoid(torch.tensor(
                            cfg.alpha_c * (c_t - cfg.tau_c), dtype=Q.dtype, device=Q.device))
                        a_r = torch.sigmoid(cfg.alpha_r * (torch.log(r_t + 1e-12) - cfg.tau_r))
                        a_t = float(a_c * a_r)
                        H_t = (sym(M_P_new) + cfg.lambda_S * symlogm(S.to(Q.dtype))).to(torch.float32)
                        S_new = affine_invariant_step(S, H_t, gamma_t * a_t)
                        st['S'] = spectral_clip(S_new, cfg.lambda_min, cfg.lambda_max)
                        self._S_updated.add(id(Q))

                Q.data.copy_(Q_new)
                st['Q_prev'] = Q_new.clone()
                st['step'] = t + 1

                eigvals, _ = fp32_eigh(st['S'])
                self._plog[id(Q)] = {
                    'P_t': P_t.detach().clone(),
                    'P_norm': float(P_t.norm()),
                    'G_tan_norm': float(G_tan.norm()),
                    'lambda_min_S': float(eigvals.min()),
                    'lambda_max_S': float(eigvals.max()),
                    'c_t': c_t, 'a_t': a_t, 'step': t, 'do_geo': do_geo,
                }
        return None

    def get_spectral_state(self, Q):
        st = self.state.get(Q, {})
        S = st.get('S')
        if S is None: return None
        eigvals, _ = fp32_eigh(S)
        return float(eigvals.max()), float(eigvals.min())


def cosine_lr(base_lr, epoch, total):
    return 0.5 * base_lr * (1 + math.cos(math.pi * epoch / total))

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def train_ablation(ablation_mode, stiefel_mode, train_ld, test_ld, device,
                   seed, lr, wd, n_epochs, mf_config, out_dict, record_every=5):

    torch.manual_seed(seed); np.random.seed(seed)

    model = HybridR18(mode=ablation_mode, stiefel_mode=stiefel_mode,
                      num_classes=100).to(device)

    stiefel_params = model.get_stiefel_params()
    other_params   = model.get_other_params()
    total_steps    = n_epochs * len(train_ld)

    print(f"  Stiefel params: {len(stiefel_params)} tensors", flush=True)
    for sp in stiefel_params:
        print(f"    shape: {list(sp.shape)}", flush=True)

    opt_s = StiefelSGD(stiefel_params, lr=lr, momentum=0.9,
                       mf_config=mf_config, total_steps=total_steps,
                       mode=stiefel_mode)
    opt_other = torch.optim.SGD(other_params, lr=lr, momentum=0.9, weight_decay=wd)

    if stiefel_mode == 'mf':
        for module in model.modules():
            if isinstance(module, (StiefelConv2d, StiefelLinear)):
                r = module.Q.shape[-1]
                module._sqrtS = torch.eye(r, dtype=module.Q.dtype, device=device)

    history = []
    P_epoch_store = {id(Q): None for Q in stiefel_params}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        lr_e = cosine_lr(lr, epoch, n_epochs)
        for g in opt_s.param_groups:   g['lr'] = lr_e
        for g in opt_other.param_groups: g['lr'] = lr_e

        tr_correct = tr_total = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt_s.zero_grad(); opt_other.zero_grad()
            out = model(x)
            F.cross_entropy(out, y).backward()
            nn.utils.clip_grad_norm_(stiefel_params, max_norm=10.0)
            nn.utils.clip_grad_norm_(other_params,   max_norm=10.0)
            opt_s.step()
            opt_other.step()
            if stiefel_mode == 'mf' and opt_s._S_updated:
                model.update_sqrtS_cache(opt_s)
            tr_correct += (out.detach().argmax(1) == y).sum().item()
            tr_total   += y.size(0)

        tr_acc = tr_correct / tr_total

        cos_P_vals = []
        for Q in stiefel_params:
            plog = opt_s._plog.get(id(Q), {})
            P_curr = plog.get('P_t')
            P_prev = P_epoch_store[id(Q)]
            if P_curr is not None and P_prev is not None:
                cos_v = float((P_curr.cpu().flatten() * P_prev.flatten()).sum() /
                              (P_curr.cpu().norm() * P_prev.norm() + 1e-12))
                cos_P_vals.append(cos_v)
            if P_curr is not None:
                P_epoch_store[id(Q)] = P_curr.clone().cpu()

        if (epoch + 1) % record_every == 0 or epoch == n_epochs - 1:
            te_acc = eval_acc(model, test_ld, device)
            rec = {
                "epoch": epoch + 1,
                "train_acc": tr_acc,
                "test_acc": te_acc,
                "lr": lr_e,
                "elapsed_s": time.time() - t0,
            }

            spectral_info = []
            for Q in stiefel_params:
                spec = opt_s.get_spectral_state(Q)
                plog = opt_s._plog.get(id(Q), {})
                if spec:
                    lmax, lmin = spec
                    spectral_info.append({
                        "lambda_max": lmax, "lambda_min": lmin,
                        "lambda_ratio": lmax / (lmin + 1e-12),
                        "c_t": plog.get('c_t', 0.0),
                        "a_t": plog.get('a_t', 0.0),
                        "P_norm": plog.get('P_norm', 0.0),
                    })
            rec["spectral"] = spectral_info
            rec["cos_P_this_epoch"] = float(np.mean(cos_P_vals)) if cos_P_vals else None
            rec["cos_P_vals"] = cos_P_vals

            history.append(rec)
            out_dict["epochs"] = history
            out_dict["current_epoch"] = epoch + 1
            out_dict["current_test_acc"] = te_acc
            flush_all()

            print(f"  ep{epoch+1:3d}/{n_epochs} | tr {tr_acc*100:.2f}% "
                  f"te {te_acc*100:.2f}% | lr {lr_e:.4f} | "
                  f'cos_P: {rec.get("cos_P_this_epoch") or 0:.4f} | '
                  f"{time.time()-t0:.0f}s", flush=True)

    final_te = eval_acc(model, test_ld, device)

    cos_early = [r["cos_P_this_epoch"] for r in history
                 if r["epoch"] <= 10 and r["cos_P_this_epoch"] is not None]
    cos_all   = [r["cos_P_this_epoch"] for r in history
                 if r["cos_P_this_epoch"] is not None]

    out_dict.update({
        "status": "done",
        "final_test_acc": final_te,
        "H1_cos_P_early_mean": float(np.mean(cos_early)) if cos_early else None,
        "H1_cos_P_all_mean":   float(np.mean(cos_all)) if cos_all else None,
        "H1_verdict": ("SIGNAL>0.3" if (cos_early and np.mean(cos_early) > 0.3)
                       else "NO_SIGNAL" if (cos_early and np.mean(cos_early) <= 0.1)
                       else "WEAK"),
        "spectral_final": history[-1].get("spectral", []) if history else [],
    })
    flush_all()
    return out_dict


def legacy_main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atexit.register(flush_all)
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("\nLoading CIFAR-100 ...", flush=True)
    train_ld, test_ld = get_loaders(batch_size=128, num_workers=4)
    print(f"Train batches: {len(train_ld)}, Test batches: {len(test_ld)}", flush=True)

    mf_cfg = ManifoldFlowConfig(
        rho_geo=3e-3, lambda_S=1e-3, K_geo=10, beta_P=0.95,
        tau_c=0.1, tau_r=0.0, alpha_c=5.0, alpha_r=2.0,
        lambda_min=0.25, lambda_max=4.0, warmup_frac=0.05,
    )

    n_epochs = 30
    seed     = 42
    lr       = 0.1
    wd       = 5e-4

    print("\n" + "="*60, flush=True)
    print("B1: Stiefel-on-fc-only (MF mode)", flush=True)
    print("="*60, flush=True)

    _results_b1.update({
        "ablation": "fc_only", "stiefel_mode": "mf",
        "n_epochs": n_epochs, "seed": seed, "lr": lr,
        "mf_config": {"rho_geo": 3e-3, "lambda_S": 1e-3, "K_geo": 10},
        "hypothesis": "If fc cos_P > 0.3 → H1 signal exists in fc",
        "acceptance": "cos_early > 0.3 → SIGNAL | cos ≈ 0 → NO_SIGNAL",
    })
    flush_all()

    train_ablation(
        ablation_mode='fc_only',
        stiefel_mode='mf',
        train_ld=train_ld, test_ld=test_ld,
        device=device, seed=seed, lr=lr, wd=wd,
        n_epochs=n_epochs, mf_config=mf_cfg,
        out_dict=_results_b1, record_every=5,
    )
    print(f"\nB1 done. H1 verdict: {_results_b1.get('H1_verdict')}", flush=True)
    print(f"  cos_P early (ep1-10): {_results_b1.get('H1_cos_P_early_mean'):.4f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("B2: Stiefel-on-layer4-only (MF mode)", flush=True)
    print("="*60, flush=True)

    _results_b2.update({
        "ablation": "layer4_only", "stiefel_mode": "mf",
        "n_epochs": n_epochs, "seed": seed, "lr": lr,
        "mf_config": {"rho_geo": 3e-3, "lambda_S": 1e-3, "K_geo": 10},
        "hypothesis": "If layer4 cos_P > 0.3 → H1 signal in deep conv (cross-layer interference was suppressing it)",
        "acceptance": "cos_early > 0.3 → SIGNAL | cos ≈ 0 → vision truly has no H1 conv signal",
    })
    flush_all()

    train_ablation(
        ablation_mode='layer4_only',
        stiefel_mode='mf',
        train_ld=train_ld, test_ld=test_ld,
        device=device, seed=seed, lr=lr, wd=wd,
        n_epochs=n_epochs, mf_config=mf_cfg,
        out_dict=_results_b2, record_every=5,
        )
    print(f"\nB2 done. H1 verdict: {_results_b2.get('H1_verdict')}", flush=True)
    print(f"  cos_P early (ep1-10): {_results_b2.get('H1_cos_P_early_mean'):.4f}", flush=True)

    write_report()
    print("\nAll done!", flush=True)


def write_report():
    b1 = _results_b1
    b2 = _results_b2

    def fmt_cos(v):
        if v is None: return "N/A"
        return f"{v:.4f}"

    b1_cos_early = b1.get("H1_cos_P_early_mean")
    b2_cos_early = b2.get("H1_cos_P_early_mean")

    b1_verdict_str = ("SIGNAL (>0.3) ✅" if b1_cos_early and b1_cos_early > 0.3
                      else "NO_SIGNAL ❌" if b1_cos_early is not None
                      else "INCOMPLETE")
    b2_verdict_str = ("SIGNAL (>0.3) ✅" if b2_cos_early and b2_cos_early > 0.3
                      else "NO_SIGNAL ❌" if b2_cos_early is not None
                      else "INCOMPLETE")

    if b1_cos_early and b1_cos_early > 0.3:
        interp_b1 = "H1 signal EXISTS in fc layer. Layer-selective Stiefel (fc pivot) has value."
    elif b1_cos_early is not None:
        interp_b1 = "H1 signal ABSENT even in fc. Vision fc has no temporal coherence in pressure."
    else:
        interp_b1 = "B1 did not complete."

    if b2_cos_early and b2_cos_early > 0.3:
        interp_b2 = "H1 signal EXISTS in layer4 conv. Cross-layer interference was masking it."
    elif b2_cos_early is not None:
        interp_b2 = "H1 signal ABSENT in layer4 conv. Vision truly lacks H1 signal in deep conv."
    else:
        interp_b2 = "B2 did not complete."

    if b1_cos_early is not None and b2_cos_early is not None:
        if b1_cos_early > 0.3 or b2_cos_early > 0.3:
            overall = "PARTIAL H1 SIGNAL IN VISION: Layer-selective pivot may rescue results."
        else:
            overall = "NO H1 SIGNAL IN VISION: Stiefel geometry inactive on vision tasks. Pursue negative paper path."
    else:
        overall = "INCOMPLETE - check partial results."

    lines = [
        "# CIFAR-100 R18 Layer-Selective Stiefel H1 Diagnostic",
        f"\n**Date**: {time.strftime('%Y-%m-%d %H:%M UTC')}",
        f"**GPU**: [machine_name] cuda:2",
        f"**Config**: rho_geo=3e-3 λS=1e-3 K_geo=10 (CIFAR Stage A best) | 30 epochs | seed=42\n",
        "## Summary\n",
        f"| Ablation | cos_P early (ep1-10) | H1 verdict |",
        f"|----------|----------------------|------------|",
        f"| B1: fc-only | {fmt_cos(b1_cos_early)} | {b1_verdict_str} |",
        f"| B2: layer4-only | {fmt_cos(b2_cos_early)} | {b2_verdict_str} |",
        f"\n**Overall**: {overall}\n",
        "## B1: Stiefel-on-fc-only\n",
        f"- H1 cos_P early mean: {fmt_cos(b1_cos_early)}",
        f"- H1 cos_P all mean: {fmt_cos(b1.get('H1_cos_P_all_mean'))}",
        f"- Final test acc: {b1.get('final_test_acc', 0)*100:.2f}%",
        f"- Interpretation: {interp_b1}\n",
    ]

    if b1.get("epochs"):
        lines += ["| Epoch | train% | test% | cos_P | λ_ratio |",
                  "|-------|--------|-------|-------|---------|"]
        for rec in b1["epochs"]:
            spec = rec.get("spectral", [{}])
            ratio = spec[0].get("lambda_ratio", 1.0) if spec else 1.0
            cp = fmt_cos(rec.get("cos_P_this_epoch"))
            lines.append(f"| {rec['epoch']} | {rec['train_acc']*100:.2f}% | "
                         f"{rec['test_acc']*100:.2f}% | {cp} | {ratio:.4f} |")

    lines += [
        f"\n## B2: Stiefel-on-layer4-only\n",
        f"- H1 cos_P early mean: {fmt_cos(b2_cos_early)}",
        f"- H1 cos_P all mean: {fmt_cos(b2.get('H1_cos_P_all_mean'))}",
        f"- Final test acc: {b2.get('final_test_acc', 0)*100:.2f}%",
        f"- Interpretation: {interp_b2}\n",
    ]

    if b2.get("epochs"):
        lines += ["| Epoch | train% | test% | cos_P | λ_ratio |",
                  "|-------|--------|-------|-------|---------|"]
        for rec in b2["epochs"]:
            spec = rec.get("spectral", [{}])
            ratio = spec[0].get("lambda_ratio", 1.0) if spec else 1.0
            cp = fmt_cos(rec.get("cos_P_this_epoch"))
            lines.append(f"| {rec['epoch']} | {rec['train_acc']*100:.2f}% | "
                         f"{rec['test_acc']*100:.2f}% | {cp} | {ratio:.4f} |")

    lines += [
        "\n## Cross-task Context\n",
        "- Cora-GCN: cos_P ≈ 0.82 (H1 confirmed)",
        "- CIFAR-100 all-conv (previous): cos_P ≈ 0 across ALL layers",
        "- This diagnostic tests if isolation removes cross-layer interference",
    ]

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"Report written to {OUT_DIR / 'REPORT.md'}", flush=True)

import argparse
import hashlib
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from manifoldflow.parametrization import PlateauStopper, SpectrumLinear, make_optimizers, resolve_device, set_seed as _current_set_seed, write_json
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig

def current_build_model(architecture, classes, method, alpha, beta):
    if architecture == 'simplecnn':
        model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(256 * 4 * 4, 128), nn.ReLU(), SpectrumLinear(128, classes, method, alpha, beta))
    else:
        from torchvision.models import resnet18, resnet50
        factory = resnet18 if architecture == 'resnet18' else resnet50
        model = factory(weights=None, num_classes=classes)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = SpectrumLinear(model.fc.in_features, classes, method, alpha, beta)
    return model

def current_initial_digest(model):
    digest = hashlib.sha256()
    excluded = set()
    for (name, module) in model.named_modules():
        if isinstance(module, SpectrumLinear):
            digest.update(name.encode())
            digest.update(module.effective_weight().detach().cpu().numpy().tobytes())
            if module.bias is not None:
                digest.update(module.bias.detach().cpu().numpy().tobytes())
            excluded.update((id(value) for value in module.parameters()))
    for (name, parameter) in model.named_parameters():
        if id(parameter) not in excluded:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def current_make_loaders(args, seed, dataset_name):
    from sklearn.model_selection import train_test_split
    from torchvision import datasets, transforms
    if dataset_name == 'cifar10':
        (factory, classes) = (datasets.CIFAR10, 10)
        (mean, std) = ((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
    else:
        (factory, classes) = (datasets.CIFAR100, 100)
        (mean, std) = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    training = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = factory(args.data_dir, train=True, transform=training, download=args.download)
    val_set = factory(args.data_dir, train=True, transform=evaluation, download=args.download)
    test_set = factory(args.data_dir, train=False, transform=evaluation, download=args.download)
    (train_indices, val_indices) = train_test_split(np.arange(len(train_set)), test_size=args.validation_fraction, random_state=seed, stratify=np.asarray(train_set.targets))
    generator = torch.Generator().manual_seed(seed)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.device.type == 'cuda')
    train_loader = DataLoader(Subset(train_set, train_indices.tolist()), shuffle=True, generator=generator, **options)
    val_loader = DataLoader(Subset(val_set, val_indices.tolist()), shuffle=False, generator=torch.Generator().manual_seed(seed + 1), **options)
    test_loader = DataLoader(test_set, shuffle=False, generator=torch.Generator().manual_seed(seed + 2), **options)
    metadata = {'name': dataset_name, 'classes': classes, 'train_examples': len(train_indices), 'validation_examples': len(val_indices), 'test_examples': len(test_set), 'validation_fraction': args.validation_fraction, 'split_seed': seed, 'training_indices_sha256': hashlib.sha256(train_indices.tobytes()).hexdigest(), 'validation_indices_sha256': hashlib.sha256(val_indices.tobytes()).hexdigest(), 'normalization_mean': mean, 'normalization_std': std, 'augmentation': ['random_crop_32_padding_4', 'random_horizontal_flip']}
    return (train_loader, val_loader, test_loader, metadata)

@torch.no_grad()
def current_evaluate(model, loader, device):
    model.eval()
    losses = 0.0
    correct = count = 0
    for (inputs, labels) in loader:
        (inputs, labels) = (inputs.to(device), labels.to(device))
        logits = model(inputs)
        losses += float(F.cross_entropy(logits, labels, reduction='sum'))
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
    return {'loss': losses / count, 'accuracy': 100.0 * correct / count}

def current_train_one(args, architecture, dataset_name, optimizer_name, method, seed, directory):
    _current_set_seed(seed)
    (train_loader, val_loader, test_loader, data) = current_make_loaders(args, seed, dataset_name)
    _current_set_seed(seed)
    model = current_build_model(architecture, data['classes'], method, args.alpha, args.beta).to(args.device)
    initialization = current_initial_digest(model)
    config = ManifoldFlowConfig(rho_geo=args.rho_geo, beta_P=args.beta_p, lambda_S=args.lambda_s, K_geo=args.k_geo, lambda_min=args.alpha, lambda_max=args.beta, warmup_frac=args.warmup_frac)
    lr = args.lr_adam if optimizer_name == 'adam' else args.lr_sgd
    bundle = make_optimizers(model, optimizer_name, lr, args.max_epochs * len(train_loader), config, momentum=args.momentum, weight_decay=args.weight_decay)
    stopper = PlateauStopper(args.min_epochs, args.patience, args.min_delta, mode='max')
    metadata = {'architecture': architecture, 'dataset': data, 'method': method, 'accuracy_units': 'percent', 'optimizer': optimizer_name, 'lr': lr, 'seed': seed, 'mf_config': asdict(config), 'batch_size': args.batch_size, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'max_epochs': args.max_epochs, 'min_epochs': args.min_epochs, 'patience': args.patience, 'min_delta': args.min_delta, 'record_test_history': args.record_test_history, 'parameterized_modules': [name for (name, module) in model.named_modules() if isinstance(module, SpectrumLinear)], 'initial_realized_parameters_sha256': initialization, 'parameter_count': sum((value.numel() for value in model.parameters())), 'device': str(args.device), 'dtype': 'float32', 'torch_version': torch.__version__, 'hardware': torch.cuda.get_device_name(args.device) if args.device.type == 'cuda' else 'CPU', 'cifar_stem': '3x3_stride1_no_maxpool' if architecture != 'simplecnn' else None, 'simplecnn_channels': [64, 128, 256] if architecture == 'simplecnn' else None, 'classifier_selection': 'best_validation_accuracy', 'provenance': 'new_execution_of_current_protocol'}
    history = []
    best_state = None
    began = time.perf_counter()
    _current_set_seed(seed)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = count = 0
        for (inputs, labels) in train_loader:
            (inputs, labels) = (inputs.to(args.device), labels.to(args.device))
            bundle.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('training loss is nonfinite')
            loss.backward()
            bundle.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.detach().argmax(1) == labels).sum())
            count += len(labels)
        validation = current_evaluate(model, val_loader, args.device)
        previous_best = stopper.best_epoch
        should_stop = stopper.update(validation['accuracy'], epoch)
        if stopper.best_epoch != previous_best:
            best_state = {name: value.detach().cpu().clone() for (name, value) in model.state_dict().items()}
        record = {'epoch': epoch, 'train_loss': loss_sum / count, 'train_accuracy': 100.0 * correct / count, 'validation': validation, 'diagnostics': bundle.diagnostics()}
        if args.record_test_history:
            record['test'] = current_evaluate(model, test_loader, args.device)
        history.append(record)
        write_json(directory / 'result.json', {'status': 'running', 'config': metadata, 'history': history})
        print(f"{architecture}/{dataset_name} {optimizer_name}/{method} seed={seed} epoch={epoch} val_acc={validation['accuracy']:.4f}", flush=True)
        if should_stop:
            break
    final_validation = current_evaluate(model, val_loader, args.device)
    final_test = current_evaluate(model, test_loader, args.device)
    final_diagnostics = bundle.diagnostics()
    if args.save_models:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), directory / 'final_model.pt')
    model.load_state_dict(best_state)
    selected_test = current_evaluate(model, test_loader, args.device)
    selected_diagnostics = bundle.diagnostics(include_gradient_metrics=False)
    if args.save_models:
        torch.save(model.state_dict(), directory / 'best_validation_model.pt')
    result = {'status': 'completed', 'config': metadata, 'history': history, 'epochs_run': len(history), 'stopping_reason': 'validation_plateau' if should_stop else 'max_epochs', 'best_validation_epoch': stopper.best_epoch, 'best_validation_accuracy': stopper.best_value, 'test_at_best_validation': selected_test, 'final_validation': final_validation, 'final_test': final_test, 'final_diagnostics': final_diagnostics, 'selected_diagnostics': selected_diagnostics, 'elapsed_seconds': time.perf_counter() - began}
    write_json(directory / 'result.json', result)
    return {key: result[key] for key in ('epochs_run', 'best_validation_epoch', 'best_validation_accuracy', 'test_at_best_validation', 'final_validation', 'final_test')}

def current_parser():
    cli = argparse.ArgumentParser()
    cli.add_argument('--settings', nargs='+', choices=['resnet50_cifar100', 'resnet18_cifar10', 'simplecnn_cifar10'], default=['resnet50_cifar100', 'resnet18_cifar10', 'simplecnn_cifar10'])
    cli.add_argument('--methods', nargs='+', choices=['dense', 'fs', 'scalar', 'diagonal', 'mf'], default=['fs', 'mf'])
    cli.add_argument('--optimizers', nargs='+', choices=['adam', 'sgd'], default=['adam', 'sgd'])
    cli.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7, 2024, 2025])
    cli.add_argument('--device', default='auto')
    cli.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'raw')
    cli.add_argument('--download', action='store_true')
    cli.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'image_classifiers')
    cli.add_argument('--max-epochs', type=int, default=1000)
    cli.add_argument('--min-epochs', type=int, default=300)
    cli.add_argument('--patience', type=int, default=100)
    cli.add_argument('--min-delta', type=float, default=0.0)
    cli.add_argument('--validation-fraction', type=float, default=0.15)
    cli.add_argument('--batch-size', type=int, default=128)
    cli.add_argument('--workers', type=int, default=0)
    cli.add_argument('--lr-adam', type=float, default=0.001)
    cli.add_argument('--lr-sgd', type=float, default=0.01)
    cli.add_argument('--momentum', type=float, default=0.9)
    cli.add_argument('--weight-decay', type=float, default=0.0001)
    cli.add_argument('--alpha', type=float, default=0.25)
    cli.add_argument('--beta', type=float, default=4.0)
    cli.add_argument('--rho-geo', type=float, default=0.003)
    cli.add_argument('--beta-p', type=float, default=0.95)
    cli.add_argument('--lambda-s', type=float, default=0.001)
    cli.add_argument('--k-geo', type=int, default=10)
    cli.add_argument('--warmup-frac', type=float, default=0.05)
    cli.add_argument('--record-test-history', action='store_true')
    cli.add_argument('--save-models', action='store_true')
    return cli

def current_main():
    cli = current_parser()
    args = cli.parse_args()
    if not 1 <= args.min_epochs <= args.max_epochs or args.batch_size < 1 or args.workers < 0:
        cli.error('invalid epoch, batch-size, or worker configuration')
    if not 0.0 < args.validation_fraction < 1.0:
        cli.error('validation fraction must be in (0, 1)')
    args.device = resolve_device(args.device)
    torch.set_default_dtype(torch.float32)
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    output = args.output_dir / session
    summary = {}
    for setting in args.settings:
        (architecture, dataset_name) = setting.split('_')
        for optimizer_name in args.optimizers:
            for seed in args.seeds:
                for method in args.methods:
                    key = f'{setting}/{optimizer_name}/seed_{seed}/{method}'
                    summary[key] = current_train_one(args, architecture, dataset_name, optimizer_name, method, seed, output / key)
                    write_json(output / 'summary.json', summary)
    print(str(output), flush=True)

def main():
    if "--legacy" in sys.argv[1:]:
        sys.argv.remove("--legacy")
        legacy_main()
    else:
        current_main()


if __name__ == "__main__":
    main()
