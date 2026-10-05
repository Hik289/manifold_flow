#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

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

class QDiagLinear(nn.Module):

    def __init__(self, in_dim, out_dim, seed=None):
        super().__init__()
        if out_dim <= in_dim:
            n, r = in_dim, out_dim; self.transpose = True
        else:
            n, r = out_dim, in_dim; self.transpose = False
        self.n, self.r = n, r
        self.Q     = nn.Parameter(_qr_init(n, r, seed))
        self.log_s = nn.Parameter(torch.zeros(r))
        self.bias  = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        s     = torch.exp(self.log_s)
        W_base = self.Q * s.unsqueeze(0)
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

class DenseMLP(nn.Module):

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

class SpectralNormMLP(nn.Module):

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

def legacy_main():
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

import argparse
import gc
import hashlib
import math
import platform
import statistics
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
from manifoldflow.parametrization import SpectrumLinear, make_optimizers, resolve_device, set_seed as _current_set_seed, write_json
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig

class ScalingModel(nn.Module):

    def __init__(self, input_dim, backbone_width, backbone_depth, p, r, mode, alpha, beta):
        super().__init__()
        dimensions = [input_dim] + [backbone_width] * backbone_depth + [p]
        layers = []
        for (input_width, output_width) in zip(dimensions[:-1], dimensions[1:]):
            layers.extend((nn.Linear(input_width, output_width), nn.ReLU()))
        self.backbone = nn.Sequential(*layers)
        self.spectrum = SpectrumLinear(p, r, mode, alpha=alpha, beta=beta)
        self.readout_scale = 1.0 / math.sqrt(r)

    def forward(self, inputs):
        features = self.spectrum(self.backbone(inputs))
        return features.sum(dim=-1) * self.readout_scale

def scaling_tensor_digest(tensors):
    digest = hashlib.sha256()
    for (name, tensor) in sorted(tensors.items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(cpu_tensor.shape)).encode('utf-8'))
        digest.update(str(cpu_tensor.dtype).encode('utf-8'))
        digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()

def scaling_make_model(args, r, mode):
    return ScalingModel(args.input_dim, args.backbone_width, args.backbone_depth, args.p, r, mode, args.alpha, args.beta)

def scaling_training_step(model, optimizers, inputs, targets):
    optimizers.zero_grad()
    predictions = model(inputs)
    loss = F.mse_loss(predictions, targets)
    loss.backward()
    optimizers.step()
    return loss.detach()

def scaling_measure_mode(args, device, r, mode, initial_state, initial_digest, inputs, targets, config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    _current_set_seed(args.seed)
    model = scaling_make_model(args, r, mode).to(dtype=torch.float32)
    loaded_state = model.load_state_dict(initial_state, strict=False)
    expected_missing = {'spectrum.gram_S', 'spectrum.sqrt_S'} if mode == 'mf' else set()
    if set(loaded_state.missing_keys) != expected_missing or loaded_state.unexpected_keys:
        raise RuntimeError('Unexpected initial-state differences between FS and MF')
    if scaling_tensor_digest(dict(model.named_parameters())) != initial_digest:
        raise RuntimeError('FS/MF initial parameters do not match')
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    optimizers = make_optimizers(model, args.optimizer, args.lr, args.warmup + args.steps, config, momentum=args.momentum, weight_decay=args.weight_decay, collect_diagnostics=False)
    for _ in range(args.warmup):
        torch.cuda.synchronize(device)
        loss = scaling_training_step(model, optimizers, inputs, targets)
        torch.cuda.synchronize(device)
    if args.warmup and (not torch.isfinite(loss).item()):
        raise RuntimeError(f'Non-finite loss during {mode.upper()} warm-up at r={r}')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before_measurement = torch.cuda.memory_allocated(device)
    durations_ms = []
    for _ in range(args.steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = scaling_training_step(model, optimizers, inputs, targets)
        torch.cuda.synchronize(device)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    final_loss = float(loss.item())
    if not math.isfinite(final_loss):
        raise RuntimeError(f'Non-finite loss during {mode.upper()} measurement at r={r}')
    ordered_durations = sorted(durations_ms)
    result = {'mode': mode, 'measured_steps': args.steps, 'mean_step_time_ms': statistics.fmean(durations_ms), 'median_step_time_ms': statistics.median(durations_ms), 'step_time_standard_deviation_ms': statistics.pstdev(durations_ms), 'minimum_step_time_ms': ordered_durations[0], 'maximum_step_time_ms': ordered_durations[-1], 'p95_step_time_ms': ordered_durations[math.ceil(0.95 * len(ordered_durations)) - 1], 'peak_allocated_gpu_memory_bytes': peak_allocated_bytes, 'peak_allocated_gpu_memory_GB': peak_allocated_bytes / 1000000000.0, 'allocated_gpu_memory_before_measurement_bytes': int(allocated_before_measurement), 'trainable_parameters': sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad)), 'registered_buffer_elements': sum((buffer.numel() for buffer in model.buffers())), 'constrained_optimizer_classes': [type(optimizer).__name__ for optimizer in optimizers.constrained], 'initial_parameter_sha256': initial_digest, 'final_loss': final_loss, 'pressure_logging': False, 'diagnostic_collection': False}
    if args.save_step_times:
        result['step_times_ms'] = durations_ms
    return result

def scaling_parse_args():
    parser = argparse.ArgumentParser(description='Measure paired FS/MF training-step runtime and CUDA allocated memory.')
    parser.add_argument('--r', nargs='+', type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument('--p', type=int, default=1024)
    parser.add_argument('--input-dim', type=int, default=128)
    parser.add_argument('--backbone-width', type=int, default=2048)
    parser.add_argument('--backbone-depth', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--rho-geo', type=float, default=0.01)
    parser.add_argument('--beta-p', type=float, default=0.95)
    parser.add_argument('--lambda-s', type=float, default=0.001)
    parser.add_argument('--k-geo', type=int, default=10)
    parser.add_argument('--tau-c', type=float, default=0.1)
    parser.add_argument('--tau-r', type=float, default=0.0)
    parser.add_argument('--alpha-c', type=float, default=5.0)
    parser.add_argument('--alpha-r', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--beta', type=float, default=4.0)
    parser.add_argument('--geometry-warmup-fraction', type=float, default=0.05)
    parser.add_argument('--save-step-times', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.r or any((r <= 0 for r in args.r)) or len(set(args.r)) != len(args.r):
        parser.error('--r requires distinct positive dimensions')
    if args.p < max(args.r):
        parser.error('--p must be at least every requested Stiefel dimension')
    if min(args.input_dim, args.backbone_width, args.batch_size, args.steps, args.k_geo) <= 0:
        parser.error('Model dimensions, batch size, measured steps, and K_geo must be positive')
    if args.backbone_depth < 0 or args.warmup < 0:
        parser.error('Backbone depth and timing warm-up must be nonnegative')
    if not 0 <= args.seed < 2 ** 32:
        parser.error('Seed must be in [0, 2**32)')
    if not all((math.isfinite(value) for value in (args.lr, args.momentum, args.weight_decay, args.rho_geo, args.beta_p, args.lambda_s, args.tau_c, args.tau_r, args.alpha_c, args.alpha_r, args.alpha, args.beta, args.geometry_warmup_fraction))):
        parser.error('Floating-point settings must be finite')
    if args.lr <= 0 or args.rho_geo <= 0 or args.lambda_s < 0 or (args.weight_decay < 0):
        parser.error('Learning rates must be positive and regularization must be nonnegative')
    if not 0 <= args.momentum < 1 or not 0 <= args.beta_p < 1:
        parser.error('Momentum and pressure EMA decay must be in [0, 1)')
    if args.alpha_c <= 0 or args.alpha_r <= 0:
        parser.error('Gate slopes must be positive')
    if not 0 < args.alpha <= 1 <= args.beta:
        parser.error('Spectral bounds must satisfy 0 < alpha <= 1 <= beta')
    if not 0 <= args.geometry_warmup_fraction <= 1:
        parser.error('Geometry warm-up fraction must be in [0, 1]')
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        args.output = REPOSITORY_ROOT / 'outputs' / 'computational_scaling' / f'scaling_{timestamp}_{uuid.uuid4().hex[:8]}.json'
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    try:
        requested_device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(f'Invalid device: {exc}')
    if requested_device.type != 'cuda':
        parser.error('This benchmark requires CUDA because it measures allocated GPU memory')
    return args

def scaling_main():
    args = scaling_parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; computational scaling was not measured')
    device = resolve_device(args.device)
    if device.type != 'cuda':
        raise SystemExit('A CUDA device is required; CPU fallback cannot measure allocated GPU memory')
    device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    config = ManifoldFlowConfig(rho_geo=args.rho_geo, beta_P=args.beta_p, lambda_S=args.lambda_s, K_geo=args.k_geo, tau_c=args.tau_c, tau_r=args.tau_r, alpha_c=args.alpha_c, alpha_r=args.alpha_r, lambda_min=args.alpha, lambda_max=args.beta, warmup_frac=args.geometry_warmup_fraction)
    generator = torch.Generator(device='cpu').manual_seed(args.seed + 1000003)
    input_cpu = torch.randn(args.batch_size, args.input_dim, generator=generator, dtype=torch.float32)
    target_cpu = torch.randn(args.batch_size, generator=generator, dtype=torch.float32)
    input_digest = scaling_tensor_digest({'inputs': input_cpu, 'targets': target_cpu})
    inputs = input_cpu.to(device=device)
    targets = target_cpu.to(device=device)
    properties = torch.cuda.get_device_properties(device)
    cuda_index = torch.cuda.current_device()
    result = {'benchmark': 'computational_scaling', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'paper_table_reproduction': False, 'architecture_note': 'The paper does not specify the benchmark backbone. This configurable synthetic regression network measures the implemented FS/MF pair. Backbone dimensions and p remain fixed within this sweep; the tested layer output dimension, Q column dimension, and SPD dimension change with r.', 'hardware': {'device': f'cuda:{cuda_index}', 'gpu_name': properties.name, 'gpu_total_memory_bytes': properties.total_memory, 'gpu_compute_capability': [properties.major, properties.minor], 'gpu_multiprocessor_count': properties.multi_processor_count, 'visible_cuda_device_count': torch.cuda.device_count(), 'platform': platform.platform(), 'python_version': platform.python_version(), 'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version()}, 'protocol': {'precision': 'FP32', 'autocast': False, 'tf32': False, 'batch_size': args.batch_size, 'timing_warmup_iterations': args.warmup, 'measured_training_iterations': args.steps, 'timer': 'time.perf_counter', 'synchronization': 'CUDA synchronization immediately before and after each complete training step', 'timed_operations': ['zero_grad', 'forward', 'MSE loss', 'backward', 'optimizer step', 'post-step spectrum cache update'], 'peak_memory': 'torch.cuda.max_memory_allocated after reset_peak_memory_stats following timing warm-up; includes model, inputs, gradients, optimizer state, and temporary allocations', 'time_unit': 'ms', 'memory_base_unit': 'bytes', 'memory_display_unit': 'GB, where 1 GB = 1,000,000,000 bytes', 'optimizer': args.optimizer, 'learning_rate': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'geometry_config': asdict(config), 'geometry_implementation': 'Pressure EMA aligned using the previous Q; anisotropy-damped gate; affine-invariant SPD update with the log-spectrum regularizer', 'geometry_warmup_iterations': math.ceil(config.warmup_frac * (args.warmup + args.steps)), 'fixed_spectrum_optimizer': 'FixedStiefelOptimizer', 'learnable_spectrum_optimizer': 'ManifoldFlowOptimizer', 'paired_tangent_update': 'The same tangent SGD or Adam implementation and hyperparameters are used by both modes', 'pressure_logging': False, 'diagnostic_collection': False, 'seed': args.seed, 'input_and_target_sha256': input_digest, 'data': 'One seeded synthetic regression batch held on the selected CUDA device and reused for every step and both modes at all r dimensions; no data loading is timed', 'mode_order': 'FS then MF for even sweep positions, MF then FS for odd sweep positions'}, 'architecture': {'input_dim': args.input_dim, 'backbone_hidden_width': args.backbone_width, 'backbone_hidden_depth': args.backbone_depth, 'backbone_linear_layer_count': args.backbone_depth + 1, 'tested_spectrum_linear_layer_count': 1, 'total_linear_layer_count': args.backbone_depth + 2, 'backbone_linear_dimensions': [args.input_dim] + [args.backbone_width] * args.backbone_depth + [args.p], 'activation_after_each_backbone_linear': 'ReLU', 'p': args.p, 'r_values': args.r, 'tested_linear_map': 'SpectrumLinear(in_features=p, out_features=r)', 'matrix_orientation': 'co_stiefel when r < p; stiefel when r = p', 'Q_shape': 'p x r', 'realized_weight_shape': 'r x p', 'spectral_identity': 'For the realized PyTorch weight, W @ W.T = S when r < p and W.T @ W = S when r = p; Q.T @ Q = I', 'readout': 'sum of the r tested-layer outputs divided by sqrt(r), with no trainable readout parameters', 'loss': 'mean squared error on a scalar target per example'}, 'measurements': []}
    for (index, r) in enumerate(args.r):
        _current_set_seed(args.seed)
        initial_model = scaling_make_model(args, r, 'fs').to(dtype=torch.float32)
        initial_state = {name: tensor.detach().cpu().clone() for (name, tensor) in initial_model.state_dict().items()}
        initial_digest = scaling_tensor_digest(dict(initial_model.named_parameters()))
        del initial_model
        order = ('fs', 'mf') if index % 2 == 0 else ('mf', 'fs')
        paired_results = {}
        for mode in order:
            paired_results[mode] = scaling_measure_mode(args, device, r, mode, initial_state, initial_digest, inputs, targets, config)
            measurement = paired_results[mode]
            print(f"r={r} {mode.upper()}: {measurement['mean_step_time_ms']:.4f} ms/step, {measurement['peak_allocated_gpu_memory_GB']:.4f} GB allocated", flush=True)
        fs = paired_results['fs']
        mf = paired_results['mf']
        result['measurements'].append({'r': r, 'p': args.p, 'Q_shape': [args.p, r], 'S_shape': [r, r], 'realized_weight_shape': [r, args.p], 'matrix_orientation': 'co_stiefel' if r < args.p else 'stiefel', 'spectral_identity': 'W @ W.T = S' if r < args.p else 'W.T @ W = S', 'measurement_order': list(order), 'fs': fs, 'mf': mf, 'time_ratio_mf_over_fs': mf['mean_step_time_ms'] / fs['mean_step_time_ms'], 'memory_ratio_mf_over_fs': mf['peak_allocated_gpu_memory_bytes'] / fs['peak_allocated_gpu_memory_bytes']})
        gc.collect()
        torch.cuda.empty_cache()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing result: {args.output}')
    write_json(args.output, result)
    print(f'Saved measured computational scaling results to {args.output}', flush=True)

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
TASKS = ('sequence', 'spectrum', 'mlp', 'cnn', 'stability', 'ablations', 'transformer', 'scaling')

def current_build_jobs(args):
    chosen = TASKS if 'all' in args.tasks else args.tasks
    seeds = args.seeds if args.seeds is not None else [42, 123, 7, 2024, 2025]
    ordinary = ['--device', args.device, '--seeds', *map(str, seeds)]
    protocol = ['--max-epochs', str(args.max_epochs), '--min-epochs', str(args.min_epochs), '--patience', str(args.patience)]
    sequence_data = []
    if args.sequence_data_dir is not None:
        sequence_data += ['--data-dir', str(args.sequence_data_dir)]
    if args.hf_cache is not None:
        sequence_data += ['--hf-cache', str(args.hf_cache)]
    classification_data = ['--data-dir', str(args.data_dir)] if args.data_dir is not None else []
    if args.download:
        classification_data.append('--download')
    output_root = args.output_dir if args.output_dir is not None else ROOT / 'outputs' / 'current_protocol'
    jobs = []

    def add(task, script, options):
        jobs.append({'task': task, 'command': [args.python, str(ROOT / script), *options]})
    if 'sequence' in chosen:
        for (architecture, dataset, hidden) in (('lstm', 'wikitext2', 128), ('lstm', 'wikitext2', 256), ('gru', 'wikitext2', 128), ('lstm', 'wikitext103', 128), ('gru', 'wikitext2', 256), ('gru', 'wikitext103', 128)):
            add('sequence', 'experiments/b12_lstm_5seeds.py', [*ordinary, *protocol, *sequence_data, '--architectures', architecture, '--datasets', dataset, '--hidden-sizes', str(hidden), '--methods', 'fs', 'mf', '--optimizers', 'adam', 'sgd', '--output-dir', str(output_root / 'sequence')])
    if 'spectrum' in chosen:
        add('spectrum', 'experiments/b14_lstm_baselines.py', [*ordinary, *protocol, *sequence_data, '--architectures', 'lstm', '--datasets', 'wikitext2', '--hidden-sizes', '128', '--methods', 'dense', 'fs', 'scalar', 'diagonal', 'mf', '--optimizers', 'adam', 'sgd', '--output-dir', str(output_root / 'spectrum')])
    if 'mlp' in chosen:
        add('mlp', 'experiments/mlp_batch9.py', [*ordinary, *protocol, *classification_data, '--datasets', 'adult', 'covertype', 'fashionmnist', 'cifar10', 'wine', '--methods', 'fs', 'mf', '--optimizers', 'adam', 'sgd', '--record-test-history', '--output-dir', str(output_root / 'mlp')])
    if 'cnn' in chosen:
        options = [*ordinary, *protocol, *classification_data, '--record-test-history', '--output-dir', str(output_root / 'cnn')]
        add('cnn', 'experiments/cifar100_layer_selective.py', options)
    if 'stability' in chosen:
        add('stability', 'experiments/mlp_batch9.py', ['--device', args.device, '--seeds', str(seeds[0]), *classification_data, '--datasets', 'adult', '--methods', 'dense', 'fs', 'mf', '--optimizers', 'adam', '--max-epochs', '500', '--min-epochs', '500', '--patience', '500', '--disable-early-stopping', '--record-test-history', '--output-dir', str(output_root / 'stability')])
    if 'ablations' in chosen:
        for (dataset, epochs) in (('adult', 100), ('covertype', 80)):
            add('ablations', 'experiments/mlp_batch9.py', [*ordinary, *classification_data, '--datasets', dataset, '--methods', 'mf', '--optimizers', 'adam', '--mf-ablations', 'full', 'no_ema', 'no_gate', 'random_pressure', '--max-epochs', str(epochs), '--min-epochs', str(epochs), '--patience', str(epochs), '--disable-early-stopping', '--output-dir', str(output_root / 'ablations')])
    if 'transformer' in chosen:
        transformer_seeds = args.seeds if args.seeds is not None else [42, 123, 2024]
        add('transformer', 'experiments/transformer_wikitext_b10.py', ['--device', args.device, '--seeds', *map(str, transformer_seeds), *sequence_data, '--architectures', 'transformer', '--datasets', 'wikitext2', '--hidden-sizes', '128', '--methods', 'fs', 'mf', '--optimizers', 'adam', 'sgd', '--seq-len', '128', '--max-epochs', '12', '--min-epochs', '1', '--patience', '12', '--output-dir', str(output_root / 'transformer')])
    if 'scaling' in chosen:
        add('scaling', 'experiments/e2_baselines.py', ['--scaling', '--device', 'cuda' if args.device == 'auto' else args.device, '--seed', str(seeds[0]), '--output', str(output_root / 'scaling' / f'scaling_{uuid4().hex}.json')])
    return jobs

def current_main():
    parser = argparse.ArgumentParser(description="Print the current manuscript's experiment commands; execute them only with --execute.")
    parser.add_argument('--tasks', nargs='+', choices=['all', *TASKS], default=['all'])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--min-epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--sequence-data-dir', type=Path)
    parser.add_argument('--hf-cache', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.min_epochs <= args.max_epochs or args.patience < 1:
        parser.error('invalid epoch or patience configuration')
    if ('all' in args.tasks or 'scaling' in args.tasks) and args.device != 'auto' and (not args.device.startswith('cuda')):
        parser.error('computational scaling requires a CUDA device')
    jobs = current_build_jobs(args)
    print(json.dumps({'paper_source': 'main-37.tex', 'execution_requested': args.execute, 'jobs': jobs}, indent=2), flush=True)
    if args.execute:
        environment = os.environ.copy()
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT), environment.get('PYTHONPATH', '')])
        for job in jobs:
            subprocess.run(job['command'], cwd=ROOT, env=environment, check=True)

def main():
    if "--legacy" in sys.argv[1:]:
        sys.argv.remove("--legacy")
        legacy_main()
    elif "--scaling" in sys.argv[1:]:
        sys.argv.remove("--scaling")
        scaling_main()
    else:
        current_main()


if __name__ == "__main__":
    main()
