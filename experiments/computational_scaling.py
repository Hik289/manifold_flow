from __future__ import annotations

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
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from experiments.common import SpectrumLinear, make_optimizers, resolve_device, set_seed, write_json
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig


class ScalingModel(nn.Module):
    def __init__(self, input_dim, backbone_width, backbone_depth, p, r, mode, alpha, beta):
        super().__init__()
        dimensions = [input_dim] + [backbone_width] * backbone_depth + [p]
        layers = []
        for input_width, output_width in zip(dimensions[:-1], dimensions[1:]):
            layers.extend((nn.Linear(input_width, output_width), nn.ReLU()))
        self.backbone = nn.Sequential(*layers)
        self.spectrum = SpectrumLinear(p, r, mode, alpha=alpha, beta=beta)
        self.readout_scale = 1.0 / math.sqrt(r)

    def forward(self, inputs):
        features = self.spectrum(self.backbone(inputs))
        return features.sum(dim=-1) * self.readout_scale


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
        digest.update(str(cpu_tensor.dtype).encode("utf-8"))
        digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()


def make_model(args, r, mode):
    return ScalingModel(
        args.input_dim,
        args.backbone_width,
        args.backbone_depth,
        args.p,
        r,
        mode,
        args.alpha,
        args.beta,
    )


def training_step(model, optimizers, inputs, targets):
    optimizers.zero_grad()
    predictions = model(inputs)
    loss = F.mse_loss(predictions, targets)
    loss.backward()
    optimizers.step()
    return loss.detach()


def measure_mode(args, device, r, mode, initial_state, initial_digest, inputs, targets, config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    set_seed(args.seed)
    model = make_model(args, r, mode).to(dtype=torch.float32)
    loaded_state = model.load_state_dict(initial_state, strict=False)
    expected_missing = {"spectrum.gram_S", "spectrum.sqrt_S"} if mode == "mf" else set()
    if set(loaded_state.missing_keys) != expected_missing or loaded_state.unexpected_keys:
        raise RuntimeError("Unexpected initial-state differences between FS and MF")
    if tensor_digest(dict(model.named_parameters())) != initial_digest:
        raise RuntimeError("FS/MF initial parameters do not match")
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    optimizers = make_optimizers(
        model,
        args.optimizer,
        args.lr,
        args.warmup + args.steps,
        config,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        collect_diagnostics=False,
    )
    for _ in range(args.warmup):
        torch.cuda.synchronize(device)
        loss = training_step(model, optimizers, inputs, targets)
        torch.cuda.synchronize(device)
    if args.warmup and not torch.isfinite(loss).item():
        raise RuntimeError(f"Non-finite loss during {mode.upper()} warm-up at r={r}")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before_measurement = torch.cuda.memory_allocated(device)
    durations_ms = []
    for _ in range(args.steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = training_step(model, optimizers, inputs, targets)
        torch.cuda.synchronize(device)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    final_loss = float(loss.item())
    if not math.isfinite(final_loss):
        raise RuntimeError(f"Non-finite loss during {mode.upper()} measurement at r={r}")
    ordered_durations = sorted(durations_ms)
    result = {
        "mode": mode,
        "measured_steps": args.steps,
        "mean_step_time_ms": statistics.fmean(durations_ms),
        "median_step_time_ms": statistics.median(durations_ms),
        "step_time_standard_deviation_ms": statistics.pstdev(durations_ms),
        "minimum_step_time_ms": ordered_durations[0],
        "maximum_step_time_ms": ordered_durations[-1],
        "p95_step_time_ms": ordered_durations[math.ceil(0.95 * len(ordered_durations)) - 1],
        "peak_allocated_gpu_memory_bytes": peak_allocated_bytes,
        "peak_allocated_gpu_memory_GB": peak_allocated_bytes / 1_000_000_000.0,
        "allocated_gpu_memory_before_measurement_bytes": int(allocated_before_measurement),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "registered_buffer_elements": sum(buffer.numel() for buffer in model.buffers()),
        "constrained_optimizer_classes": [type(optimizer).__name__ for optimizer in optimizers.constrained],
        "initial_parameter_sha256": initial_digest,
        "final_loss": final_loss,
        "pressure_logging": False,
        "diagnostic_collection": False,
    }
    if args.save_step_times:
        result["step_times_ms"] = durations_ms
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Measure paired FS/MF training-step runtime and CUDA allocated memory.")
    parser.add_argument("--r", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument("--p", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--backbone-width", type=int, default=2048)
    parser.add_argument("--backbone-depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rho-geo", type=float, default=0.01)
    parser.add_argument("--beta-p", type=float, default=0.95)
    parser.add_argument("--lambda-s", type=float, default=0.001)
    parser.add_argument("--k-geo", type=int, default=10)
    parser.add_argument("--tau-c", type=float, default=0.1)
    parser.add_argument("--tau-r", type=float, default=0.0)
    parser.add_argument("--alpha-c", type=float, default=5.0)
    parser.add_argument("--alpha-r", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--geometry-warmup-fraction", type=float, default=0.05)
    parser.add_argument("--save-step-times", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.r or any(r <= 0 for r in args.r) or len(set(args.r)) != len(args.r):
        parser.error("--r requires distinct positive dimensions")
    if args.p < max(args.r):
        parser.error("--p must be at least every requested Stiefel dimension")
    if min(args.input_dim, args.backbone_width, args.batch_size, args.steps, args.k_geo) <= 0:
        parser.error("Model dimensions, batch size, measured steps, and K_geo must be positive")
    if args.backbone_depth < 0 or args.warmup < 0:
        parser.error("Backbone depth and timing warm-up must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32)")
    if not all(math.isfinite(value) for value in (
        args.lr, args.momentum, args.weight_decay, args.rho_geo, args.beta_p,
        args.lambda_s, args.tau_c, args.tau_r, args.alpha_c, args.alpha_r,
        args.alpha, args.beta, args.geometry_warmup_fraction,
    )):
        parser.error("Floating-point settings must be finite")
    if args.lr <= 0 or args.rho_geo <= 0 or args.lambda_s < 0 or args.weight_decay < 0:
        parser.error("Learning rates must be positive and regularization must be nonnegative")
    if not 0 <= args.momentum < 1 or not 0 <= args.beta_p < 1:
        parser.error("Momentum and pressure EMA decay must be in [0, 1)")
    if args.alpha_c <= 0 or args.alpha_r <= 0:
        parser.error("Gate slopes must be positive")
    if not 0 < args.alpha <= 1 <= args.beta:
        parser.error("Spectral bounds must satisfy 0 < alpha <= 1 <= beta")
    if not 0 <= args.geometry_warmup_fraction <= 1:
        parser.error("Geometry warm-up fraction must be in [0, 1]")
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = REPOSITORY_ROOT / "outputs" / "computational_scaling" / f"scaling_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        requested_device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(f"Invalid device: {exc}")
    if requested_device.type != "cuda":
        parser.error("This benchmark requires CUDA because it measures allocated GPU memory")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; computational scaling was not measured")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("A CUDA device is required; CPU fallback cannot measure allocated GPU memory")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    config = ManifoldFlowConfig(
        rho_geo=args.rho_geo,
        beta_P=args.beta_p,
        lambda_S=args.lambda_s,
        K_geo=args.k_geo,
        tau_c=args.tau_c,
        tau_r=args.tau_r,
        alpha_c=args.alpha_c,
        alpha_r=args.alpha_r,
        lambda_min=args.alpha,
        lambda_max=args.beta,
        warmup_frac=args.geometry_warmup_fraction,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1_000_003)
    input_cpu = torch.randn(args.batch_size, args.input_dim, generator=generator, dtype=torch.float32)
    target_cpu = torch.randn(args.batch_size, generator=generator, dtype=torch.float32)
    input_digest = tensor_digest({"inputs": input_cpu, "targets": target_cpu})
    inputs = input_cpu.to(device=device)
    targets = target_cpu.to(device=device)
    properties = torch.cuda.get_device_properties(device)
    cuda_index = torch.cuda.current_device()
    result = {
        "benchmark": "computational_scaling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_table_reproduction": False,
        "architecture_note": "The paper does not specify the benchmark backbone. This configurable synthetic regression network measures the implemented FS/MF pair. Backbone dimensions and p remain fixed within this sweep; the tested layer output dimension, Q column dimension, and SPD dimension change with r.",
        "hardware": {
            "device": f"cuda:{cuda_index}",
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": [properties.major, properties.minor],
            "gpu_multiprocessor_count": properties.multi_processor_count,
            "visible_cuda_device_count": torch.cuda.device_count(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "protocol": {
            "precision": "FP32",
            "autocast": False,
            "tf32": False,
            "batch_size": args.batch_size,
            "timing_warmup_iterations": args.warmup,
            "measured_training_iterations": args.steps,
            "timer": "time.perf_counter",
            "synchronization": "CUDA synchronization immediately before and after each complete training step",
            "timed_operations": ["zero_grad", "forward", "MSE loss", "backward", "optimizer step", "post-step spectrum cache update"],
            "peak_memory": "torch.cuda.max_memory_allocated after reset_peak_memory_stats following timing warm-up; includes model, inputs, gradients, optimizer state, and temporary allocations",
            "time_unit": "ms",
            "memory_base_unit": "bytes",
            "memory_display_unit": "GB, where 1 GB = 1,000,000,000 bytes",
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "geometry_config": asdict(config),
            "geometry_implementation": "Pressure EMA aligned using the previous Q; anisotropy-damped gate; affine-invariant SPD update with the log-spectrum regularizer",
            "geometry_warmup_iterations": math.ceil(config.warmup_frac * (args.warmup + args.steps)),
            "fixed_spectrum_optimizer": "FixedStiefelOptimizer",
            "learnable_spectrum_optimizer": "ManifoldFlowOptimizer",
            "paired_tangent_update": "The same tangent SGD or Adam implementation and hyperparameters are used by both modes",
            "pressure_logging": False,
            "diagnostic_collection": False,
            "seed": args.seed,
            "input_and_target_sha256": input_digest,
            "data": "One seeded synthetic regression batch held on the selected CUDA device and reused for every step and both modes at all r dimensions; no data loading is timed",
            "mode_order": "FS then MF for even sweep positions, MF then FS for odd sweep positions",
        },
        "architecture": {
            "input_dim": args.input_dim,
            "backbone_hidden_width": args.backbone_width,
            "backbone_hidden_depth": args.backbone_depth,
            "backbone_linear_layer_count": args.backbone_depth + 1,
            "tested_spectrum_linear_layer_count": 1,
            "total_linear_layer_count": args.backbone_depth + 2,
            "backbone_linear_dimensions": [args.input_dim] + [args.backbone_width] * args.backbone_depth + [args.p],
            "activation_after_each_backbone_linear": "ReLU",
            "p": args.p,
            "r_values": args.r,
            "tested_linear_map": "SpectrumLinear(in_features=p, out_features=r)",
            "matrix_orientation": "co_stiefel when r < p; stiefel when r = p",
            "Q_shape": "p x r",
            "realized_weight_shape": "r x p",
            "spectral_identity": "For the realized PyTorch weight, W @ W.T = S when r < p and W.T @ W = S when r = p; Q.T @ Q = I",
            "readout": "sum of the r tested-layer outputs divided by sqrt(r), with no trainable readout parameters",
            "loss": "mean squared error on a scalar target per example",
        },
        "measurements": [],
    }
    for index, r in enumerate(args.r):
        set_seed(args.seed)
        initial_model = make_model(args, r, "fs").to(dtype=torch.float32)
        initial_state = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
        initial_digest = tensor_digest(dict(initial_model.named_parameters()))
        del initial_model
        order = ("fs", "mf") if index % 2 == 0 else ("mf", "fs")
        paired_results = {}
        for mode in order:
            paired_results[mode] = measure_mode(args, device, r, mode, initial_state, initial_digest, inputs, targets, config)
            measurement = paired_results[mode]
            print(f"r={r} {mode.upper()}: {measurement['mean_step_time_ms']:.4f} ms/step, {measurement['peak_allocated_gpu_memory_GB']:.4f} GB allocated", flush=True)
        fs = paired_results["fs"]
        mf = paired_results["mf"]
        result["measurements"].append({
            "r": r,
            "p": args.p,
            "Q_shape": [args.p, r],
            "S_shape": [r, r],
            "realized_weight_shape": [r, args.p],
            "matrix_orientation": "co_stiefel" if r < args.p else "stiefel",
            "spectral_identity": "W @ W.T = S" if r < args.p else "W.T @ W = S",
            "measurement_order": list(order),
            "fs": fs,
            "mf": mf,
            "time_ratio_mf_over_fs": mf["mean_step_time_ms"] / fs["mean_step_time_ms"],
            "memory_ratio_mf_over_fs": mf["peak_allocated_gpu_memory_bytes"] / fs["peak_allocated_gpu_memory_bytes"],
        })
        gc.collect()
        torch.cuda.empty_cache()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {args.output}")
    write_json(args.output, result)
    print(f"Saved measured computational scaling results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
