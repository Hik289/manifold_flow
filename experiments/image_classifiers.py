from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from experiments.common import PlateauStopper, SpectrumLinear, make_optimizers, resolve_device, set_seed, write_json
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig


def build_model(architecture, classes, method, alpha, beta):
    if architecture == "simplecnn":
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(256 * 4 * 4, 128), nn.ReLU(),
            SpectrumLinear(128, classes, method, alpha, beta),
        )
    else:
        from torchvision.models import resnet18, resnet50
        factory = resnet18 if architecture == "resnet18" else resnet50
        model = factory(weights=None, num_classes=classes)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = SpectrumLinear(model.fc.in_features, classes, method, alpha, beta)
    return model


def initial_digest(model):
    digest = hashlib.sha256()
    excluded = set()
    for name, module in model.named_modules():
        if isinstance(module, SpectrumLinear):
            digest.update(name.encode())
            digest.update(module.effective_weight().detach().cpu().numpy().tobytes())
            if module.bias is not None:
                digest.update(module.bias.detach().cpu().numpy().tobytes())
            excluded.update(id(value) for value in module.parameters())
    for name, parameter in model.named_parameters():
        if id(parameter) not in excluded:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_loaders(args, seed, dataset_name):
    from sklearn.model_selection import train_test_split
    from torchvision import datasets, transforms
    if dataset_name == "cifar10":
        factory, classes = datasets.CIFAR10, 10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        factory, classes = datasets.CIFAR100, 100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    training = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_set = factory(args.data_dir, train=True, transform=training, download=args.download)
    val_set = factory(args.data_dir, train=True, transform=evaluation, download=args.download)
    test_set = factory(args.data_dir, train=False, transform=evaluation, download=args.download)
    train_indices, val_indices = train_test_split(
        np.arange(len(train_set)), test_size=args.validation_fraction,
        random_state=seed, stratify=np.asarray(train_set.targets),
    )
    generator = torch.Generator().manual_seed(seed)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.device.type == "cuda")
    train_loader = DataLoader(Subset(train_set, train_indices.tolist()), shuffle=True, generator=generator, **options)
    val_loader = DataLoader(Subset(val_set, val_indices.tolist()), shuffle=False,
                            generator=torch.Generator().manual_seed(seed + 1), **options)
    test_loader = DataLoader(test_set, shuffle=False,
                             generator=torch.Generator().manual_seed(seed + 2), **options)
    metadata = {
        "name": dataset_name, "classes": classes, "train_examples": len(train_indices),
        "validation_examples": len(val_indices), "test_examples": len(test_set),
        "validation_fraction": args.validation_fraction, "split_seed": seed,
        "training_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest(),
        "normalization_mean": mean, "normalization_std": std,
        "augmentation": ["random_crop_32_padding_4", "random_horizontal_flip"],
    }
    return train_loader, val_loader, test_loader, metadata


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = 0.0
    correct = count = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        losses += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
    return {"loss": losses / count, "accuracy": 100.0 * correct / count}


def train_one(args, architecture, dataset_name, optimizer_name, method, seed, directory):
    set_seed(seed)
    train_loader, val_loader, test_loader, data = make_loaders(args, seed, dataset_name)
    set_seed(seed)
    model = build_model(architecture, data["classes"], method, args.alpha, args.beta).to(args.device)
    initialization = initial_digest(model)
    config = ManifoldFlowConfig(rho_geo=args.rho_geo, beta_P=args.beta_p, lambda_S=args.lambda_s,
                                K_geo=args.k_geo, lambda_min=args.alpha, lambda_max=args.beta,
                                warmup_frac=args.warmup_frac)
    lr = args.lr_adam if optimizer_name == "adam" else args.lr_sgd
    bundle = make_optimizers(model, optimizer_name, lr, args.max_epochs * len(train_loader), config,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    stopper = PlateauStopper(args.min_epochs, args.patience, args.min_delta, mode="max")
    metadata = {
        "architecture": architecture, "dataset": data, "method": method,
        "accuracy_units": "percent",
        "optimizer": optimizer_name, "lr": lr, "seed": seed, "mf_config": asdict(config),
        "batch_size": args.batch_size, "momentum": args.momentum, "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs, "min_epochs": args.min_epochs, "patience": args.patience,
        "min_delta": args.min_delta, "record_test_history": args.record_test_history,
        "parameterized_modules": [name for name, module in model.named_modules() if isinstance(module, SpectrumLinear)],
        "initial_realized_parameters_sha256": initialization,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "device": str(args.device), "dtype": "float32", "torch_version": torch.__version__,
        "hardware": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "CPU",
        "cifar_stem": "3x3_stride1_no_maxpool" if architecture != "simplecnn" else None,
        "simplecnn_channels": [64, 128, 256] if architecture == "simplecnn" else None,
        "classifier_selection": "best_validation_accuracy",
        "provenance": "new_execution_of_current_protocol",
    }
    history = []
    best_state = None
    began = time.perf_counter()
    set_seed(seed)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            bundle.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training loss is nonfinite")
            loss.backward()
            bundle.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.detach().argmax(1) == labels).sum())
            count += len(labels)
        validation = evaluate(model, val_loader, args.device)
        previous_best = stopper.best_epoch
        should_stop = stopper.update(validation["accuracy"], epoch)
        if stopper.best_epoch != previous_best:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        record = {"epoch": epoch, "train_loss": loss_sum / count, "train_accuracy": 100.0 * correct / count,
                  "validation": validation, "diagnostics": bundle.diagnostics()}
        if args.record_test_history:
            record["test"] = evaluate(model, test_loader, args.device)
        history.append(record)
        write_json(directory / "result.json", {"status": "running", "config": metadata, "history": history})
        print(f"{architecture}/{dataset_name} {optimizer_name}/{method} seed={seed} epoch={epoch} val_acc={validation['accuracy']:.4f}", flush=True)
        if should_stop:
            break
    final_validation = evaluate(model, val_loader, args.device)
    final_test = evaluate(model, test_loader, args.device)
    final_diagnostics = bundle.diagnostics()
    if args.save_models:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), directory / "final_model.pt")
    model.load_state_dict(best_state)
    selected_test = evaluate(model, test_loader, args.device)
    selected_diagnostics = bundle.diagnostics(include_gradient_metrics=False)
    if args.save_models:
        torch.save(model.state_dict(), directory / "best_validation_model.pt")
    result = {
        "status": "completed", "config": metadata, "history": history, "epochs_run": len(history),
        "stopping_reason": "validation_plateau" if should_stop else "max_epochs",
        "best_validation_epoch": stopper.best_epoch, "best_validation_accuracy": stopper.best_value,
        "test_at_best_validation": selected_test, "final_validation": final_validation, "final_test": final_test,
        "final_diagnostics": final_diagnostics, "selected_diagnostics": selected_diagnostics,
        "elapsed_seconds": time.perf_counter() - began,
    }
    write_json(directory / "result.json", result)
    return {key: result[key] for key in ("epochs_run", "best_validation_epoch", "best_validation_accuracy", "test_at_best_validation", "final_validation", "final_test")}


def parser():
    cli = argparse.ArgumentParser()
    cli.add_argument("--settings", nargs="+", choices=["resnet50_cifar100", "resnet18_cifar10", "simplecnn_cifar10"],
                     default=["resnet50_cifar100", "resnet18_cifar10", "simplecnn_cifar10"])
    cli.add_argument("--methods", nargs="+", choices=["dense", "fs", "scalar", "diagonal", "mf"], default=["fs", "mf"])
    cli.add_argument("--optimizers", nargs="+", choices=["adam", "sgd"], default=["adam", "sgd"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 2024, 2025])
    cli.add_argument("--device", default="auto")
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    cli.add_argument("--download", action="store_true")
    cli.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "image_classifiers")
    cli.add_argument("--max-epochs", type=int, default=1000)
    cli.add_argument("--min-epochs", type=int, default=300)
    cli.add_argument("--patience", type=int, default=100)
    cli.add_argument("--min-delta", type=float, default=0.0)
    cli.add_argument("--validation-fraction", type=float, default=0.15)
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--workers", type=int, default=0)
    cli.add_argument("--lr-adam", type=float, default=0.001)
    cli.add_argument("--lr-sgd", type=float, default=0.01)
    cli.add_argument("--momentum", type=float, default=0.9)
    cli.add_argument("--weight-decay", type=float, default=0.0001)
    cli.add_argument("--alpha", type=float, default=0.25)
    cli.add_argument("--beta", type=float, default=4.0)
    cli.add_argument("--rho-geo", type=float, default=0.003)
    cli.add_argument("--beta-p", type=float, default=0.95)
    cli.add_argument("--lambda-s", type=float, default=0.001)
    cli.add_argument("--k-geo", type=int, default=10)
    cli.add_argument("--warmup-frac", type=float, default=0.05)
    cli.add_argument("--record-test-history", action="store_true")
    cli.add_argument("--save-models", action="store_true")
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    if not 1 <= args.min_epochs <= args.max_epochs or args.batch_size < 1 or args.workers < 0:
        cli.error("invalid epoch, batch-size, or worker configuration")
    if not 0.0 < args.validation_fraction < 1.0:
        cli.error("validation fraction must be in (0, 1)")
    args.device = resolve_device(args.device)
    torch.set_default_dtype(torch.float32)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output_dir / session
    summary = {}
    for setting in args.settings:
        architecture, dataset_name = setting.split("_")
        for optimizer_name in args.optimizers:
            for seed in args.seeds:
                for method in args.methods:
                    key = f"{setting}/{optimizer_name}/seed_{seed}/{method}"
                    summary[key] = train_one(args, architecture, dataset_name, optimizer_name, method, seed, output / key)
                    write_json(output / "summary.json", summary)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
