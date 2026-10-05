from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import PlateauStopper, SpectrumLinear, make_optimizers, resolve_device, set_seed, write_json
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig


def dataset_name(value):
    name = value.lower().replace("-", "").replace("_", "")
    name = {"winequality": "wine", "fashionmnist": "fashionmnist"}.get(name, name)
    if name not in {"adult", "covertype", "fashionmnist", "cifar10", "wine"}:
        raise argparse.ArgumentTypeError("unknown MLP dataset")
    return name


def preprocess_tabular(train, validation, test):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    numeric = list(train.select_dtypes(include=[np.number]).columns)
    categorical = [column for column in train.columns if column not in numeric]
    transformations = []
    if numeric:
        transformations.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformations.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler()),
        ]), categorical))
    processor = ColumnTransformer(transformations)
    transformed = [processor.fit_transform(train), processor.transform(validation), processor.transform(test)]
    return [np.asarray(values, dtype=np.float32) for values in transformed], {
        "fit_split": "train", "numeric_columns": numeric, "categorical_columns": categorical,
        "categorical_encoding": "train-fitted ordinal; unseen categories map to -1",
        "numeric_imputation": "train median", "categorical_imputation": "train most frequent",
        "normalization": "train-fitted StandardScaler after imputation and encoding",
    }


def load_data(args, name, seed):
    from sklearn.model_selection import train_test_split
    metadata = {"dataset": name, "split_seed": seed, "validation_fraction": args.validation_fraction}
    if name in {"fashionmnist", "cifar10"}:
        from torchvision.datasets import CIFAR10, FashionMNIST
        factory = FashionMNIST if name == "fashionmnist" else CIFAR10
        training = factory(str(args.data_dir), train=True, download=args.download)
        testing = factory(str(args.data_dir), train=False, download=args.download)
        def flatten(dataset):
            values = np.asarray(dataset.data, dtype=np.float32) / 255.0
            if name == "cifar10":
                values = values.transpose(0, 3, 1, 2)
            return values.reshape(len(values), -1), np.asarray(dataset.targets, dtype=np.int64)
        features, labels = flatten(training)
        test_features, test_labels = flatten(testing)
        train_indices, val_indices = train_test_split(
            np.arange(len(labels)), test_size=args.validation_fraction, random_state=seed, stratify=labels)
        train_features, val_features = features[train_indices], features[val_indices]
        train_labels, val_labels = labels[train_indices], labels[val_indices]
        mean = train_features.mean(axis=0, dtype=np.float64)
        std = train_features.std(axis=0, dtype=np.float64)
        std = np.where(std > 0.0, std, 1.0)
        arrays = [(values - mean).astype(np.float32) / std.astype(np.float32)
                  for values in (train_features, val_features, test_features)]
        classes = [str(value) for value in range(10)]
        metadata.update({"source": "torchvision official train/test splits", "test_split": "official",
                         "normalization": "per-feature mean/std fitted only on selected training rows",
                         "flatten_order": "CHW" if name == "cifar10" else "HW"})
    else:
        import pandas as pd
        if name == "adult":
            from sklearn.datasets import fetch_openml
            cache = args.data_dir / "adult_v2.csv"
            if cache.is_file():
                frame = pd.read_csv(cache)
                raw_labels = frame.pop("_target").astype(str).to_numpy()
            else:
                if not args.download:
                    raise FileNotFoundError(f"Missing {cache}; allow --download to prepare the Adult cache")
                fetched = fetch_openml("adult", version=2, as_frame=True, data_home=str(args.data_dir), parser="auto")
                frame = fetched.data.copy()
                raw_labels = np.asarray(fetched.target.astype(str))
                cached_frame = frame.copy()
                cached_frame["_target"] = raw_labels
                cache.parent.mkdir(parents=True, exist_ok=True)
                cached_frame.to_csv(cache, index=False)
            metadata.update({"source": "OpenML Adult version 2", "cache_file": str(cache),
                             "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()})
        elif name == "covertype":
            from sklearn.datasets import fetch_covtype
            fetched = fetch_covtype(data_home=str(args.data_dir), download_if_missing=args.download)
            indices = np.arange(len(fetched.target))
            if args.covertype_samples > 0 and args.covertype_samples < len(indices):
                indices = np.random.default_rng(seed).choice(indices, args.covertype_samples, replace=False)
            frame = pd.DataFrame(fetched.data[indices])
            raw_labels = fetched.target[indices]
            metadata.update({"source": "scikit-learn UCI Covertype", "subset_rows": len(indices),
                             "subset_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
        else:
            paths = [args.wine_file] if args.wine_file is not None else [
                args.data_dir / f"winequality-{color}.csv"
                for color in (["red", "white"] if args.wine_color == "both" else [args.wine_color])]
            frames = []
            for path in paths:
                if not path.is_file():
                    if not args.download or args.wine_file is not None:
                        raise FileNotFoundError(f"Missing {path}; provide --wine-file or allow --download")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/" + path.name, path)
                frames.append(pd.read_csv(path, sep=";"))
            frame = pd.concat(frames, ignore_index=True)
            raw_labels = frame.pop("quality").to_numpy()
            if args.wine_binary_threshold is not None:
                raw_labels = (raw_labels >= args.wine_binary_threshold).astype(np.int64)
            metadata.update({"source": "UCI Wine Quality CSV", "wine_color": args.wine_color,
                             "files": [str(path) for path in paths],
                             "label_definition": "original quality classes" if args.wine_binary_threshold is None
                             else f"quality >= {args.wine_binary_threshold}", "exploratory": True})
        for column in frame.select_dtypes(exclude=[np.number]).columns:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan).replace("?", np.nan)
        classes, labels = np.unique(raw_labels.astype(str), return_inverse=True)
        train_val, test_indices = train_test_split(np.arange(len(labels)), test_size=args.test_fraction,
                                                   random_state=seed, stratify=labels)
        train_indices, val_indices = train_test_split(train_val, test_size=args.validation_fraction,
                                                      random_state=seed, stratify=labels[train_val])
        arrays, preprocessing = preprocess_tabular(frame.iloc[train_indices], frame.iloc[val_indices], frame.iloc[test_indices])
        train_labels, val_labels, test_labels = labels[train_indices], labels[val_indices], labels[test_indices]
        metadata.update({"test_split": "seeded stratified holdout", "test_fraction": args.test_fraction,
                         "preprocessing": preprocessing})
        classes = classes.tolist()
    labels_by_split = [train_labels, val_labels, test_labels]
    tensors = [TensorDataset(torch.from_numpy(values.astype(np.float32)), torch.from_numpy(targets.astype(np.int64)))
               for values, targets in zip(arrays, labels_by_split)]
    metadata.update({"classes": classes, "input_features": arrays[0].shape[1],
                     "split_sizes": dict(zip(("train", "validation", "test"), map(len, tensors))),
                     "train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
                     "validation_indices_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest()})
    return tensors, metadata


class MLP(nn.Module):
    def __init__(self, input_dim, classes, method, seed, args):
        super().__init__()
        layers = []
        width = input_dim
        for index in range(args.hidden_layers):
            torch.manual_seed(seed * 1000 + index)
            layers.extend([SpectrumLinear(width, args.hidden_size, method, args.alpha, args.beta), nn.ReLU()])
            width = args.hidden_size
        torch.manual_seed(seed * 1000 + args.hidden_layers)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, classes)

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        correct += int((logits.argmax(1) == targets).sum())
        count += len(targets)
    return {"loss": total_loss / count, "accuracy": 100.0 * correct / count}


def train_run(args, name, data, data_metadata, optimizer_name, method, ablation, seed, directory):
    set_seed(seed)
    batch_size = args.batch_size or (512 if name == "covertype" else 256)
    loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=index == 0,
                          generator=torch.Generator().manual_seed(seed + index), num_workers=0)
               for index, dataset in enumerate(data)]
    train_loader, val_loader, test_loader = loaders
    model = MLP(data_metadata["input_features"], len(data_metadata["classes"]), method, seed, args).to(args.device)
    config = ManifoldFlowConfig(rho_geo=args.rho_geo, beta_P=0.0 if ablation == "no_ema" else args.beta_p,
                                lambda_S=args.lambda_s, K_geo=args.k_geo, lambda_min=args.alpha,
                                lambda_max=args.beta, warmup_frac=args.warmup_frac,
                                use_gate=ablation != "no_gate", pressure_mode="random" if ablation == "random_pressure" else "gradient")
    epochs, minimum, patience = (args.wine_epochs, args.wine_min_epochs, args.wine_patience) if name == "wine" else (
        args.max_epochs, args.min_epochs, args.patience)
    lr = args.lr_adam if optimizer_name == "adam" else args.lr_sgd
    bundle = make_optimizers(model, optimizer_name, lr, epochs * len(train_loader), config,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    stopper = PlateauStopper(minimum, patience, args.min_delta, mode="max")
    settings = {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in vars(args).items()}
    settings.update({"dataset": name, "method": method, "ablation": ablation, "optimizer": optimizer_name,
                     "seed": seed, "lr": lr, "batch_size": batch_size, "actual_epochs_limit": epochs,
                     "actual_min_epochs": minimum, "actual_patience": patience, "mf_config": asdict(config),
                     "accuracy_units": "percent", "checkpoint_selection": "best validation accuracy",
                     "gradient_norm_definition": "all trainable parameter gradients before clipping",
                     "pressure_diagnostics_sampling": "last training step in each epoch",
                     "provenance": "new execution; reported paper values are not embedded"})
    hardware = {"device": str(args.device), "torch_version": torch.__version__, "dtype": "float32",
                "name": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "CPU"}
    history = []
    best_state = None
    started = time.perf_counter()
    result = {"status": "running", "config": settings, "data": data_metadata, "hardware": hardware,
              "initial_diagnostics": bundle.diagnostics(), "history": history}
    write_json(directory / "result.json", result)
    set_seed(seed)
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            correct = count = 0
            batch_losses, gradient_norms = [], []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                bundle.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip or math.inf, error_if_nonfinite=True)
                gradient_norms.append(float(gradient))
                bundle.step()
                value = float(loss.detach())
                batch_losses.append(value)
                loss_sum += value * len(targets)
                correct += int((logits.detach().argmax(1) == targets).sum())
                count += len(targets)
            validation = evaluate(model, val_loader, args.device)
            previous_best = stopper.best_epoch
            should_stop = stopper.update(validation["accuracy"], epoch)
            if previous_best != stopper.best_epoch:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            record = {"epoch": epoch, "train_loss": loss_sum / count, "train_accuracy": 100.0 * correct / count,
                      "validation": validation, "batch_loss_std": float(np.std(batch_losses)),
                      "preclip_gradient_norm_mean": float(np.mean(gradient_norms)),
                      "preclip_gradient_norm_peak": max(gradient_norms), "diagnostics": bundle.diagnostics()}
            if args.record_test_history:
                record["test"] = evaluate(model, test_loader, args.device)
            history.append(record)
            write_json(directory / "result.json", result)
            print(f"{name} {optimizer_name}/{method}/{ablation} seed={seed} epoch={epoch} val_acc={validation['accuracy']:.4f}", flush=True)
            if should_stop and not args.disable_early_stopping:
                break
        final_test = evaluate(model, test_loader, args.device)
        final_diagnostics = bundle.diagnostics()
        if args.save_models:
            torch.save(model.state_dict(), directory / "final_model.pt")
        model.load_state_dict(best_state)
        selected_test = evaluate(model, test_loader, args.device)
        if args.save_models:
            torch.save(model.state_dict(), directory / "best_validation_model.pt")
        result.update({"status": "completed", "epochs_run": len(history), "best_validation_epoch": stopper.best_epoch,
                       "best_validation_accuracy": stopper.best_value, "test_at_best_validation": selected_test,
                       "final_validation": history[-1]["validation"], "final_test": final_test,
                       "final_diagnostics": final_diagnostics, "selected_diagnostics": bundle.diagnostics(False),
                       "last_ten_epochs_gradient_peak_mean": float(np.mean([row["preclip_gradient_norm_peak"] for row in history[-10:]])),
                       "stopping_reason": "validation_plateau" if should_stop and not args.disable_early_stopping else "epochs_limit",
                       "elapsed_seconds": time.perf_counter() - started})
        write_json(directory / "result.json", result)
    except BaseException as error:
        result.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       "error": f"{type(error).__name__}: {error}"})
        write_json(directory / "result.json", result)
        raise
    return {key: result[key] for key in ("epochs_run", "best_validation_epoch", "best_validation_accuracy",
                                         "test_at_best_validation", "final_validation", "final_test")}


def parser():
    cli = argparse.ArgumentParser()
    cli.add_argument("--datasets", "--dataset", nargs="+", type=dataset_name, default=["adult"])
    cli.add_argument("--methods", nargs="+", choices=["dense", "fs", "scalar", "diagonal", "mf"], default=["fs", "mf"])
    cli.add_argument("--optimizers", nargs="+", choices=["adam", "sgd"], default=["adam", "sgd"])
    cli.add_argument("--mf-ablations", nargs="+", choices=["full", "no_ema", "no_gate", "random_pressure"], default=["full"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 2024, 2025])
    cli.add_argument("--max-epochs", "--epochs", type=int, default=1000)
    cli.add_argument("--min-epochs", type=int, default=300)
    cli.add_argument("--patience", type=int, default=100)
    cli.add_argument("--min-delta", type=float, default=0.0)
    cli.add_argument("--wine-epochs", type=int, default=100)
    cli.add_argument("--wine-min-epochs", type=int, default=30)
    cli.add_argument("--wine-patience", type=int, default=30)
    cli.add_argument("--wine-color", choices=["red", "white", "both"], default="red")
    cli.add_argument("--wine-binary-threshold", type=float)
    cli.add_argument("--wine-file", type=Path)
    cli.add_argument("--covertype-samples", type=int, default=100000)
    cli.add_argument("--validation-fraction", type=float, default=0.15)
    cli.add_argument("--test-fraction", type=float, default=0.15)
    cli.add_argument("--hidden-size", type=int, default=128)
    cli.add_argument("--hidden-layers", type=int, default=4)
    cli.add_argument("--batch-size", type=int)
    cli.add_argument("--lr-adam", type=float, default=0.001)
    cli.add_argument("--lr-sgd", type=float, default=0.01)
    cli.add_argument("--momentum", type=float, default=0.9)
    cli.add_argument("--weight-decay", type=float, default=0.0001)
    cli.add_argument("--gradient-clip", type=float, default=0.0)
    cli.add_argument("--alpha", type=float, default=0.25)
    cli.add_argument("--beta", type=float, default=4.0)
    cli.add_argument("--rho-geo", type=float, default=0.003)
    cli.add_argument("--beta-p", type=float, default=0.95)
    cli.add_argument("--lambda-s", type=float, default=0.001)
    cli.add_argument("--k-geo", type=int, default=10)
    cli.add_argument("--warmup-frac", type=float, default=0.05)
    cli.add_argument("--device", default="auto")
    cli.add_argument("--data-dir", "--data-root", "--cache-dir", type=Path, default=ROOT / "data" / "raw")
    cli.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mlp_convergence")
    cli.add_argument("--download", action="store_true")
    cli.add_argument("--record-test-history", action="store_true")
    cli.add_argument("--disable-early-stopping", action="store_true")
    cli.add_argument("--save-models", action="store_true")
    return cli


def validate_args(args):
    if not 1 <= args.min_epochs <= args.max_epochs <= 1000 or args.patience < 1:
        raise ValueError("invalid main epoch limits")
    if not 1 <= args.wine_min_epochs <= args.wine_epochs <= 1000 or args.wine_patience < 1:
        raise ValueError("invalid exploratory Wine epoch limits")
    if min(args.hidden_size, args.hidden_layers, args.k_geo) < 1 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError("dimensions, batch size and geometry interval must be positive")
    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        raise ValueError("seeds must be in [0, 2**32)")
    if args.gradient_clip < 0 or args.min_delta < 0:
        raise ValueError("gradient clipping and stopping tolerance must be nonnegative")
    for name in ("lr_adam", "lr_sgd", "momentum", "weight_decay", "gradient_clip", "min_delta",
                 "alpha", "beta", "rho_geo", "beta_p", "lambda_s", "warmup_frac"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if min(args.lr_adam, args.lr_sgd) <= 0 or min(args.weight_decay, args.rho_geo, args.lambda_s) < 0:
        raise ValueError("learning rates must be positive and regularization nonnegative")
    if not 0 <= args.momentum < 1 or not 0 <= args.beta_p < 1 or not 0 <= args.warmup_frac <= 1:
        raise ValueError("invalid momentum, EMA, or warmup fraction")
    if not 0 < args.alpha <= 1 <= args.beta:
        raise ValueError("spectral bounds must contain one")
    if args.wine_binary_threshold is not None and not math.isfinite(args.wine_binary_threshold):
        raise ValueError("wine binary threshold must be finite")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.wine_file is not None:
        args.wine_file = args.wine_file.expanduser().resolve()


def main():
    args = parser().parse_args()
    validate_args(args)
    args.device = resolve_device(args.device)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output_dir / session
    outcomes = {}
    for name in args.datasets:
        for seed in args.seeds:
            data, metadata = load_data(args, name, seed)
            for optimizer_name in args.optimizers:
                for method in args.methods:
                    for ablation in args.mf_ablations if method == "mf" else ["full"]:
                        key = f"{name}/{optimizer_name}/seed_{seed}/{method}/{ablation}"
                        outcomes[key] = train_run(args, name, data, metadata, optimizer_name, method, ablation, seed, output / key)
                        write_json(output / "summary.json", outcomes)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
