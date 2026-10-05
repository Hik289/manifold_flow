from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.common import (
    PlateauStopper,
    SpectrumLinear,
    make_optimizers,
    resolve_device,
    set_seed,
    write_json,
)
from manifoldflow.manifoldflow_optimizer import ManifoldFlowConfig


DATASET_CONFIGS = {
    "wikitext2": "wikitext-2-raw-v1",
    "wikitext103": "wikitext-103-raw-v1",
}


def dataset_name(value: str) -> str:
    normalized = value.lower().replace("-", "").replace("_", "")
    aliases = {"wt2": "wikitext2", "wt103": "wikitext103"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in DATASET_CONFIGS:
        raise argparse.ArgumentTypeError("choose wikitext2 or wikitext103")
    return normalized


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--architectures", "--architecture", nargs="+", type=str.lower,
                        choices=("lstm", "gru", "transformer"), default=["lstm"])
    result.add_argument("--datasets", "--dataset", nargs="+", type=dataset_name,
                        default=["wikitext2"])
    result.add_argument("--hidden-sizes", "--hidden-size", nargs="+", type=int,
                        choices=(128, 256), default=[128])
    result.add_argument("--optimizers", "--optimizer", nargs="+", type=str.lower,
                        choices=("adam", "sgd"), default=["adam", "sgd"])
    result.add_argument("--methods", nargs="+", type=str.lower,
                        choices=("dense", "fs", "scalar", "diagonal", "mf"),
                        default=["fs", "mf"])
    result.add_argument("--spectrum-suite", action="store_true")
    result.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 2024, 2025])
    result.add_argument("--max-epochs", "--epochs", type=int, default=1000)
    result.add_argument("--min-epochs", type=int, default=300)
    result.add_argument("--patience", type=int, default=100)
    result.add_argument("--min-delta", type=float, default=0.0)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--eval-batch-size", type=int, default=1)
    result.add_argument("--seq-len", type=int, default=35)
    result.add_argument("--embedding-size", type=int, default=128)
    result.add_argument("--num-layers", type=int, default=1)
    result.add_argument("--dropout", type=float)
    result.add_argument("--transformer-layers", type=int, default=2)
    result.add_argument("--transformer-heads", type=int, default=4)
    result.add_argument("--ffn-dim", type=int, default=256)
    result.add_argument("--transformer-dropout", type=float)
    result.add_argument("--hidden-state", choices=("detach", "reset"), default="detach")
    result.add_argument("--line-eos", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--max-vocab-size", type=int, default=10000)
    result.add_argument("--lr", type=float)
    result.add_argument("--lr-adam", "--adam-lr", type=float, default=0.003)
    result.add_argument("--lr-sgd", "--sgd-lr", type=float, default=0.01)
    result.add_argument("--momentum", type=float, default=0.9)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--gradient-clip", type=float, default=0.5)
    result.add_argument("--alpha", type=float, default=0.25)
    result.add_argument("--beta", type=float, default=4.0)
    result.add_argument("--rho-geo", type=float, default=0.01)
    result.add_argument("--beta-p", type=float, default=0.95)
    result.add_argument("--lambda-s", type=float, default=0.001)
    result.add_argument("--k-geo", type=int, default=10)
    result.add_argument("--warmup-frac", type=float, default=0.05)
    result.add_argument("--tau-c", type=float, default=0.1)
    result.add_argument("--tau-r", type=float, default=0.0)
    result.add_argument("--alpha-c", type=float, default=5.0)
    result.add_argument("--alpha-r", type=float, default=2.0)
    result.add_argument("--data-dir", type=Path)
    result.add_argument("--hf-cache", "--cache-dir", type=Path)
    result.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/sequence_models")
    result.add_argument("--device", default="auto")
    result.add_argument("--num-threads", type=int)
    result.add_argument("--allow-nondeterministic", action="store_true")
    result.add_argument("--save-models", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    for name in ("max_epochs", "min_epochs", "patience", "batch_size",
                 "eval_batch_size", "seq_len", "embedding_size", "num_layers", "k_geo",
                 "transformer_layers", "transformer_heads", "ffn_dim"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_epochs < args.min_epochs:
        raise ValueError("max_epochs must be at least min_epochs")
    if args.min_delta < 0 or not math.isfinite(args.min_delta):
        raise ValueError("min_delta must be finite and nonnegative")
    if not 2 <= args.max_vocab_size <= 10000:
        raise ValueError("max_vocab_size must be between 2 and 10000")
    for name in ("dropout", "transformer_dropout"):
        value = getattr(args, name)
        if value is not None and not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if "transformer" in args.architectures and any(
            size % args.transformer_heads != 0 for size in args.hidden_sizes):
        raise ValueError("Transformer hidden sizes must be divisible by transformer_heads")
    if not 0 < args.alpha <= 1 <= args.beta or not math.isfinite(args.beta):
        raise ValueError("spectral bounds must satisfy 0 < alpha <= 1 <= beta")
    for name in ("lr_adam", "lr_sgd", "gradient_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("lr must be finite and positive")
    if not 0 <= args.momentum < 1 or not 0 <= args.beta_p < 1:
        raise ValueError("momentum and beta_p must be in [0, 1)")
    for name in ("weight_decay", "rho_geo", "lambda_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= args.warmup_frac <= 1:
        raise ValueError("warmup_frac must be in [0, 1]")
    for name in ("tau_c", "tau_r", "alpha_c", "alpha_r"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"{name} must be finite")
    if args.num_threads is not None and args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        raise ValueError("seeds must be in [0, 2**32)")
    for name in ("architectures", "datasets", "hidden_sizes", "optimizers", "methods", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    if args.spectrum_suite:
        args.methods = ["dense", "fs", "scalar", "diagonal", "mf"]
    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()
        if len(args.datasets) > 1 and (args.data_dir / "train.txt").exists():
            raise ValueError("multiple datasets require data_dir/wikitext2 and data_dir/wikitext103")
    if args.hf_cache is not None:
        args.hf_cache = args.hf_cache.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_records(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        yield from handle


def tokens(records: Iterable[str], line_eos: bool = True) -> Iterable[str]:
    for record in records:
        if not isinstance(record, str):
            raise ValueError("all corpus records must contain text strings")
        for line in record.splitlines() or [""]:
            yield from line.split()
            if line_eos:
                yield "<eos>"


def load_corpus(name: str, args: argparse.Namespace) -> Dict[str, Any]:
    if args.data_dir is not None:
        directory = args.data_dir
        if not (directory / "train.txt").is_file():
            directory = directory / name
        paths = {"train": directory / "train.txt", "validation": directory / "valid.txt",
                 "test": directory / "test.txt"}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing local corpus files: " + ", ".join(missing))
        records = {split: (lambda path=path: local_records(path))
                   for split, path in paths.items()}
        source = {
            "kind": "local_utf8_text", "dataset_label": name,
            "files": {split: {"path": str(path), "bytes": path.stat().st_size,
                              "sha256": file_hash(path)} for split, path in paths.items()},
        }
    else:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import datasets
            from datasets import DownloadConfig, load_dataset
        except ImportError as error:
            raise RuntimeError("install datasets or provide --data-dir with local text files") from error
        datasets.config.HF_DATASETS_OFFLINE = True
        from huggingface_hub import constants as hub_constants
        hub_constants.HF_HUB_OFFLINE = True
        try:
            dataset = load_dataset(
                "wikitext", DATASET_CONFIGS[name],
                cache_dir=str(args.hf_cache) if args.hf_cache is not None else None,
                download_config=DownloadConfig(local_files_only=True),
                download_mode="reuse_dataset_if_exists",
            )
        except Exception as error:
            raise RuntimeError(
                f"{name} is unavailable in the offline datasets cache; provide --data-dir "
                "or point --hf-cache at an existing cache"
            ) from error
        for split in ("train", "validation", "test"):
            if split not in dataset or "text" not in dataset[split].column_names:
                raise ValueError(f"cached {name} requires a text column in {split}")
        records = {split: (lambda split=split: (row["text"] for row in dataset[split]))
                   for split in ("train", "validation", "test")}
        source = {
            "kind": "huggingface_cached_dataset", "offline_only": True,
            "dataset": "wikitext", "dataset_config": DATASET_CONFIGS[name],
            "datasets_version": datasets.__version__,
            "cache_dir": str(args.hf_cache) if args.hf_cache is not None else None,
            "splits": {split: {"rows": len(dataset[split]),
                               "fingerprint": getattr(dataset[split], "_fingerprint", None),
                               "cache_files": dataset[split].cache_files}
                       for split in records},
        }
    counts = Counter(tokens(records["train"](), args.line_eos))
    ordinary = sorted((token for token in counts if token not in {"<unk>", "<eos>"}),
                      key=lambda token: (-counts[token], token))
    vocabulary = ["<unk>", "<eos>"] + ordinary[:args.max_vocab_size - 2]
    indices = {token: index for index, token in enumerate(vocabulary)}
    encoded = {}
    split_metadata = {}
    for split, factory in records.items():
        buffer = array.array("q")
        unknown = 0
        for token in tokens(factory(), args.line_eos):
            index = indices.get(token, 0)
            buffer.append(index)
            unknown += int(index == 0)
        if not buffer:
            raise ValueError(f"{split} has no tokens")
        encoded[split] = torch.frombuffer(buffer, dtype=torch.int64)
        split_metadata[split] = {
            "tokens": len(buffer), "unknown_id_tokens": unknown,
            "unknown_fraction": unknown / len(buffer),
        }
    vocabulary_hash = hashlib.sha256("\n".join(vocabulary).encode("utf-8")).hexdigest()
    metadata = {
        "source": source,
        "tokenization": {
            "scheme": "case_sensitive_whitespace", "encoding": "utf-8",
            "line_boundary": "one <eos> per source line including blank lines" if args.line_eos
                             else "whitespace only; no explicit line-boundary token",
            "vocabulary_split": "train", "frequency_ties": "lexicographic",
            "unknown_token": "<unk>", "unknown_id": 0, "eos_id": 1,
        },
        "vocabulary": {"size": len(vocabulary), "maximum_size": args.max_vocab_size,
                       "sha256": vocabulary_hash, "tokens": vocabulary},
        "splits": split_metadata,
    }
    return {"encoded": encoded, "metadata": metadata}


def batchify(stream: torch.Tensor, batch_size: int) -> torch.Tensor:
    columns = stream.numel() // batch_size
    if columns < 2:
        raise ValueError(f"at least {2 * batch_size} tokens are required for batch_size={batch_size}")
    return stream[:columns * batch_size].view(batch_size, columns).transpose(0, 1)


def get_batch(source: torch.Tensor, offset: int, seq_len: int,
              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    length = min(seq_len, source.size(0) - 1 - offset)
    inputs = source[offset:offset + length].to(device)
    targets = source[offset + 1:offset + length + 1].reshape(-1).to(device)
    return inputs, targets


def detach_hidden(hidden: Any) -> Any:
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(component.detach() for component in hidden)
    return hidden.detach()


class RecurrentLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, architecture: str, hidden_size: int,
                 method: str, args: argparse.Namespace):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, args.embedding_size)
        recurrent = nn.LSTM if architecture == "lstm" else nn.GRU
        dropout = args.dropout if args.dropout is not None else 0.0
        self.recurrent = recurrent(args.embedding_size, hidden_size,
                                   num_layers=args.num_layers,
                                   dropout=dropout if args.num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.projection = SpectrumLinear(hidden_size, vocab_size, method,
                                         alpha=args.alpha, beta=args.beta)

    def forward(self, inputs: torch.Tensor, hidden: Any = None) -> Tuple[torch.Tensor, Any]:
        embedding = self.dropout(self.embedding(inputs))
        output, hidden = self.recurrent(embedding, hidden)
        logits = self.projection(self.dropout(output).reshape(-1, output.size(-1)))
        return logits, hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, method: str, dropout: float,
                 args: argparse.Namespace):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, args.transformer_heads,
                                               dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn1 = SpectrumLinear(hidden_size, args.ffn_dim, method,
                                   alpha=args.alpha, beta=args.beta)
        self.ffn2 = SpectrumLinear(args.ffn_dim, hidden_size, method,
                                   alpha=args.alpha, beta=args.beta)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attention(inputs, inputs, inputs, attn_mask=mask,
                                      need_weights=False)
        hidden = self.norm1(inputs + self.dropout(attention))
        feed_forward = self.ffn2(self.dropout(F.gelu(self.ffn1(hidden))))
        return self.norm2(hidden + self.dropout(feed_forward))


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, method: str,
                 args: argparse.Namespace):
        super().__init__()
        dropout = args.transformer_dropout
        if dropout is None:
            dropout = args.dropout if args.dropout is not None else 0.1
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position = nn.Embedding(max(512, args.seq_len), hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, method, dropout, args)
            for _ in range(args.transformer_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, hidden: Any = None) -> Tuple[torch.Tensor, Any]:
        positions = torch.arange(inputs.size(0), device=inputs.device).unsqueeze(1)
        output = self.dropout(self.embedding(inputs) + self.position(positions))
        mask = torch.triu(torch.ones(inputs.size(0), inputs.size(0), dtype=torch.bool,
                                     device=inputs.device), diagonal=1)
        for block in self.blocks:
            output = block(output, mask)
        logits = self.projection(self.norm(output)).reshape(-1, self.projection.out_features)
        return logits, None


def perplexity(loss: float) -> Optional[float]:
    if not math.isfinite(loss) or loss > math.log(sys.float_info.max):
        return None
    return math.exp(loss)


def evaluate(model: nn.Module, source: torch.Tensor, seq_len: int,
             device: torch.device, hidden_state: str = "detach") -> Dict[str, Any]:
    model.eval()
    hidden = None
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for offset in range(0, source.size(0) - 1, seq_len):
            inputs, targets = get_batch(source, offset, seq_len, device)
            logits, hidden = model(inputs, detach_hidden(hidden) if hidden_state == "detach" else None)
            loss = F.cross_entropy(logits, targets, reduction="sum").item()
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite evaluation cross entropy")
            total_loss += loss
            total_tokens += targets.numel()
    average = total_loss / total_tokens
    return {"loss": average, "ppl": perplexity(average), "tokens": total_tokens}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def hardware(device: torch.device) -> Dict[str, Any]:
    information = {
        "device": str(device), "device_type": device.type,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "torch_num_threads": torch.get_num_threads(), "dtype": "float32",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        information.update({"accelerator_name": properties.name,
                            "accelerator_total_memory_bytes": properties.total_memory,
                            "compute_capability": [properties.major, properties.minor]})
    elif device.type == "mps":
        information["accelerator_name"] = "Apple Metal Performance Shaders"
    return information


def tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def initialization_metadata(model: nn.Module) -> Dict[str, Any]:
    layers = [(name, layer) for name, layer in model.named_modules()
              if isinstance(layer, SpectrumLinear)]
    excluded = {id(parameter) for _, layer in layers for parameter in layer.parameters()}
    ordinary = {}
    for name, parameter in model.named_parameters():
        if id(parameter) not in excluded:
            ordinary[name] = tensor_hash(parameter)
    return {"ordinary_parameter_sha256": ordinary,
            "spectrum_layers": {
                name: {"realized_weight_sha256": tensor_hash(layer.effective_weight()),
                       "bias_sha256": tensor_hash(layer.bias) if layer.bias is not None else None}
                for name, layer in layers}}


def configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}


def manifold_config(args: argparse.Namespace) -> ManifoldFlowConfig:
    return ManifoldFlowConfig(
        rho_geo=args.rho_geo, beta_P=args.beta_p, lambda_S=args.lambda_s,
        K_geo=args.k_geo, tau_c=args.tau_c, tau_r=args.tau_r,
        alpha_c=args.alpha_c, alpha_r=args.alpha_r,
        lambda_min=args.alpha, lambda_max=args.beta, warmup_frac=args.warmup_frac,
    )


def snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_run(args: argparse.Namespace, corpus: Dict[str, Any], architecture: str,
              dataset: str, hidden_size: int, optimizer: str, method: str,
              seed: int, device: torch.device, output_path: Path) -> Dict[str, Any]:
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vocab_size = corpus["metadata"]["vocabulary"]["size"]
    model = (TransformerLanguageModel(vocab_size, hidden_size, method, args)
             if architecture == "transformer" else
             RecurrentLanguageModel(vocab_size, architecture, hidden_size, method, args)).to(device)
    initial = initialization_metadata(model)
    train_data = batchify(corpus["encoded"]["train"], args.batch_size)
    validation_data = batchify(corpus["encoded"]["validation"], args.eval_batch_size)
    test_data = batchify(corpus["encoded"]["test"], args.eval_batch_size)
    steps_per_epoch = math.ceil((train_data.size(0) - 1) / args.seq_len)
    learning_rate = args.lr if args.lr is not None else (
        args.lr_adam if optimizer == "adam" else args.lr_sgd)
    mf_config = manifold_config(args)
    bundle = make_optimizers(model, optimizer, learning_rate,
                             steps_per_epoch * args.max_epochs, mf_config,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    stopper = PlateauStopper(args.min_epochs, args.patience, args.min_delta, mode="min")
    run_config = configuration(args)
    run_config.update({"architecture": architecture, "dataset": dataset,
                       "hidden_size": hidden_size, "optimizer": optimizer,
                       "method": method, "seed": seed, "learning_rate": learning_rate,
                       "manifold_flow": asdict(mf_config),
                       "steps_per_epoch": steps_per_epoch,
                       "maximum_optimizer_steps": steps_per_epoch * args.max_epochs,
                       "learning_rate_schedule": "constant",
                       "constrained_layers": [name for name, layer in model.named_modules()
                                               if isinstance(layer, SpectrumLinear)],
                       "shuffle": False,
                       "effective_dropout": model.dropout.p,
                       "effective_embedding_size": hidden_size if architecture == "transformer"
                                                   else args.embedding_size,
                       "hidden_state": args.hidden_state if architecture != "transformer"
                                       else "none; causal attention within each fixed segment",
                       "checkpoint_selection": "minimum validation cross entropy",
                       "early_stopping_metric": "validation_cross_entropy",
                       "test_evaluation": "once after restoring the best validation checkpoint"})
    result = {"status": "running", "protocol": "new_sequence_runner",
              "config": run_config, "data": corpus["metadata"],
              "code_source": {"sequence_models": {"path": str(Path(__file__).resolve()),
                                                   "sha256": file_hash(Path(__file__))},
                              "common": {"path": str(ROOT / "experiments/common.py"),
                                         "sha256": file_hash(ROOT / "experiments/common.py")},
                              "manifoldflow_optimizer": {
                                  "path": str(ROOT / "src/manifoldflow/manifoldflow_optimizer.py"),
                                  "sha256": file_hash(ROOT / "src/manifoldflow/manifoldflow_optimizer.py")}},
              "actual_hardware": hardware(device), "initialization": initial,
              "history": [], "best_epoch": None, "best_val_ppl": None,
              "final_val_ppl": None, "test_ppl": None,
              "batchification": {
                  split: {"batch_size": data.size(1), "retained_tokens": data.numel(),
                          "discarded_remainder_tokens": corpus["encoded"][split].numel() - data.numel(),
                          "prediction_tokens_per_pass": (data.size(0) - 1) * data.size(1)}
                  for split, data in (("train", train_data), ("validation", validation_data),
                                      ("test", test_data))}}
    write_json(output_path, result)
    set_seed(seed)
    best_state = None
    best_loss = math.inf
    started = time.perf_counter()
    try:
        for epoch in range(1, args.max_epochs + 1):
            synchronize(device)
            epoch_started = time.perf_counter()
            model.train()
            hidden = None
            total_loss = 0.0
            total_tokens = 0
            last_gradient_norm = None
            for offset in range(0, train_data.size(0) - 1, args.seq_len):
                inputs, targets = get_batch(train_data, offset, args.seq_len, device)
                hidden = detach_hidden(hidden) if args.hidden_state == "detach" else None
                bundle.zero_grad()
                logits, hidden = model(inputs, hidden)
                loss = F.cross_entropy(logits, targets)
                if not math.isfinite(loss.item()):
                    raise FloatingPointError(f"nonfinite training cross entropy in epoch {epoch}")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                last_gradient_norm = float(gradient_norm.item())
                bundle.step()
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
            validation = evaluate(model, validation_data, args.seq_len, device, args.hidden_state)
            stop = stopper.update(validation["loss"], epoch)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_state = snapshot(model)
                result.update({"best_epoch": epoch, "best_val_loss": best_loss,
                               "best_val_ppl": validation["ppl"]})
            synchronize(device)
            training_loss = total_loss / total_tokens
            entry = {"epoch": epoch, "train_loss": training_loss,
                     "train_ppl": perplexity(training_loss), "train_prediction_tokens": total_tokens,
                     "val_loss": validation["loss"], "val_ppl": validation["ppl"],
                     "validation_prediction_tokens": validation["tokens"],
                     "optimizer_steps": epoch * steps_per_epoch,
                     "last_unclipped_gradient_norm": last_gradient_norm,
                     "epoch_seconds": time.perf_counter() - epoch_started,
                     "diagnostics": bundle.diagnostics()}
            result["history"].append(entry)
            result.update({"completed_epochs": epoch, "final_epoch": epoch,
                           "final_val_loss": validation["loss"],
                           "final_val_ppl": validation["ppl"],
                           "elapsed_seconds": time.perf_counter() - started})
            write_json(output_path, result)
            print(f"{architecture} {dataset} h={hidden_size} {optimizer} {method} seed={seed} "
                  f"epoch={epoch} validation_ppl={validation['ppl']}", flush=True)
            if stop:
                result["stop_reason"] = "validation_plateau"
                break
        if best_state is None:
            raise RuntimeError("no finite validation checkpoint was obtained")
        if args.save_models:
            checkpoint_path = output_path.with_suffix(".final.pt")
            torch.save({"state_dict": snapshot(model), "epoch": result["final_epoch"],
                        "validation_loss": result["final_val_loss"], "config": run_config},
                       checkpoint_path)
            result["final_checkpoint_path"] = str(checkpoint_path)
        model.load_state_dict(best_state)
        selected_diagnostics = bundle.diagnostics(include_gradient_metrics=False)
        test = evaluate(model, test_data, args.seq_len, device, args.hidden_state)
        synchronize(device)
        result.update({"status": "completed", "test_epoch": result["best_epoch"],
                       "test_loss": test["loss"], "test_ppl": test["ppl"],
                       "test_prediction_tokens": test["tokens"], "test_evaluations": 1,
                       "best_checkpoint_diagnostics": selected_diagnostics,
                       "elapsed_seconds": time.perf_counter() - started})
        result.setdefault("stop_reason", "maximum_epochs")
        if args.save_models:
            checkpoint_path = output_path.with_suffix(".best.pt")
            torch.save({"state_dict": best_state, "epoch": result["best_epoch"],
                        "validation_loss": result["best_val_loss"], "config": run_config},
                       checkpoint_path)
            result["best_checkpoint_path"] = str(checkpoint_path)
        if device.type == "cuda":
            result["actual_hardware"]["peak_allocated_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            result["actual_hardware"]["peak_reserved_memory_bytes"] = torch.cuda.max_memory_reserved(device)
        write_json(output_path, result)
    except BaseException as error:
        result.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       "error": f"{type(error).__name__}: {error}",
                       "elapsed_seconds": time.perf_counter() - started})
        write_json(output_path, result)
        raise
    return {"path": str(output_path), "architecture": architecture, "dataset": dataset,
            "hidden_size": hidden_size, "optimizer": optimizer, "method": method,
            "seed": seed, "status": result["status"], "best_epoch": result["best_epoch"],
            "final_epoch": result["final_epoch"], "best_val_ppl": result["best_val_ppl"],
            "final_val_ppl": result["final_val_ppl"], "test_ppl": result["test_ppl"],
            "stop_reason": result["stop_reason"]}


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(not args.allow_nondeterministic)
    torch.backends.cudnn.deterministic = not args.allow_nondeterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    device = torch.device(resolve_device(args.device))
    corpus_data = {name: load_corpus(name, args) for name in args.datasets}
    session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:12]
    args.session_dir = args.output_dir / session_name
    manifest_path = args.session_dir / "summary.json"
    manifest = {"status": "running", "protocol": "new_sequence_runner",
                "config": configuration(args), "actual_hardware": hardware(device),
                "datasets": {name: corpus["metadata"] for name, corpus in corpus_data.items()},
                "runs": [], "planned_runs": len(args.architectures) * len(args.datasets)
                * len(args.hidden_sizes) * len(args.optimizers) * len(args.methods) * len(args.seeds)}
    write_json(manifest_path, manifest)
    try:
        for architecture, name, hidden_size, optimizer, seed, method in itertools.product(
                args.architectures, args.datasets, args.hidden_sizes, args.optimizers,
                args.seeds, args.methods):
            output = args.session_dir / architecture / name / f"h{hidden_size}" / optimizer / f"{method}_seed{seed}.json"
            outcome = train_run(args, corpus_data[name], architecture, name, hidden_size,
                                optimizer, method, seed, device, output)
            manifest["runs"].append(outcome)
            write_json(manifest_path, manifest)
        manifest["status"] = "completed"
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         "error": f"{type(error).__name__}: {error}"})
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
