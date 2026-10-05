from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sequence", "spectrum", "mlp", "cnn", "stability", "ablations", "transformer", "scaling")


def build_jobs(args):
    chosen = TASKS if "all" in args.tasks else args.tasks
    seeds = args.seeds if args.seeds is not None else [42, 123, 7, 2024, 2025]
    ordinary = ["--device", args.device, "--seeds", *map(str, seeds)]
    protocol = ["--max-epochs", str(args.max_epochs), "--min-epochs", str(args.min_epochs),
                "--patience", str(args.patience)]
    sequence_data = []
    if args.sequence_data_dir is not None:
        sequence_data += ["--data-dir", str(args.sequence_data_dir)]
    if args.hf_cache is not None:
        sequence_data += ["--hf-cache", str(args.hf_cache)]
    classification_data = ["--data-dir", str(args.data_dir)] if args.data_dir is not None else []
    if args.download:
        classification_data.append("--download")
    output_root = args.output_dir if args.output_dir is not None else ROOT / "outputs" / "current_protocol"
    jobs = []

    def add(task, script, options):
        jobs.append({"task": task, "command": [args.python, str(ROOT / script), *options]})

    if "sequence" in chosen:
        for architecture, dataset, hidden in (
            ("lstm", "wikitext2", 128), ("lstm", "wikitext2", 256),
            ("gru", "wikitext2", 128), ("lstm", "wikitext103", 128),
            ("gru", "wikitext2", 256), ("gru", "wikitext103", 128),
        ):
            add("sequence", "experiments/sequence_models.py", [*ordinary, *protocol, *sequence_data,
                "--architectures", architecture, "--datasets", dataset, "--hidden-sizes", str(hidden),
                "--methods", "fs", "mf", "--optimizers", "adam", "sgd",
                "--output-dir", str(output_root / "sequence")])
    if "spectrum" in chosen:
        add("spectrum", "experiments/sequence_models.py", [*ordinary, *protocol, *sequence_data,
            "--architectures", "lstm", "--datasets", "wikitext2", "--hidden-sizes", "128",
            "--methods", "dense", "fs", "scalar", "diagonal", "mf", "--optimizers", "adam", "sgd",
            "--output-dir", str(output_root / "spectrum")])
    if "mlp" in chosen:
        add("mlp", "experiments/mlp_convergence.py", [*ordinary, *protocol, *classification_data,
            "--datasets", "adult", "covertype", "fashionmnist", "cifar10", "wine",
            "--methods", "fs", "mf", "--optimizers", "adam", "sgd", "--record-test-history",
            "--output-dir", str(output_root / "mlp")])
    if "cnn" in chosen:
        options = [*ordinary, *protocol, *classification_data, "--record-test-history",
                   "--output-dir", str(output_root / "cnn")]
        add("cnn", "experiments/image_classifiers.py", options)
    if "stability" in chosen:
        add("stability", "experiments/mlp_convergence.py", ["--device", args.device, "--seeds", str(seeds[0]),
            *classification_data, "--datasets", "adult", "--methods", "dense", "fs", "mf",
            "--optimizers", "adam", "--max-epochs", "500", "--min-epochs", "500", "--patience", "500",
            "--disable-early-stopping", "--record-test-history", "--output-dir", str(output_root / "stability")])
    if "ablations" in chosen:
        for dataset, epochs in (("adult", 100), ("covertype", 80)):
            add("ablations", "experiments/mlp_convergence.py", [*ordinary, *classification_data,
                "--datasets", dataset, "--methods", "mf", "--optimizers", "adam",
                "--mf-ablations", "full", "no_ema", "no_gate", "random_pressure",
                "--max-epochs", str(epochs), "--min-epochs", str(epochs), "--patience", str(epochs),
                "--disable-early-stopping", "--output-dir", str(output_root / "ablations")])
    if "transformer" in chosen:
        transformer_seeds = args.seeds if args.seeds is not None else [42, 123, 2024]
        add("transformer", "experiments/sequence_models.py", ["--device", args.device,
            "--seeds", *map(str, transformer_seeds), *sequence_data,
            "--architectures", "transformer", "--datasets", "wikitext2", "--hidden-sizes", "128",
            "--methods", "fs", "mf", "--optimizers", "adam", "sgd", "--seq-len", "128",
            "--max-epochs", "12", "--min-epochs", "1", "--patience", "12",
            "--output-dir", str(output_root / "transformer")])
    if "scaling" in chosen:
        add("scaling", "experiments/computational_scaling.py",
            ["--device", "cuda" if args.device == "auto" else args.device, "--seed", str(seeds[0]),
             "--output", str(output_root / "scaling" / f"scaling_{uuid4().hex}.json")])
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Print the current manuscript's experiment commands; execute them only with --execute.")
    parser.add_argument("--tasks", nargs="+", choices=["all", *TASKS], default=["all"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--sequence-data-dir", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_epochs <= args.max_epochs or args.patience < 1:
        parser.error("invalid epoch or patience configuration")
    if ("all" in args.tasks or "scaling" in args.tasks) and args.device != "auto" and not args.device.startswith("cuda"):
        parser.error("computational scaling requires a CUDA device")
    jobs = build_jobs(args)
    print(json.dumps({"paper_source": "main-37.tex", "execution_requested": args.execute, "jobs": jobs}, indent=2), flush=True)
    if args.execute:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", "")])
        for job in jobs:
            subprocess.run(job["command"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
