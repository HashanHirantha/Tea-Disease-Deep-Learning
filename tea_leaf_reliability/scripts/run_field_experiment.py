"""Run the predefined model/seed combinations for each prepared field fold."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tea_research.data import load_manifest
from tea_research.learning import ARCHITECTURES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--models", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=25)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    folds = sorted(args.manifests.glob("fold_*.csv"))
    if not folds:
        p.error("No fold_*.csv manifests found.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        p.error("Do not repeat seeds or model names.")
    tasks = []
    common_classes = None
    for fold in folds:
        _, audit = load_manifest(fold)
        if audit["protocol"] != "field":
            p.error("The experiment runner accepts real field manifests only.")
        if common_classes is None:
            common_classes = audit["classes"]
        if audit["classes"] != common_classes:
            p.error("Freeze the same class set for all estates.")
        for model in args.models:
            for seed in args.seeds:
                dest = args.output.resolve() / fold.stem / model / f"seed_{seed}"
                tasks.append([sys.executable, "-m", "tea_research", "run",
                              "--manifest", str(fold.resolve()),
                              "--image-root", str(args.image_root.resolve()),
                              "--output", str(dest), "--model", model, "--seed", str(seed),
                              "--batch-size", str(args.batch_size), "--head-epochs", str(args.head_epochs),
                              "--finetune-epochs", str(args.finetune_epochs),
                              "--bootstrap", str(args.bootstrap)])
    print(f"{len(folds)} folds x {len(args.models)} models x {len(args.seeds)} seeds = {len(tasks)} trainings.")
    if args.dry_run:
        for task in tasks:
            print(task)
        return
    for i, task in enumerate(tasks, 1):
        print(f"Starting run {i}/{len(tasks)}: {task[9]}", flush=True)
        subprocess.run(task, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
