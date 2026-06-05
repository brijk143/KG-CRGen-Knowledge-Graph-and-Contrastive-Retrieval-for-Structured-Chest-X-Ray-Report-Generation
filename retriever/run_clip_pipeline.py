"""
run_clip_pipeline.py
====================
Master script that runs the full CLIP retriever pipeline:
  Step 1: Prepare data splits (train/val/test JSON)
  Step 2: Train OpenCLIP model with contrastive learning
  Step 3: Build text embedding database
  Step 4: Evaluate on both val and test sets
  Step 5: Generate all visualizations

Usage:
  python run_clip_pipeline.py                # Run everything
  python run_clip_pipeline.py --step 2       # Run from step 2 onward
  python run_clip_pipeline.py --only 5       # Run only step 5
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/Users/bkishor/Desktop/kg_new")
SCRIPTS_DIR = BASE_DIR / "Retriever"
def run_step(script_name: str, description: str, extra_args: list = None):
    """Run a python script and stream output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\n {script_name} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"\n {description} — DONE")


def main():
    parser = argparse.ArgumentParser(description="Run CLIP retriever pipeline")
    parser.add_argument("--step", type=int, default=1,
                        help="Start from this step (1-5)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    steps = {
        1: ("01_prepare_data.py", "STEP 1: Prepare Data Splits"),
        2: ("02_train.py", "STEP 2: Train OpenCLIP Model"),
        3: ("03_build_database.py", "STEP 3: Build Report Database"),
        4: ("04_evaluate.py", "STEP 4: Evaluate Val & Test"),
        5: ("05_visualize.py", "STEP 5: Generate Visualizations"),
    }

    # Build extra args for training
    train_args = []
    if args.epochs:
        train_args.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        train_args.extend(["--batch_size", str(args.batch_size)])
    if args.lr:
        train_args.extend(["--lr", str(args.lr)])
    if args.patience:
        train_args.extend(["--patience", str(args.patience)])

    if args.only:
        script, desc = steps[args.only]
        extra = train_args if args.only == 2 else None
        run_step(script, desc, extra)
    else:
        for step_num in sorted(steps.keys()):
            if step_num < args.step:
                continue
            script, desc = steps[step_num]
            extra = train_args if step_num == 2 else None
            run_step(script, desc, extra)

    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Splits:       {SCRIPTS_DIR / 'splits'}")
    print(f"  Checkpoint:   {SCRIPTS_DIR / 'checkpoints' / 'best_indiana_clip.pt'}")
    print(f"  Database:     {SCRIPTS_DIR / 'database'}")
    print(f"  Evaluation:   {SCRIPTS_DIR / 'evaluation'}")
    print(f"  Plots:        {SCRIPTS_DIR / 'plots'}")


if __name__ == "__main__":
    main()
