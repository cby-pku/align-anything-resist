import argparse
import os
import sys
from pathlib import Path
import torch

from align_anything.evaluation.paloma_collapse import PalomaCollapseConfig, PalomaCollapseEvaluator
from align_anything.utils.tools import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description="Offline PALOMA Collapse Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of PALOMA dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of tasks (optional)")
    parser.add_argument("--limit_per_task", type=int, default=None, help="Max documents per task")
    parser.add_argument("--context_window", type=int, default=None, help="Context window size")
    parser.add_argument("--model_max_length", type=int, default=None, help="Model max length")
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type (bf16, fp16, fp32)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    # Configure the evaluator
    cfg = PalomaCollapseConfig(
        enabled=True,
        data_root=args.data_root,
        tasks=args.tasks,
        limit_per_task=args.limit_per_task,
        output_subdir_prefix="offline_paloma",
        context_window=args.context_window,
        model_max_length=args.model_max_length,
        dtype=args.dtype,
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Initialize Evaluator with model: {args.model_path}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Dir: {args.output_dir}")

    evaluator = PalomaCollapseEvaluator(
        cfg=cfg,
        output_root=args.output_dir,
        logger=None # Will print to stdout
    )
    
    # Override device if needed (PalomaCollapseEvaluator defaults to cuda if available)
    if args.device != "cuda" and torch.cuda.is_available():
         print(f"Warning: Forcing device to {args.device}")
         evaluator.device = torch.device(args.device)

    # Run evaluation
    # 'step' is used for folder naming, we use 0 or arbitrary for offline
    evaluator.evaluate(checkpoint_dir=args.model_path, step=0)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()

