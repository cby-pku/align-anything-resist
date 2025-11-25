import argparse
import json
import math
from pathlib import Path
import numpy as np

def calculate_weighted_ppl(metrics_file):
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    total_nll_sum = 0.0
    total_tokens_sum = 0
    
    task_results = []
    
    for task_name, task_data in metrics.items():
        # PalomaCollapseEvaluator output structure:
        # {
        #   'documents_evaluated': ...,
        #   'tokens_evaluated': ...,
        #   'avg_negative_log_likelihood': ..., # avg nll per token
        #   'perplexity': ...,
        #   'per_domain': ...
        # }
        
        tokens = task_data.get('tokens_evaluated', 0)
        avg_nll = task_data.get('avg_negative_log_likelihood')
        
        if tokens > 0 and avg_nll is not None:
            # Recover total NLL for this task
            # total_nll = avg_nll * tokens
            # However, avg_nll might be None if no tokens.
            
            task_total_nll = avg_nll * tokens
            
            total_nll_sum += task_total_nll
            total_tokens_sum += tokens
            
            task_results.append({
                'task': task_name,
                'ppl': task_data.get('perplexity'),
                'tokens': tokens,
                'weight': 0.0 # Will be calculated later
            })
            
    # Calculate overall PPL
    # Overall PPL = exp( (Sum of all NLLs) / (Sum of all tokens) )
    # This is equivalent to token-weighted average of NLLs, then exp.
    
    overall_ppl = float('inf')
    if total_tokens_sum > 0:
        overall_avg_nll = total_nll_sum / total_tokens_sum
        overall_ppl = math.exp(overall_avg_nll)
        
    # Update weights for display
    for res in task_results:
        if total_tokens_sum > 0:
            res['weight'] = res['tokens'] / total_tokens_sum
            
    return overall_ppl, total_tokens_sum, task_results

def main():
    parser = argparse.ArgumentParser(description="Aggregate PALOMA PPL scores")
    parser.add_argument("metrics_file", type=str, help="Path to metrics.json")
    args = parser.parse_args()
    
    overall_ppl, total_tokens, tasks = calculate_weighted_ppl(args.metrics_file)
    
    print(f"\n{'='*60}")
    print(f"PALOMA Aggregate Report")
    print(f"{'='*60}")
    print(f"Source: {args.metrics_file}")
    print(f"Total Tokens Evaluated: {total_tokens:,}")
    print(f"{'-'*60}")
    print(f"{'Task Name':<40} | {'Tokens':<10} | {'Weight':<8} | {'PPL':<10}")
    print(f"{'-'*60}")
    
    # Sort by weight desc
    tasks.sort(key=lambda x: x['tokens'], reverse=True)
    
    for t in tasks:
        print(f"{t['task']:<40} | {t['tokens']:<10,} | {t['weight']:.2%}   | {t['ppl']:.4f}")
        
    print(f"{'-'*60}")
    print(f"{'OVERALL (Token-Weighted)':<40} | {'':<10} | {'':<8} | {overall_ppl:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

