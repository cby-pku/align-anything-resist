"""PALOMA-based model collapse evaluation."""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DEFAULT_PALOMA_TASKS: tuple[str, ...] = (
    'm2d2_s2orc_unsplit',
    'm2d2_wikipedia_unsplit',
    'c4_100_domains',
    'c4_en',
    'mc4',
    '4chan_meta_sep',
    'manosphere_meta_sep',
    'gab',
    'twitterAAE_HELM_fixed',
    'wikitext_103',
    'ptb',
    'redpajama',
    'falcon-refinedweb',
    'dolma-v1_5',
    'dolma_100_subreddits',
    'dolma_100_programing_languages',
)


def _ensure_list(value: Any | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    return list(value)


@dataclass
class PalomaCollapseConfig:
    """Configuration for PALOMA collapse evaluation."""

    enabled: bool = False
    data_root: str | None = None
    split: str = 'val'
    tasks: list[str] | None = field(default_factory=list)
    limit_per_task: int | None = None
    num_passes: int | None = None
    steps: list[int] | None = field(default_factory=list)
    output_subdir_prefix: str = 'paloma_collapse'
    model_max_length: int | None = None
    context_window: int | None = None
    dtype: str | None = None
    truncate_tokens: int | None = None
    trust_remote_code: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PalomaCollapseConfig:
        kwargs = {**data}
        kwargs['tasks'] = _ensure_list(kwargs.get('tasks'))
        raw_steps = _ensure_list(kwargs.get('steps'))
        if raw_steps is not None:
            kwargs['steps'] = [int(step) for step in raw_steps]
        limit_val = kwargs.get('limit_per_task')
        if isinstance(limit_val, str) and limit_val.isdigit():
            kwargs['limit_per_task'] = int(limit_val)
        truncate_val = kwargs.get('truncate_tokens')
        if isinstance(truncate_val, str) and truncate_val.isdigit():
            kwargs['truncate_tokens'] = int(truncate_val)
        context_val = kwargs.get('context_window')
        if isinstance(context_val, str) and context_val.isdigit():
            kwargs['context_window'] = int(context_val)
        model_len = kwargs.get('model_max_length')
        if isinstance(model_len, str) and model_len.isdigit():
            kwargs['model_max_length'] = int(model_len)
        return cls(**kwargs)

    @property
    def resolved_tasks(self) -> list[str]:
        tasks = self.tasks or list(DEFAULT_PALOMA_TASKS)
        # Remove duplicates while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for task in tasks:
            if task and task not in seen:
                ordered.append(task)
                seen.add(task)
        return ordered

    def resolve_data_root(self) -> Path:
        data_root = self.data_root or os.environ.get('PALOMA_DATA_ROOT')
        if not data_root:
            raise ValueError(
                'PALOMA data root is not set. '
                'Set collapse_eval_cfgs.data_root or PALOMA_DATA_ROOT.'
            )
        path = Path(data_root).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'PALOMA data root not found: {path}')
        return path

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class PalomaCollapseEvaluator:
    """Run PALOMA perplexity evaluation for model collapse detection."""

    def __init__(
        self,
        cfg: PalomaCollapseConfig,
        output_root: str,
        logger: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.output_root = Path(output_root)
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self, checkpoint_dir: str, step: int) -> Path:
        """Run PALOMA evaluation for the checkpoint at the given step."""
        start_time = time.time()
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint to evaluate not found: {checkpoint_path}')
        data_root = self.cfg.resolve_data_root()
        output_dir = (
            self.output_root
            / (self.cfg.output_subdir_prefix or 'paloma_collapse')
            / f'step_{step:06d}'
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log(
            f'[PALOMA] Evaluating checkpoint at step {step} '
            f'from {checkpoint_path} using data root {data_root}'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.padding_side = 'left'

        torch_dtype = self._resolve_dtype()
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=self.cfg.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        model.to(self.device)
        model.eval()

        chunk_size = self._determine_chunk_size(model, tokenizer)
        truncate_tokens = self.cfg.truncate_tokens

        task_metrics: dict[str, dict[str, Any]] = {}
        tasks = self.cfg.resolved_tasks
        # Use sys.stdout explicitly and force disable=None to respect environment
        task_iter = tqdm(tasks, desc="[PALOMA] Tasks", file=sys.stdout) 
        
        for task_name in task_iter:
            task_iter.set_description(f"[PALOMA] Task: {task_name}")
            metrics = self._evaluate_single_task(
                task_name=task_name,
                data_root=data_root,
                model=model,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                truncate_tokens=truncate_tokens,
            )
            task_metrics[task_name] = metrics

        elapsed = time.time() - start_time
        summary = {
            'step': step,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            'elapsed_seconds': elapsed,
            'metrics': task_metrics,
            'config': self.cfg.to_dict(),
        }
        metrics_path = output_dir / 'metrics.json'
        with metrics_path.open('w', encoding='utf-8') as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

        self._log(
            f'[PALOMA] Finished evaluation for step {step}. '
            f'Results saved to {metrics_path}.'
        )

        del model
        torch.cuda.empty_cache()
        return metrics_path

    @staticmethod
    def cleanup_checkpoint(checkpoint_dir: str) -> None:
        """Remove temporary checkpoints created for evaluation."""
        if not checkpoint_dir:
            return
        path = Path(checkpoint_dir)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def _evaluate_single_task(
        self,
        task_name: str,
        data_root: Path,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        chunk_size: int,
        truncate_tokens: int | None,
    ) -> dict[str, Any]:
        task_dir = data_root / task_name / (self.cfg.split or 'val')
        if not task_dir.exists():
            raise FileNotFoundError(f'PALOMA task directory not found: {task_dir}')

        total_loss = 0.0
        total_tokens = 0
        doc_count = 0
        per_domain: dict[str, dict[str, float]] = {}
        limit = self.cfg.limit_per_task

        # Use sys.stdout and set position to avoid conflict with training bars if any
        pbar = tqdm(total=limit, desc=f"  Eval {task_name}", unit="doc", leave=False, file=sys.stdout)
        
        for text, meta in self._iter_task_documents(task_dir):
            if limit is not None and doc_count >= limit:
                break
            loss_sum, token_count = self._score_text(
                text=text,
                model=model,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                truncate_tokens=truncate_tokens,
            )
            if token_count == 0:
                continue
            doc_count += 1
            total_loss += loss_sum
            total_tokens += token_count
            domain = meta.get('subdomain') or meta.get('domain')
            if domain:
                domain_stat = per_domain.setdefault(domain, {'loss': 0.0, 'tokens': 0})
                domain_stat['loss'] += loss_sum
                domain_stat['tokens'] += token_count

            pbar.update(1)
            if total_tokens > 0:
                current_ppl = math.exp(total_loss / total_tokens)
                pbar.set_postfix(ppl=f"{current_ppl:.2f}")
        
        pbar.close()

        avg_nll = (total_loss / total_tokens) if total_tokens > 0 else None
        perplexity = math.exp(avg_nll) if avg_nll is not None else None
        domain_metrics = {
            domain: {
                'avg_nll': stats['loss'] / stats['tokens'],
                'perplexity': math.exp(stats['loss'] / stats['tokens']),
                'num_tokens': stats['tokens'],
            }
            for domain, stats in per_domain.items()
            if stats['tokens'] > 0
        }

        self._log(
            f'[PALOMA] Task {task_name}: docs={doc_count}, '
            f'tokens={total_tokens}, ppl={perplexity:.4f}' if perplexity else
            f'[PALOMA] Task {task_name}: no valid samples evaluated.'
        )

        return {
            'documents_evaluated': doc_count,
            'tokens_evaluated': total_tokens,
            'avg_negative_log_likelihood': avg_nll,
            'perplexity': perplexity,
            'per_domain': domain_metrics,
        }

    def _score_text(
        self,
        text: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        chunk_size: int,
        truncate_tokens: int | None,
    ) -> tuple[float, int]:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'][0]
        if truncate_tokens is not None and truncate_tokens > 0:
            input_ids = input_ids[:truncate_tokens]
        if input_ids.numel() < 2:
            return 0.0, 0
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        total_loss = 0.0
        total_tokens = 0
        seq_len = input_ids.size(1)
        chunk = max(chunk_size, 2)

        with torch.inference_mode():
            for start_idx in range(0, seq_len - 1, chunk):
                end_idx = min(start_idx + chunk, seq_len)
                chunk_ids = input_ids[:, start_idx:end_idx]
                if chunk_ids.size(1) < 2:
                    break
                chunk_attention = attention_mask[:, start_idx:end_idx]
                outputs = model(
                    input_ids=chunk_ids,
                    attention_mask=chunk_attention,
                    labels=chunk_ids,
                )
                loss_val = float(outputs.loss.detach().cpu())
                token_count = chunk_ids.size(1) - 1
                total_loss += loss_val * token_count
                total_tokens += token_count

        return total_loss, total_tokens

    def _iter_task_documents(self, task_dir: Path) -> Iterator[tuple[str, dict[str, Any]]]:
        exts = ('.jsonl', '.json', '.txt')
        files = sorted(
            path
            for path in task_dir.rglob('*')
            if path.is_file()
            and any(path.name.endswith(ext) or path.name.endswith(f'{ext}.gz') for ext in exts)
        )
        for file_path in files:
            opener = gzip.open if file_path.suffix == '.gz' else open
            mode = 'rt'
            with opener(file_path, mode, encoding='utf-8') as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    text, metadata = self._parse_record(line)
                    if text:
                        yield text, metadata

    @staticmethod
    def _parse_record(line: str) -> tuple[str | None, dict[str, Any]]:
        metadata: dict[str, Any] = {}
        text = None
        if line.startswith('{') and line.endswith('}'):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict):
                metadata = record.get('metadata', {})
                text = record.get('text') or record.get('document') or record.get('body')
        if text is None:
            text = line
        return text, metadata

    def _resolve_dtype(self) -> torch.dtype | None:
        if not self.cfg.dtype:
            return None
        mapping = {
            'bf16': torch.bfloat16,
            'bfloat16': torch.bfloat16,
            'fp16': torch.float16,
            'float16': torch.float16,
            'fp32': torch.float32,
            'float32': torch.float32,
        }
        dtype_key = self.cfg.dtype.lower()
        return mapping.get(dtype_key, None)

    def _determine_chunk_size(self, model, tokenizer) -> int:
        candidates = [
            self.cfg.context_window,
            self.cfg.model_max_length,
            getattr(model.config, 'max_position_embeddings', None),
            getattr(tokenizer, 'model_max_length', None),
        ]
        valid_candidates = [val for val in candidates if isinstance(val, int) and val > 0]
        if not valid_candidates:
            return 2048
        return max(2, min(valid_candidates))

    def _log(self, message: str) -> None:
        if self.logger is not None and hasattr(self.logger, 'print'):
            self.logger.print(message)
        else:
            tqdm.write(message)

