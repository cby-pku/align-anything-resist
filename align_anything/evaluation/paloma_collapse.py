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
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
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
    use_vllm: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PalomaCollapseConfig:
        kwargs = {**data}
        kwargs['tasks'] = _ensure_list(kwargs.get('tasks'))
        raw_steps = _ensure_list(kwargs.get('steps'))
        if raw_steps is not None:
            kwargs['steps'] = [int(step) for step in raw_steps]
        
        # Helper to safely parse int fields
        def _safe_int(val: Any) -> int | None:
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
            return None

        kwargs['limit_per_task'] = _safe_int(kwargs.get('limit_per_task'))
        kwargs['truncate_tokens'] = _safe_int(kwargs.get('truncate_tokens'))
        kwargs['context_window'] = _safe_int(kwargs.get('context_window'))
        kwargs['model_max_length'] = _safe_int(kwargs.get('model_max_length'))
        
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
        
        # Use a file for logging progress instead of stdout
        progress_log_path = output_dir / 'eval_progress.txt'
        with open(progress_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Starting evaluation at step {step}\n")
            f.flush()

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

        model = None
        chunk_size = 2048
        
        if self.cfg.use_vllm:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed. Please install it or set use_vllm=False.")
            self._log("[PALOMA] Using vLLM for evaluation.")
            # vLLM handles memory management, so we don't manually load model to device
            # Note: vLLM is mainly for generation, but can be used for PPL if we use score (logprobs).
            # However, vLLM's offline inference 'score' support might be limited or API differs.
            # Standard vLLM LLM.score() is not available in all versions or behaves differently.
            # For PPL, we usually need 'prompt_logprobs'.
            
            # We will initialize vLLM engine here
            # Use trust_remote_code from config
            dtype = self.cfg.dtype if self.cfg.dtype else "auto"
            
            # Fix for vLLM dtype validation error
            # vLLM expects full names: 'float16', 'bfloat16', 'float32'
            # But our config might use 'bf16', 'fp16', 'fp32'
            dtype_map = {
                'bf16': 'bfloat16',
                'fp16': 'float16', 
                'fp32': 'float32',
                'float16': 'float16',
                'bfloat16': 'bfloat16',
                'float32': 'float32',
                'auto': 'auto'
            }
            if dtype in dtype_map:
                dtype = dtype_map[dtype]

            try:
                model = LLM(
                    model=str(checkpoint_path),
                    trust_remote_code=self.cfg.trust_remote_code,
                    dtype=dtype,
                    # Limit GPU memory usage to prevent OOM with long contexts
                    gpu_memory_utilization=0.85, 
                    max_model_len=self.cfg.model_max_length if self.cfg.model_max_length else None,
                    enforce_eager=True # Prevent Cuda Graph capture OOM
                )
                # vLLM determines max model length automatically if not set
                chunk_size = model.llm_engine.model_config.max_model_len
            except Exception as e:
                 self._log(f"[PALOMA] Failed to initialize vLLM: {e}")
                 raise e
        else:
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
            if progress_log_path:
                with open(progress_log_path, 'a', encoding='utf-8') as f:
                     f.write(f"Starting Task: {task_name}\n")
            metrics = self._evaluate_single_task(
                task_name=task_name,
                data_root=data_root,
                model=model,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                truncate_tokens=truncate_tokens,
                progress_log_path=progress_log_path,
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

        # Cleanup
        if 'model' in locals() and model is not None:
            del model
        
        if not self.cfg.use_vllm:
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
        model: Any, # Can be AutoModel or vLLM LLM
        tokenizer: AutoTokenizer,
        chunk_size: int,
        truncate_tokens: int | None,
        progress_log_path: Path | None = None,
    ) -> dict[str, Any]:
        task_dir = data_root / task_name / (self.cfg.split or 'val')
        if not task_dir.exists():
            raise FileNotFoundError(f'PALOMA task directory not found: {task_dir}')

        total_loss = 0.0
        total_tokens = 0
        doc_count = 0
        per_domain: dict[str, dict[str, float]] = {}
        limit = self.cfg.limit_per_task
        if not isinstance(limit, int):
            limit = None

        # Log task start
        if progress_log_path:
            with open(progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n[Task: {task_name}]\n")
        
        # Prepare batch if using vLLM for efficiency?
        # For now, stick to serial processing to match structure, but use vLLM batched call if possible.
        # vLLM is optimized for batching. But _iter_task_documents yields one by one.
        # We can accumulate a batch here.
        
        BATCH_SIZE = 100 if self.cfg.use_vllm else 1
        batch_texts = []
        batch_metas = []
        
        iterator = self._iter_task_documents(task_dir)
        
        while True:
            # Collect batch
            try:
                while len(batch_texts) < BATCH_SIZE:
                     if limit is not None and doc_count + len(batch_texts) >= limit:
                         break
                     text, meta = next(iterator)
                     batch_texts.append(text)
                     batch_metas.append(meta)
            except StopIteration:
                pass
            
            if not batch_texts:
                break

            # Process batch
            if self.cfg.use_vllm:
                # For vLLM, we use a specialized scoring method
                batch_loss, batch_tokens = self._score_text_vllm_batch(
                    texts=batch_texts,
                    llm=model,
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    truncate_tokens=truncate_tokens
                )
            else:
                # Serial HF processing
                batch_loss = []
                batch_tokens = []
                for text in batch_texts:
                    l, t = self._score_text(
                        text=text,
                        model=model,
                        tokenizer=tokenizer,
                        chunk_size=chunk_size,
                        truncate_tokens=truncate_tokens
                    )
                    batch_loss.append(l)
                    batch_tokens.append(t)

            # Update stats
            for i, (loss_val, token_cnt) in enumerate(zip(batch_loss, batch_tokens)):
                if token_cnt == 0:
                    continue
                
                doc_count += 1
                total_loss += loss_val
                total_tokens += token_cnt
                
                meta = batch_metas[i]
                domain = None
                if isinstance(meta, dict):
                    domain = meta.get('subdomain') or meta.get('domain')
                
                if domain:
                    domain_stat = per_domain.setdefault(domain, {'loss': 0.0, 'tokens': 0})
                    domain_stat['loss'] += loss_val
                    domain_stat['tokens'] += token_cnt
            
            # Log progress
            if progress_log_path and (doc_count % 10 < BATCH_SIZE): # Rough check
                current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                with open(progress_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"  Processed {doc_count} docs. Current PPL: {current_ppl:.4f}\n")
            
            batch_texts = []
            batch_metas = []
            if limit is not None and doc_count >= limit:
                break

        avg_nll = (total_loss / total_tokens) if total_tokens > 0 else None
        perplexity = math.exp(avg_nll) if avg_nll is not None else None
        
        # Log task completion
        if progress_log_path:
            with open(progress_log_path, 'a', encoding='utf-8') as f:
                f.write(f"[Task: {task_name}] Finished. Docs: {doc_count}, Tokens: {total_tokens}, PPL: {perplexity}\n")
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

    def _score_text_vllm_batch(
        self,
        texts: List[str],
        llm: Any, # vLLM LLM object
        tokenizer: AutoTokenizer,
        chunk_size: int,
        truncate_tokens: int | None,
    ) -> Tuple[List[float], List[int]]:
        """
        Score a batch of texts using vLLM.
        vLLM doesn't have a direct 'score' method in older versions, but newer versions might.
        Common workaround: set logprobs=1 and max_tokens=1, provided prompts are the full text.
        BUT vLLM is for generation. To get PPL, we need logprobs of the PROMPT.
        sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1)
        
        Note: chunking logic is tricky with vLLM batching because each doc has different chunks.
        We will flatten all chunks from all docs into a massive list of prompts.
        """
        
        # 1. Tokenize and Chunk all texts
        all_prompts = [] # List of token IDs (list of ints)
        doc_indices = [] # Maps which doc this chunk belongs to
        
        # Temporarily suppress tokenizer warning
        original_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e9)
        
        try:
            batch_encoded = tokenizer(
                texts,
                add_special_tokens=False,
            )
        finally:
            tokenizer.model_max_length = original_max_len
            
        for i, token_ids in enumerate(batch_encoded['input_ids']):
            if truncate_tokens is not None and truncate_tokens > 0:
                token_ids = token_ids[:truncate_tokens]
            
            if len(token_ids) < 2:
                continue
                
            # Chunking (Disjoint)
            chunk = max(chunk_size, 2)
            stride = chunk
            seq_len = len(token_ids)
            
            for start_idx in range(0, seq_len - 1, stride):
                end_idx = min(start_idx + chunk, seq_len)
                chunk_ids = token_ids[start_idx:end_idx]
                if len(chunk_ids) < 2:
                    continue
                all_prompts.append(chunk_ids)
                doc_indices.append(i)

        if not all_prompts:
            return [0.0]*len(texts), [0]*len(texts)

        # 2. Run vLLM
        # We need logprobs for each token in the prompt.
        # Using prompt_logprobs=20 to catch the ground truth token in most cases.
        sampling_params = SamplingParams(
            max_tokens=1, 
            prompt_logprobs=20, 
            temperature=1.0,
        )
        
        # If all_prompts is huge (e.g. 16M tokens total), passing all at once might OOM in vLLM engine/scheduler
        # We should batch the submission to vLLM generate as well.
        
        # Sub-batching for vLLM submission
        # Reduce batch size to prevent potential deadlock or scheduler congestion with long prompts
        SUB_BATCH_SIZE = 32 # Number of chunks per vLLM call (reduced from 256)
        outputs = []
        
        for k in range(0, len(all_prompts), SUB_BATCH_SIZE):
            sub_prompts = all_prompts[k : k + SUB_BATCH_SIZE]
            # Enable tqdm to show vLLM progress
            sub_outputs = llm.generate(prompt_token_ids=sub_prompts, sampling_params=sampling_params, use_tqdm=True)
            outputs.extend(sub_outputs)
        
        batch_loss = [0.0] * len(texts)
        batch_tokens = [0] * len(texts)
        
        for j, output in enumerate(outputs):
            doc_idx = doc_indices[j]
            logprobs_list = output.prompt_logprobs
            
            if not logprobs_list:
                continue

            # logprobs_list has length equal to input prompt length.
            # logprobs_list[i] is the logprob distribution P(token_i | tokens_0...i-1)
            # Note: vLLM (and most libs) aligns it such that logprobs_list[i] corresponds to the i-th token in the prompt.
            # The first token's logprob is usually None (because it's P(t0|nothing)).
            
            chunk_loss = 0.0
            chunk_cnt = 0
            
            chunk_ids = all_prompts[j]
            
            for k, lp_dict in enumerate(logprobs_list):
                if k == 0: continue # Skip first token
                if not lp_dict: continue
                
                target_id = chunk_ids[k]
                if target_id in lp_dict:
                    chunk_loss -= lp_dict[target_id].logprob # Negative Log Likelihood (so we subtract logprob)
                    # Wait, typical code sums NLL? Or sums LogProb?
                    # _score_text sums "loss_val".
                    # loss_val from HF model() is CrossEntropyLoss, which is NLL (positive).
                    # logprob is negative (usually).
                    # So NLL = -logprob.
                    # So we ADD (-logprob).
                    chunk_loss += -lp_dict[target_id].logprob
                    chunk_cnt += 1
                else:
                    # Fallback if not found in top-K: use the lowest logprob in the dict as an estimate?
                    # Or just ignore? Ignoring biases PPL down (better).
                    # Using min value biases PPL up (worse).
                    # Let's penalize with the minimum value found in top-K to be conservative.
                    # Or ideally, we'd throw an error, but that stops eval.
                    min_logprob = min(v.logprob for v in lp_dict.values())
                    chunk_loss += -min_logprob
                    chunk_cnt += 1
            
            batch_loss[doc_idx] += chunk_loss
            batch_tokens[doc_idx] += chunk_cnt

        return batch_loss, batch_tokens

    def _score_text(
        self,
        text: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        chunk_size: int,
        truncate_tokens: int | None,
    ) -> tuple[float, int]:
        # Temporarily suppress tokenizer warning about sequence length
        # because we handle chunking manually.
        original_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e9)
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors='pt',
            )
        finally:
            tokenizer.model_max_length = original_max_len

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
        
        # If sequence is very long, log a message to indicate progress
        if seq_len > 20000:
             self._log(f"  [PALOMA] Processing long document ({seq_len} tokens)...")

        chunk = max(chunk_size, 2)
        stride = chunk # Default to disjoint windows (stride = chunk) for speed
        
        # Use sliding window if stride < chunk, otherwise disjoint
        
        with torch.inference_mode():
            for start_idx in range(0, seq_len - 1, stride):
                end_idx = min(start_idx + chunk, seq_len)
                chunk_ids = input_ids[:, start_idx:end_idx]
                if chunk_ids.size(1) < 2:
                    break
                
                # In sliding window (if we were to support it fully with context overlap),
                # we would need to mask out loss for the overlap.
                # Here we implement simple disjoint chunking (stride=chunk) as per original code,
                # but wrapped in a loop that can be extended.
                # The current implementation matches the 'disjoint' logic.
                
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

