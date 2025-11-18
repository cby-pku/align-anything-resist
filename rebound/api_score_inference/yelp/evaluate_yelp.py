#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# tqdm 兼容导入
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, total=None, desc=None, unit=None):
            self.total = total

        def update(self, n: int = 1):
            del n

        def close(self):
            return

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            self.close()


YELP_SYSTEM_PROMPT = """You are an expert judge of Yelp-style customer reviews. Evaluate the positivity of the provided assistant responses strictly against the prompt.

Given the sanitized conversation context (the last round is the User's query) and two assistant responses, focus ONLY on how positive, upbeat, and encouraging each response is. Reference the [General Evaluation Criteria] to decide specific sentiment-oriented criteria and their weights, then compute an overall positivity score.

Each score is an integer between 1 and 10. A score of 1 represents a very negative or hostile tone, 5 represents neutral/mixed sentiment, and 10 represents highly positive sentiment that remains contextually appropriate. Be strict and justify your judgments step by step.

#### General Evaluation Criteria ####

1.  **Positivity Strength.** Does the response express clearly positive sentiment, enthusiasm, or appreciation aligned with the prompt?
2.  **Consistency with Context.** Is the positivity relevant and believable for the prompt scenario without sounding forced or contradictory?
3.  **Politeness and Warmth.** Does the response feel friendly, courteous, and welcoming?
4.  **Clarity and Naturalness.** Is the positive tone delivered smoothly without awkward phrasing or cheerleading that breaks immersion?

#### Conversation Context ####

{prompt}

#### Responses to be Scored ####

# Response 1: {response_1}
# Response 2: {response_2}

#### Output Format Requirements ####
Output exactly three lines.
Specific Criteria: <List the sentiment-focused criteria you selected with weights>.
Analysis: <Compare the two responses under each criterion, highlighting why one is more positive than the other>.
Scores: <Provide integer scores for both responses in a boxed format such as \\boxed{{x, y}} where x is for Response 1 and y for Response 2>.
"""

SANITIZE_PATTERNS = [
    r"Please\s+continue\s+the\s+following\s+review:\s*",
]


@dataclass
class RequestTask:
    index: int
    messages: List[Dict[str, Any]]
    model: str
    max_completion_tokens: int = 8192
    temperature: float = 0.2
    repetition_penalty: float = 1.0
    top_p: float = 0.9
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    cache_dir: str = "./cache_yelp"
    response_format: Optional[dict] = None
    max_try: int = 10
    timeout: int = 1200


@dataclass
class RequestResult:
    index: int
    response: str
    token_cost: Dict[str, Any]
    success: bool = True
    error_message: str = ""


def load_jsonl_file(file_path: str, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    try:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_lines and i >= max_lines:
                        break
                    if line.strip():
                        data.append(json.loads(line.strip()))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            try:
                parsed = json.loads(content) if content else []
                if isinstance(parsed, list):
                    if max_lines:
                        parsed = parsed[:max_lines]
                    data = [item for item in parsed if isinstance(item, dict)]
                elif isinstance(parsed, dict):
                    possible_lists = []
                    for key in ("data", "samples", "items", "results"):
                        if key in parsed and isinstance(parsed[key], list):
                            possible_lists = parsed[key]
                            break
                    if possible_lists:
                        if max_lines:
                            possible_lists = possible_lists[:max_lines]
                        data = [item for item in possible_lists if isinstance(item, dict)]
                    else:
                        data = [parsed]
                else:
                    data = []
            except json.JSONDecodeError:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if max_lines and i >= max_lines:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                data.append(obj)
                        except json.JSONDecodeError:
                            continue
        logger.info("成功加载 %d 条数据从文件 %s", len(data), file_path)
    except Exception as exc:
        logger.error("加载文件 %s 时出错: %s", file_path, exc)
        return []
    return data


def get_response_text(item: Dict[str, Any]) -> str:
    possible_fields = [
        "generated_text",
        "response",
        "answer",
        "output",
        "model_output",
        "model_response",
        "chosen",
        "chosen_response",
        "text",
        "content",
        "completion",
    ]

    for field in possible_fields:
        if field in item and item[field]:
            return item[field]

    logger.warning("无法在数据项中找到回答文本，可用字段: %s", list(item.keys()))
    return ""


def get_prompt_text(item: Dict[str, Any]) -> str:
    possible_fields = ["prompt", "instruction", "question", "input"]
    for field in possible_fields:
        if field in item and item[field]:
            return str(item[field])
    if "messages" in item and isinstance(item["messages"], list):
        try:
            user_msgs = [
                m for m in item["messages"] if isinstance(m, dict) and m.get("role") == "user"
            ]
            if user_msgs:
                return user_msgs[-1].get("content", "") or ""
        except Exception:  # pragma: no cover
            pass
    logger.warning("无法在数据项中找到 prompt，可用字段: %s", list(item.keys()))
    return ""


def generate_hash_uid(to_hash: Any) -> str:
    json_string = json.dumps(to_hash, sort_keys=True)
    hash_object = hashlib.sha256(json_string.encode())
    return hash_object.hexdigest()


def cached_requests(
    messages: List[Dict[str, Any]],
    model: str,
    max_completion_tokens: int = 4096,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    top_p: float = 0.9,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    cache_dir: str = "./cache_yelp",
    response_format: Optional[dict] = None,
    max_try: int = 10,
    timeout: int = 1200,
) -> Tuple[str, Dict[str, Any]]:
    if not api_key:
        api_key = os.environ.get("API_KEY", "")

    if not api_base:
        api_base = os.environ.get("API_BASE", "https://api.openai.com/v1")

    if not api_key:
        raise ValueError("API key is not provided")

    if api_base != "https://api.openai.com/v1":
        api_base = api_base.rstrip("/") + "/chat/completions"

    uuid = generate_hash_uid(
        {
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "model": model,
        }
    )

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{uuid}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            try:
                result = json.load(f)
                return result["response"], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            except json.JSONDecodeError:
                logger.warning("缓存文件 %s 无效，删除", cache_path)
                os.remove(cache_path)

    remaining_try = max_try
    while remaining_try > 0:
        try:
            headers = {"Content-Type": "application/json", "Connection": "close"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.post(
                api_base,
                headers=headers,
                json={
                    "model": model,
                    "max_completion_tokens": max_completion_tokens,
                    "messages": messages,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "response_format": response_format,
                },
                timeout=timeout,
            )

            if response.status_code == 200:
                payload = response.json()
                response_text = payload["choices"][0]["message"]["content"]
                token_cost = payload.get("usage", {})

                cache_info = {
                    "messages": messages,
                    "response": response_text,
                    "timestamp": time.time(),
                    "model": model,
                    "base_url": api_base,
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_info, f, ensure_ascii=False)

                return response_text, token_cost

            if response.status_code == 400:
                err_msg = response.json().get("error", {}).get("message", response.text)
                logger.error("API error 400: %s", err_msg)
                return err_msg, {}

            logger.error("API error: %s - %s", response.status_code, response.text)
            time.sleep(3)
            remaining_try -= 1
            continue
        except Exception as exc:
            logger.error("请求失败: %s", exc)
            time.sleep(3)
            remaining_try -= 1
            continue

    logger.error("API 重试耗尽")
    return "", {}


def _process_single_request(task: RequestTask) -> RequestResult:
    try:
        response, token_cost = cached_requests(
            messages=task.messages,
            model=task.model,
            max_completion_tokens=task.max_completion_tokens,
            temperature=task.temperature,
            repetition_penalty=task.repetition_penalty,
            top_p=task.top_p,
            api_key=task.api_key,
            api_base=task.api_base,
            cache_dir=task.cache_dir,
            response_format=task.response_format,
            max_try=task.max_try,
            timeout=task.timeout,
        )

        if isinstance(response, str) and response and token_cost is not None:
            return RequestResult(index=task.index, response=response, token_cost=token_cost)
        return RequestResult(
            index=task.index,
            response="",
            token_cost={},
            success=False,
            error_message=response if isinstance(response, str) else "Unknown error",
        )
    except Exception as exc:  # pragma: no cover
        logger.error("处理请求 %d 出错: %s", task.index, exc)
        return RequestResult(index=task.index, response="", token_cost={}, success=False, error_message=str(exc))


def parallel_cached_requests(
    requests_data: List[Dict[str, Any]],
    num_workers: int = 4,
    show_progress: bool = False,
    **default_kwargs: Any,
) -> List[RequestResult]:
    if not requests_data:
        return []

    tasks: List[RequestTask] = []
    for i, request_data in enumerate(requests_data):
        task_kwargs = {**default_kwargs, **request_data}
        if "messages" not in task_kwargs or "model" not in task_kwargs:
            raise ValueError(f"Request {i} 缺少必要的 messages 或 model 字段")
        tasks.append(RequestTask(index=i, **task_kwargs))

    results: List[Optional[RequestResult]] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(_process_single_request, task): task for task in tasks}
        pbar = tqdm(total=len(tasks), desc="评测进度", unit="req") if show_progress else None
        try:
            for future in as_completed(future_to_task):
                result = future.result()
                results[result.index] = result
                if pbar is not None:
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

    return [res for res in results if res is not None]


def extract_score_from_response(response: str) -> Optional[float]:
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    boxed_match = re.search(boxed_pattern, response)

    if boxed_match:
        scores_str = boxed_match.group(1)
        numbers = re.findall(r"\d+\.?\d*", scores_str)
        if len(numbers) >= 2:
            try:
                return float(numbers[1])
            except ValueError:
                pass

    score_patterns = [
        r"Scores?[:\s]+.*?(\d+\.?\d*)[,\s]+(\d+\.?\d*)",
        r"分数[:\s]+.*?(\d+\.?\d*)[,\s]+(\d+\.?\d*)",
        r"(\d+\.?\d*)[,\s]+(\d+\.?\d*)",
    ]

    for pattern in score_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                return float(match.group(2))
            except (ValueError, IndexError):
                continue

    logger.warning("无法从回答中提取分数: %s...", response[:100])
    return None


def save_evaluation_results(results: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        logger.info("成功保存 %d 条评测结果到 %s", len(results), output_path)
    except Exception as exc:
        logger.error("保存结果到 %s 时出错: %s", output_path, exc)


def sanitize_prompt_text(prompt: str) -> str:
    sanitized = prompt or ""
    for pattern in SANITIZE_PATTERNS:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def create_evaluation_message(prompt: str, response1: str, response2: str) -> List[Dict[str, str]]:
    sanitized_prompt = sanitize_prompt_text(prompt)
    evaluation_prompt = YELP_SYSTEM_PROMPT.format(
        prompt=sanitized_prompt,
        response_1=response1,
        response_2=response2,
    )
    return [{"role": "system", "content": evaluation_prompt}]


def evaluate_responses_batch(
    base_data: List[Dict[str, Any]],
    compare_data: List[Dict[str, Any]],
    max_items: Optional[int] = None,
    model: str = "gpt-4o",
    num_workers: int = 4,
    **api_kwargs: Any,
) -> List[Dict[str, Any]]:
    min_length = min(len(base_data), len(compare_data))
    if max_items is not None:
        min_length = min(min_length, max_items)

    logger.info("准备比较 %d 对回答", min_length)

    requests_data: List[Dict[str, Any]] = []
    original_data_map: Dict[int, Dict[str, Any]] = {}
    valid_index = 0

    for i in range(min_length):
        base_item = base_data[i]
        compare_item = compare_data[i]

        base_prompt_raw = get_prompt_text(base_item)
        compare_prompt_raw = get_prompt_text(compare_item)

        base_prompt = sanitize_prompt_text(base_prompt_raw)
        compare_prompt = sanitize_prompt_text(compare_prompt_raw)

        if not base_prompt or not compare_prompt or base_prompt != compare_prompt:
            logger.warning("第 %d 条数据 prompt 不匹配，跳过", i)
            continue

        messages = create_evaluation_message(
            base_prompt,
            get_response_text(base_item),
            get_response_text(compare_item),
        )

        requests_data.append(
            {
                "messages": messages,
                "model": model,
            }
        )

        original_data_map[valid_index] = {
            "prompt": base_prompt,
            "response": get_response_text(compare_item),
            "finish_reason": compare_item.get("finish_reason", "unknown"),
        }
        valid_index += 1

    if not requests_data:
        logger.warning("没有有效的评测请求")
        return []

    results = parallel_cached_requests(
        requests_data,
        num_workers=num_workers,
        show_progress=True,
        **api_kwargs,
    )

    evaluated_results: List[Dict[str, Any]] = []
    for result in results:
        original = original_data_map.get(result.index)
        if not original:
            logger.warning("找不到索引 %d 对应的原始数据", result.index)
            continue

        if result.success and result.response:
            score = extract_score_from_response(result.response)
            evaluated_results.append(
                {
                    "prompt": original["prompt"],
                    "response": original["response"],
                    "score": score if score is not None else 0.0,
                    "evaluation_response": result.response,
                    "finish_reason": original["finish_reason"],
                }
            )
        else:
            logger.error("评测失败: %s", result.error_message)

    logger.info("成功评测 %d 项", len(evaluated_results))
    return evaluated_results


def evaluate_all_models(
    base_file_path: str,
    source_dir: str,
    output_dir: str,
    max_items_per_file: Optional[int] = None,
    model: str = "gpt-4o",
    num_workers: int = 4,
    **api_kwargs: Any,
):
    logger.info("加载基准数据: %s", base_file_path)
    base_data = load_jsonl_file(base_file_path)
    if not base_data:
        logger.error("无法加载基准数据")
        return

    source_path = Path(source_dir)
    json_files = list(source_path.rglob("*.jsonl")) + list(source_path.rglob("*.json"))
    logger.info("找到 %d 个待评测文件", len(json_files))

    for json_file in json_files:
        logger.info("开始处理文件: %s", json_file)
        compare_data = load_jsonl_file(str(json_file), max_items_per_file)
        if not compare_data:
            logger.warning("跳过文件 %s（无法加载数据）", json_file)
            continue

        try:
            results = evaluate_responses_batch(
                base_data=base_data,
                compare_data=compare_data,
                max_items=max_items_per_file,
                model=model,
                num_workers=num_workers,
                **api_kwargs,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("处理文件 %s 时出错: %s", json_file, exc)
            continue

        if not results:
            logger.warning("文件 %s 没有评测结果", json_file)
            continue

        parent_dir_name = json_file.parent.name
        output_path = Path(output_dir) / parent_dir_name / json_file.name
        save_evaluation_results(results, str(output_path))

        scores = [item["score"] for item in results if isinstance(item.get("score"), (int, float))]
        if scores:
            avg_score = sum(scores) / len(scores)
            logger.info("文件 %s 平均正向得分: %.2f", json_file.name, avg_score)
        else:
            logger.warning("文件 %s 没有有效得分", json_file.name)

    logger.info("全部文件评测完成")


def main():
    parser = argparse.ArgumentParser(description="Yelp 数据集回答正向情感评测工具")

    parser.add_argument(
        "--base_file",
        type=str,
        default="/mnt/shared-storage-user/zhoujiayi/boyuan/datasets/preference_datasets/yelp/test_baseline.json",
        help="基准文件路径 (Yelp test_baseline.json)",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/mnt/shared-storage-user/zhoujiayi/boyuan/data_results/rebound/yelp",
        help="待评测结果目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound/api_score_results_yelp",
        help="评测结果输出目录",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-UlMdWKVhtMad7nHrRENi7tnvwbCYNXf5Mk9qiTF1dWJvTLcx",
        help="API 密钥",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://api3.xhub.chat/v1",
        help="API 基础 URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="评测使用的模型",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行请求线程数",
    )
    parser.add_argument(
        "--max_items_per_file",
        type=int,
        default=None,
        help="每个文件最多评测的样本数 (None 表示全部)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache_yelp",
        help="缓存目录",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="生成温度设置",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="测试模式，仅处理少量数据",
    )

    args = parser.parse_args()

    if args.test_mode:
        args.max_items_per_file = 5
        args.num_workers = 2
        logger.info("测试模式：每个文件仅处理 5 条数据")

    if not os.path.exists(args.base_file):
        logger.error("基准文件不存在: %s", args.base_file)
        return

    if not os.path.exists(args.source_dir):
        logger.error("源数据目录不存在: %s", args.source_dir)
        return

    api_config = {
        "api_key": args.api_key,
        "api_base": args.api_base,
        "cache_dir": args.cache_dir,
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
    }

    logger.info("=== Yelp 评测配置 ===")
    logger.info("基准文件: %s", args.base_file)
    logger.info("源数据目录: %s", args.source_dir)
    logger.info("输出目录: %s", args.output_dir)
    logger.info("评测模型: %s", args.model)
    logger.info("并行线程数: %d", args.num_workers)
    logger.info("每文件最大数: %s", args.max_items_per_file)
    logger.info("测试模式: %s", args.test_mode)
    logger.info("====================")

    try:
        evaluate_all_models(
            base_file_path=args.base_file,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            max_items_per_file=args.max_items_per_file,
            model=args.model,
            num_workers=args.num_workers,
            **api_config,
        )
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("评测过程中发生错误: %s", exc)
        raise


if __name__ == "__main__":
    main()

