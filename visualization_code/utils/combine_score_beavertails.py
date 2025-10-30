import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple


OUTPUT_ROOT_DIR = "/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound_combined_csv"
INPUT_ROOT_DIR = "/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound/api_score_results_sample_500"
OUTPUT_SUBDIR = "beavertails_sample_500"


def _parse_count_token(token: str) -> Optional[int]:
    """
    解析数量 token，例如 "100", "1k" -> 1000。
    仅支持小写 k，若无效返回 None。
    """
    if not token:
        return None
    token = token.strip()
    if token.endswith("k") and token[:-1].isdigit():
        return int(token[:-1]) * 1000
    if token.isdigit():
        return int(token)
    return None


def parse_safe_unsafe_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    从文件名中解析 safe/unsafe 数量。例如:
    pythia-1b_safe_1k_unsafe_100.json -> (1000, 100)
    """
    basename = os.path.basename(filename)
    m = re.search(r"safe_([0-9]+k?)_unsafe_([0-9]+k?)", basename)
    if not m:
        return None
    safe_token, unsafe_token = m.group(1), m.group(2)
    safe_num = _parse_count_token(safe_token)
    unsafe_num = _parse_count_token(unsafe_token)
    if safe_num is None or unsafe_num is None:
        return None
    return safe_num, unsafe_num


def _iter_all_objects(obj: Any) -> Iterable[Any]:
    """
    深度遍历 JSON 对象，yield 所有子对象。
    """
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_all_objects(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_all_objects(v)


def extract_all_scores(obj: Any) -> List[float]:
    """
    从任意 JSON 结构中提取所有数值型分数。
    优先使用 key == 'score' 的值；若遇到其他结构，也仅提取该 key。
    """
    scores: List[float] = []
    for node in _iter_all_objects(obj):
        if isinstance(node, dict) and "score" in node:
            val = node.get("score")
            # 允许字符串型数字
            if isinstance(val, (int, float)):
                scores.append(float(val))
            elif isinstance(val, str):
                try:
                    scores.append(float(val))
                except ValueError:
                    pass
    return scores


def average_score_from_json_file(path: Path) -> Optional[float]:
    """
    读取 JSON 文件并计算所有 score 的平均值。无有效分数返回 None。
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    scores = extract_all_scores(data)
    if not scores:
        return None
    return sum(scores) / len(scores)


def process_model_subdir(model_dir: Path, output_dir: Path) -> None:
    """
    处理单个模型子目录：
    - 遍历所有 .json 文件
    - 解析 safe/unsafe 数量
    - 计算平均分
    - 写出一个 {model}.csv，列: safe_num, unsafe_num, score
    """
    rows: List[Tuple[int, int, float]] = []
    for file in sorted(model_dir.glob("*.json")):
        parsed = parse_safe_unsafe_from_filename(file.name)
        if not parsed:
            continue
        safe_num, unsafe_num = parsed
        avg = average_score_from_json_file(file)
        if avg is None:
            continue
        rows.append((safe_num, unsafe_num, avg))

    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_dir.name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["safe_num", "unsafe_num", "score"])  # header
        for safe_num, unsafe_num, avg in rows:
            writer.writerow([safe_num, unsafe_num, f"{avg:.6f}"])


def main() -> None:
    input_root = Path(INPUT_ROOT_DIR)
    out_root = Path(OUTPUT_ROOT_DIR) / OUTPUT_SUBDIR
    if not input_root.exists() or not input_root.is_dir():
        print(f"输入目录不存在或不可用: {input_root}")
        return

    for child in sorted(input_root.iterdir()):
        if child.is_dir():
            process_model_subdir(child, out_root)


if __name__ == "__main__":
    main()

