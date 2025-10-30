import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple


OUTPUT_ROOT_DIR = "/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound_combined_csv"
INPUT_ROOT_DIR = "/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound/api_score_results_sample_500"
OUTPUT_SUBDIR = "beavertails_sample_500"
VERBOSE = True


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
    except Exception as e:
        # 可能是实际为 JSONL，但扩展名为 .json 的情况，尝试按 JSONL 解析
        if VERBOSE:
            print(f"读取 JSON 失败，尝试按 JSONL 解析: {path}，错误: {e}")
        return average_score_from_jsonl_file(path)

    scores = extract_all_scores(data)
    if not scores:
        return None
    return sum(scores) / len(scores)


def average_score_from_jsonl_file(path: Path) -> Optional[float]:
    """
    读取 JSONL 文件并计算所有 score 的平均值。无有效分数返回 None。
    """
    scores: List[float] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                scores.extend(extract_all_scores(obj))
    except Exception as e:
        if VERBOSE:
            print(f"读取 JSONL 失败，跳过: {path}，错误: {e}")
        return None

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
    files = list(sorted(model_dir.glob("*.json"))) + list(sorted(model_dir.glob("*.jsonl")))
    if VERBOSE:
        print(f"扫描目录: {model_dir}，发现 {len(files)} 个文件")
    for file in files:
        parsed = parse_safe_unsafe_from_filename(file.name)
        if not parsed:
            if VERBOSE:
                print(f"  跳过(命名未匹配 safe/unsafe): {file.name}")
            continue
        safe_num, unsafe_num = parsed
        avg: Optional[float]
        if file.suffix == ".jsonl":
            avg = average_score_from_jsonl_file(file)
        else:
            avg = average_score_from_json_file(file)
        if avg is None:
            if VERBOSE:
                print(f"  跳过(无有效 score): {file.name}")
            continue
        rows.append((safe_num, unsafe_num, avg))

    if not rows:
        if VERBOSE:
            print(f"目录 {model_dir.name} 无有效数据，跳过生成 CSV")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{model_dir.name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["safe_num", "unsafe_num", "score"])  # header
        for safe_num, unsafe_num, avg in rows:
            writer.writerow([safe_num, unsafe_num, f"{avg:.6f}"])
    if VERBOSE:
        print(f"已写出 {len(rows)} 行: {csv_path}")


def main() -> None:
    input_root = Path(INPUT_ROOT_DIR)
    out_root = Path(OUTPUT_ROOT_DIR) / OUTPUT_SUBDIR
    if not input_root.exists() or not input_root.is_dir():
        print(f"输入目录不存在或不可用: {input_root}")
        return

    if VERBOSE:
        print(f"输入根目录: {input_root}")
        print(f"输出根目录: {out_root}")
    for child in sorted(input_root.iterdir()):
        if child.is_dir():
            if VERBOSE:
                print(f"处理子目录: {child.name}")
            process_model_subdir(child, out_root)


if __name__ == "__main__":
    main()

