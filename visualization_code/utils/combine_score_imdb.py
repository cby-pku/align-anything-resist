import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OUTPUT_ROOT_DIR = "/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound_combined_csv"
INPUT_ROOT_DIR = "/mnt/shared-storage-user/zhoujiayi/boyuan/rm_score_rebound/imdb"
OUTPUT_SUBDIR = "imdb"
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


def parse_pos_neg_from_name(name: str) -> Optional[Tuple[int, int]]:
    """
    从名称中解析 pos/neg 数量。例如:
    OLMo-2-0425-1B_pos_5000_neg_500(.json) -> (5000, 500)
    支持小写 k（如 1k）。
    """
    basename = os.path.basename(name)
    m = re.search(r"_pos_([0-9]+k?)_neg_([0-9]+k?)", basename)
    if not m:
        return None
    pos_token, neg_token = m.group(1), m.group(2)
    pos_num = _parse_count_token(pos_token)
    neg_num = _parse_count_token(neg_token)
    if pos_num is None or neg_num is None:
        return None
    return pos_num, neg_num


def parse_model_name_from_dirname(dirname: str) -> Optional[str]:
    """
    从目录名中解析模型名，规则：取到 "_pos_" 之前的全部内容。
    例如: "OLMo-2-0425-1B_pos_5000_neg_500" -> "OLMo-2-0425-1B"
    """
    if "_pos_" not in dirname:
        return None
    return dirname.split("_pos_", 1)[0]


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


def gather_rows_by_model(input_root: Path) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    扫描输入根目录下的每个子目录，每个子目录包含一个 JSON/JSONL 文件，
    子目录名形如: <model>_pos_<POS>_neg_<NEG>
    - 解析 model、pos/neg
    - 计算该文件平均分
    返回: { model_name: [(pos_num, neg_num, avg_score), ...] }
    """
    result: Dict[str, List[Tuple[int, int, float]]] = {}
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        model_name = parse_model_name_from_dirname(child.name)
        pos_neg = parse_pos_neg_from_name(child.name)
        if model_name is None or pos_neg is None:
            if VERBOSE:
                print(f"跳过(目录名未匹配 pos/neg): {child.name}")
            continue
        pos_num, neg_num = pos_neg

        # 寻找该目录下可用的数据文件（优先 .json，其次 .jsonl）
        files = list(sorted(child.glob("*.json"))) + list(sorted(child.glob("*.jsonl")))
        if not files:
            if VERBOSE:
                print(f"  跳过(无 .json/.jsonl 文件): {child}")
            continue

        # 典型结构为仅一个文件；若有多个，逐个计算并分别计入
        for file in files:
            if file.suffix == ".jsonl":
                avg = average_score_from_jsonl_file(file)
            else:
                avg = average_score_from_json_file(file)
            if avg is None:
                if VERBOSE:
                    print(f"  跳过(无有效 score): {file.name}")
                continue
            result.setdefault(model_name, []).append((pos_num, neg_num, avg))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 IMDB pos/neg 评分为每模型 CSV")
    parser.add_argument("--input-root", type=str, default=INPUT_ROOT_DIR, help="输入根目录，如 /path/to/imdb")
    parser.add_argument("--output-root", type=str, default=OUTPUT_ROOT_DIR, help="输出根目录根，用于放置汇总 CSV")
    parser.add_argument("--output-subdir", type=str, default=OUTPUT_SUBDIR, help="输出子目录名，如 imdb")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    global VERBOSE
    if args.verbose:
        VERBOSE = True

    input_root = Path(args.input_root)
    out_root = Path(args.output_root) / args.output_subdir

    if not input_root.exists() or not input_root.is_dir():
        print(f"输入目录不存在或不可用: {input_root}")
        return

    if VERBOSE:
        print(f"输入根目录: {input_root}")
    model_to_rows = gather_rows_by_model(input_root)

    out_root.mkdir(parents=True, exist_ok=True)
    for model_name, rows in model_to_rows.items():
        if not rows:
            continue
        csv_path = out_root / f"{model_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pos_num", "neg_num", "score"])  # header
            for pos_num, neg_num, avg in rows:
                writer.writerow([pos_num, neg_num, f"{avg:.6f}"])
        if VERBOSE:
            print(f"已写出 {len(rows)} 行: {csv_path}")

    if VERBOSE:
        print(f"输出根目录: {out_root}")


if __name__ == "__main__":
    main()

