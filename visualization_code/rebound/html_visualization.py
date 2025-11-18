import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_INPUT_DIR = \
    "/Users/boyuan/local_projects/wx_workspace/nature-resist/codespace/rm_score_rebound_combined_csv/imdb"
DEFAULT_OUTPUT_HTML = "index.html"


def load_csv_series(input_dir: str) -> Tuple[Dict[str, Dict[str, Dict[str, List[float]]]], Dict[str, Dict[str, str]]]:
    """
    读取目录下所有 CSV（safe_num,unsafe_num,score），按模型(文件名去后缀)聚合。
    返回数据结构：
    {
      "ModelName": {
          "safe_num(字符串)": {"x": [unsafe_num... 按升序], "y": [score...]},
          ...
      },
      ...
    }
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    model_to_series: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    model_to_labels: Dict[str, Dict[str, str]] = {}

    for file in sorted(input_path.iterdir()):
        if not file.is_file() or file.suffix.lower() != ".csv":
            continue
        model_name = file.stem
        safe_map: Dict[str, List[Tuple[int, float]]] = {}

        with file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 兼容列名大小写/下划线写法，统一映射
            field_map = {k.strip().lower(): k for k in reader.fieldnames or []}
            # 既支持 safe_num/unsafe_num，也支持 pos_num/neg_num
            safe_key: Optional[str] = field_map.get("safe_num") or field_map.get("pos_num")
            unsafe_key: Optional[str] = field_map.get("unsafe_num") or field_map.get("neg_num")
            score_key: Optional[str] = field_map.get("score")
            if not (safe_key and unsafe_key and score_key):
                # 跳过不符合格式的文件
                continue

            # 记录该模型使用的字段名，优先保留首次识别结果
            if model_name not in model_to_labels:
                # 转回标准化的小写，以便前端显示
                safe_label = "safe_num" if safe_key.lower() == "safe_num" else "pos_num"
                unsafe_label = "unsafe_num" if unsafe_key.lower() == "unsafe_num" else "neg_num"
                model_to_labels[model_name] = {"safe_label": safe_label, "unsafe_label": unsafe_label}

            for row in reader:
                try:
                    safe_val = int(str(row[safe_key]).strip())
                    unsafe_val = int(str(row[unsafe_key]).strip())
                    score_val = float(str(row[score_key]).strip())
                except Exception:
                    continue
                key = str(safe_val)
                safe_map.setdefault(key, []).append((unsafe_val, score_val))

        # 按 unsafe_num 升序整理为 x/y
        series: Dict[str, Dict[str, List[float]]] = {}
        for safe_key_str, pairs in safe_map.items():
            pairs.sort(key=lambda t: t[0])
            series[safe_key_str] = {
                "x": [p[0] for p in pairs],
                "y": [p[1] for p in pairs],
            }

        if series:
            model_to_series[model_name] = series

    if not model_to_series:
        raise RuntimeError(f"未在目录中找到有效 CSV: {input_dir}")

    # 对缺失标签的模型填默认标签（理论上不会触发）
    for m in model_to_series.keys():
        if m not in model_to_labels:
            model_to_labels[m] = {"safe_label": "safe_num", "unsafe_label": "unsafe_num"}

    return model_to_series, model_to_labels


def generate_html(data: Dict[str, Dict[str, Dict[str, List[float]]]], labels: Dict[str, Dict[str, str]]) -> str:
    """
    生成自包含 HTML（使用 Plotly CDN）。
    - 模型下拉框选择不同 CSV（模型）
    - safe_num 勾选控制曲线显隐
    - 线型/配色参考 seaborn/tab10 风格，线宽 4，圆点 marker
    """
    # 参考 seaborn/tab10 的一组配色（与 combine_visulization.py 风格一致）
    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    ]

    data_json = json.dumps(data, ensure_ascii=False)
    palette_json = json.dumps(palette)
    labels_json = json.dumps(labels, ensure_ascii=False)

    template = """
<!doctype html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>RM Score 折线图</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, 'Noto Sans', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
      margin: 0; padding: 24px; background: #fff;
    }
    .controls {
      display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-bottom: 12px;
    }
    #safeControls label { margin-right: 12px; user-select: none; }
    #chart { width: 100%; height: 720px; }
    .hint { color: #666; font-size: 13px; margin-top: 4px; }
  </style>
  </head>
  <body>
    <div class="controls">
      <label for="modelSelect">模型：</label>
      <select id="modelSelect"></select>
      <div id="safeControls"></div>
    </div>
    <div class="hint">可通过勾选控制各 safe_num 曲线显隐；点击图例也可切换显示。</div>
    <div id="chart"></div>
    <script>
      const DATA = __DATA__;
      const PALETTE = __PALETTE__;
      const LABELS = __LABELS__;

      function buildSafeNums(model) {
        return Object.keys(DATA[model]).map(s => String(s)).sort((a,b) => Number(a) - Number(b));
      }

      function makeTraces(model) {
        const safeNums = buildSafeNums(model);
        return safeNums.map((s, i) => ({
          x: DATA[model][s].x,
          y: DATA[model][s].y,
          mode: 'lines+markers',
          name: (LABELS[model]?.safe_label || 'safe_num') + '=' + s,
          line: { width: 4, color: PALETTE[i % PALETTE.length] },
          marker: { size: 8, symbol: 'circle' },
          hovertemplate: 'unsafe_num=%{x}<br>score=%{y:.3f}<extra></extra>'
        }));
      }

      function draw(model) {
        const traces = makeTraces(model);
        const xVals = Array.from(new Set([].concat(...Object.values(DATA[model]).map(v => v.x)))).sort((a,b)=>a-b);
        const layout = {
          template: 'plotly_white',
          legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.12 },
          xaxis: { title: 'Number of Negative Data', tickmode: 'array', tickvals: xVals },
          yaxis: { title: 'Score' },
          margin: { t: 80, l: 70, r: 20, b: 80 }
        };
        Plotly.newPlot('chart', traces, layout, { displaylogo: false, responsive: true });
      }

      function rebuildSafeControls(model) {
        const container = document.getElementById('safeControls');
        container.innerHTML = '';
        const safeNums = buildSafeNums(model);
        // 创建“全选/全不选”
        const btnAll = document.createElement('button');
        btnAll.textContent = '全选';
        btnAll.onclick = () => {
          safeNums.forEach((s, i) => {
            const cb = document.getElementById('safe_' + s);
            if (cb && !cb.checked) cb.checked = true;
            Plotly.restyle('chart', { visible: true }, [i]);
          });
        };
        const btnNone = document.createElement('button');
        btnNone.textContent = '全不选';
        btnNone.style.marginRight = '8px';
        btnNone.onclick = () => {
          safeNums.forEach((s, i) => {
            const cb = document.getElementById('safe_' + s);
            if (cb && cb.checked) cb.checked = false;
            Plotly.restyle('chart', { visible: 'legendonly' }, [i]);
          });
        };
        container.appendChild(btnAll);
        container.appendChild(btnNone);

        safeNums.forEach((s, idx) => {
          const id = 'safe_' + s;
          const label = document.createElement('label');
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.id = id; cb.checked = true;
          cb.addEventListener('change', () => {
            Plotly.restyle('chart', { visible: cb.checked ? true : 'legendonly' }, [idx]);
          });
          label.appendChild(cb);
          const span = document.createElement('span');
          span.textContent = 'safe_num=' + s;
          label.appendChild(span);
          container.appendChild(label);
        });
      }

      function init() {
        const models = Object.keys(DATA).sort();
        const select = document.getElementById('modelSelect');
        models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m; opt.textContent = m; select.appendChild(opt);
        });
        select.addEventListener('change', () => {
          const model = select.value;
          rebuildSafeControls(model);
          draw(model);
        });
        if (models.length > 0) {
          select.value = models[0];
          rebuildSafeControls(models[0]);
          draw(models[0]);
        }
      }

      init();
    </script>
  </body>
</html>
"""
    return (
        template
        .replace("__DATA__", data_json)
        .replace("__PALETTE__", palette_json)
        .replace("__LABELS__", labels_json)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="从目录 CSV 生成交互式 HTML 折线图")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help="输入目录，包含多个 *.csv")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_HTML, help="输出 HTML 文件路径")
    parser.add_argument("--serve", action="store_true", help="启动内置 HTTP 服务器在线查看")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    args = parser.parse_args()

    if args.serve:
        serve_http(args.input_dir, args.host, args.port)
        return

    data, labels = load_csv_series(args.input_dir)
    html = generate_html(data, labels)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        # 若为相对路径，默认写入到输入目录下
        out_path = Path(args.input_dir) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"已生成 HTML: {out_path}")


def serve_http(input_dir: str, host: str, port: int) -> None:
    input_dir = str(input_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # type: ignore[override]
            try:
                if self.path in ("/favicon.ico",):
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path in ("/", "/index.html"):
                    data, labels = load_csv_series(input_dir)
                    html = generate_html(data, labels)
                    content = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                if self.path == "/healthz":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                self.send_response(404)
                self.end_headers()
            except Exception as e:
                msg = f"Server error: {e}".encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def log_message(self, format: str, *args) -> None:  # noqa: A003 (match BaseHTTPRequestHandler)
            # 简洁日志
            print(f"[{self.address_string()}] {self.log_date_time_string()} " + format % args)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}  (输入目录: {input_dir})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()