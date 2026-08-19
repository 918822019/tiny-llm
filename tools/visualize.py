#!/usr/bin/env python3
"""tinyqwen 性能可视化：profile / 优化历史 → 独立 HTML（零外部依赖）。

生成的 HTML 内嵌 SVG + 原生 JS，单文件可直接打开，不依赖网络/CDN。

子命令：
    # 单次 profile 可视化：火焰图 + token 时序 + op 占比
    python tools/visualize.py profile profile_macos.json -o profile_viz.html

    # 优化历史趋势：从 optimization_log.md 的汇总表提取所有记录
    python tools/visualize.py trend -o trend.html

    # 一次全出：profile 三联图 + 历史趋势，同一个 HTML
    python tools/visualize.py all profile_macos.json -o viz.html

用法示例（跑完 bench 立刻看图）：
    ./build/runtime/tinyqwen --model model.tqwen --tokens 105538,59975,100132 \\
        --max-new-tokens 32 --profile-out /tmp/p.json
    python tools/visualize.py all /tmp/p.json -o /tmp/viz.html && open /tmp/viz.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------- 前端共享资源 ----------

WEBUI_DIR = Path(__file__).resolve().parent / "webui"


def read_asset(rel_path: str) -> str:
    """从 tools/webui/ 读一个共享前端文件(CSS/JS/HTML)。"""
    return (WEBUI_DIR / rel_path).read_text(encoding="utf-8")


def _esc_json_for_script(s: str) -> str:
    """转义 JSON 字符串中的 < 和 </script>,安全内联到 <script> 块。"""
    return s.replace("<", "\\u003c").replace("-->", "--\\>")


def inline_assets(html: str) -> str:
    """把 <link href="assets/X"> → <style>,<script src="assets/X"> → <script> 内联。
    输出仍是单文件,可 file:// 直接打开。"""
    def repl_link(m):
        return "<style>\n" + read_asset(m.group(1)) + "\n</style>"
    html = re.sub(r'<link[^>]*href="assets/([^"]+)"[^>]*>', repl_link, html)

    def repl_script(m):
        attr = m.group(1) or ""
        return "<script" + attr + ">\n" + read_asset(m.group(2)) + "\n</script>"
    html = re.sub(r'<script([^>]*)\ssrc="assets/([^"]+)"[^>]*>\s*</script>', repl_script, html)
    return html


# ---------- op 分类（与 profile_diff.py 同口径）----------

CATEGORIES = {
    "matvec": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
               "down_proj", "lm_head"],
    "norm": ["layernorm", "rmsnorm"],
    "attention": ["attention", "rope", "kv_append"],
    "activation": ["swiglu", "silu"],
    "other": ["embed", "topk_argmax", "residual"],
}

CAT_COLORS = {
    "matvec": "var(--c-matvec)",
    "norm": "var(--c-norm)",
    "attention": "var(--c-attention)",
    "activation": "var(--c-activation)",
    "embed": "var(--c-embed)",
    "lm_head": "var(--c-lmhead)",
    "topk": "var(--c-topk)",
    "other": "var(--c-other)",
}


def categorize_op(op_name: str) -> str:
    lower = op_name.lower()
    if "." in lower:
        lower = lower.split(".", 1)[1]
    if "lm_head" in lower:
        return "lm_head"
    if "embed" in lower:
        return "embed"
    if "topk" in lower:
        return "topk"
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in lower:
                return cat
    return "other"


def cat_color(cat: str) -> str:
    return CAT_COLORS.get(cat, CAT_COLORS["other"])


# ---------- 火焰图树构建 ----------

def build_flame_tree(op_totals: dict) -> dict:
    """把扁平 op_totals 组织成层级树：root → (embed / layer_N / lm_head / topk) → 层内 op。

    op_totals: {op_name: {calls, total_ms}}
    """
    root_children: dict[str, dict] = {}
    layer_re = re.compile(r"^layer_(\d+)\.(.+)$")

    for op, info in op_totals.items():
        ms = info["total_ms"]
        m = layer_re.match(op)
        if m:
            layer_key = f"layer_{int(m.group(1)):02d}"  # 补零保证排序
            layer = root_children.setdefault(layer_key, {
                "name": layer_key, "value": 0.0, "children": [],
                "cat": "matvec"})
            layer["value"] += ms
            layer["children"].append({
                "name": m.group(2), "value": ms, "children": [],
                "cat": categorize_op(op)})
        else:
            root_children[op] = {
                "name": op, "value": ms, "children": [],
                "cat": categorize_op(op)}

    # 层内 op 按耗时降序
    for node in root_children.values():
        if node["children"]:
            node["children"].sort(key=lambda c: -c["value"])

    children = sorted(root_children.values(), key=lambda c: c["name"])
    # lm_head 通常是大头，排到最后（火焰图最右），便于肉眼定位
    children.sort(key=lambda c: (c["name"] != "lm_head", c["name"]))
    total = sum(c["value"] for c in children)
    return {"name": "total", "value": total, "cat": "other", "children": children}


# ---------- 优化历史解析 ----------

def parse_opt_log(log_path: str) -> list[dict]:
    """解析 optimization_log.md 的汇总表，返回按表内顺序（时间序）的记录。

    只扫描 `## 汇总表` 到下一个 `## ` 之间的表格行，忽略嵌入的线程扫描表等。
    `rolled_back` 只对真正回退的配置（如 fp32-float 标注"已回退"）生效，
    不再宽匹配 "回退"（误判 restore-double）。
    commit 非标准 hash 或 "本次" 时标记 null，不在 tooltip 显示垃圾值。
    """
    text = Path(log_path).read_text(encoding="utf-8")
    # section 锚定:只取 `## 汇总表` 到下一个 `## ` 之间
    m = re.search(r"^## 汇总表\s*$", text, re.MULTILINE)
    if not m:
        return []
    section = text[m.end():]
    m2 = re.search(r"^## ", section, re.MULTILINE)
    if m2:
        section = section[:m2.start()]

    rows = []
    commit_re = re.compile(r"^[0-9a-f]{7,}\+?$")
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        if "配置" in cells[0] or set(cells[0]) <= {"-", " "}:
            continue  # 表头 / 分隔行
        try:
            med = float(cells[2])
            p95 = float(cells[3])
        except ValueError:
            continue
        label = re.sub(r"\*\*", "", cells[0])  # 去掉 markdown 加粗
        # rolled_back: 只匹配"已回退"/"证伪归档",不匹配"回退"(restore-double 是恢复不是回退)
        rolled_back = bool(re.search(r"已回退|证伪归档|已弃用", label))
        # commit 过滤:非标准 hash 标记 null
        raw_commit = cells[1].strip()
        commit = raw_commit if commit_re.match(raw_commit) else None
        rows.append({
            "label": label,
            "commit": commit or "",
            "median_ms": med,
            "p95_ms": p95,
            "note": cells[6],
            "rolled_back": rolled_back,
        })
    return rows


# ---------- HTML 生成 ----------

# HTML 模板已移至 tools/webui/viz/template.html,运行时读盘 + inline_assets 内联为单文件。
# CSS/JS 共享源在 tools/webui/{tokens,base}.css + {util,chart,palette}.js + viz/{viz.css,viz.js}。


def build_sections(profile: bool, trend: bool) -> str:
    parts = []
    if profile:
        parts.append("""
<h2>Op 火焰图</h2>
<div class="meta">宽度 = 耗时占比。点击某块可放大其子树；颜色 = op 类别。
lm_head / 各层 matvec 通常是带宽瓶颈大头。</div>
<div class="legend">
  <span><i style="background:var(--c-matvec)"></i>matvec</span>
  <span><i style="background:var(--c-attention)"></i>attention</span>
  <span><i style="background:var(--c-norm)"></i>norm</span>
  <span><i style="background:var(--c-activation)"></i>activation</span>
  <span><i style="background:var(--c-lmhead)"></i>lm_head</span>
  <span><i style="background:var(--c-embed)"></i>embed</span>
  <span><i style="background:var(--c-topk)"></i>topk</span>
  <span><i style="background:var(--c-other)"></i>other</span>
</div>
<div class="chart" id="flame"></div>

<h2>Token 延迟时序</h2>
<div class="meta">橙 = prefill，浅蓝 = decode 预热（统计时丢弃），蓝 = 稳态 decode。
稳态段若逐渐变慢，提示设备热降频或内存压力。</div>
<div class="chart" id="timeline"></div>

<h2>Op 类别占比</h2>
<div class="chart" id="donut" style="display:flex;align-items:center;gap:24px"></div>
""")
    if trend:
        parts.append("""
<h2>优化历史趋势</h2>
<div class="meta">来自 optimization_log.md 汇总表。蓝线 = decode 中位，
红色虚线 = p95，灰点 = 已回退的配置。越靠右越快。
可切换对数/线性刻度。</div>
<div class="chart" id="trend"></div>
""")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prof = sub.add_parser("profile", help="单 profile 可视化")
    p_prof.add_argument("profile", help="runtime --profile-out 输出的 JSON")
    p_prof.add_argument("-o", "--out", default="profile_viz.html")

    p_trend = sub.add_parser("trend", help="优化历史趋势")
    p_trend.add_argument("--log", default="docs/optimization_log.md")
    p_trend.add_argument("-o", "--out", default="trend.html")

    p_all = sub.add_parser("all", help="profile + 历史，一个 HTML 全出")
    p_all.add_argument("profile", help="runtime --profile-out 输出的 JSON")
    p_all.add_argument("--log", default="docs/optimization_log.md")
    p_all.add_argument("-o", "--out", default="viz.html")

    args = ap.parse_args()

    profile_data = None
    if args.cmd in ("profile", "all"):
        pp = Path(args.profile)
        if not pp.exists():
            sys.exit(f"error: profile not found: {pp}")
        profile_data = json.loads(pp.read_text())

    trend_rows = []
    if args.cmd in ("trend", "all"):
        lp = Path(args.log)
        if not lp.exists():
            sys.exit(f"error: log not found: {lp}")
        trend_rows = parse_opt_log(str(lp))
        if not trend_rows:
            print(f"  ⚠️ {args.log} 里没解析出有效表格行", file=sys.stderr)

    # 组装数据
    flame = None
    tokens = None
    donut = None
    meta_parts = []

    if profile_data:
        op_totals = profile_data.get("op_totals", {})
        flame = build_flame_tree(op_totals)

        # color 由前端 viz.js 从 cat 派生(减少 JSON 体积 ~15KB),不再 paint

        tokens = [{"latency_ms": t["latency_ms"], "is_prefill": t["is_prefill"],
                   "pos": t["pos"]} for t in profile_data.get("tokens", [])]

        # 类别占比
        cat_sum: dict[str, float] = {}
        for op, info in op_totals.items():
            cat = categorize_op(op)
            cat_sum[cat] = cat_sum.get(cat, 0) + info["total_ms"]
        donut = [{"name": k, "value": round(v, 2), "color": cat_color(k)}
                 for k, v in sorted(cat_sum.items(), key=lambda kv: -kv[1])]

        meta_parts.append(
            f"model={profile_data.get('model', '?')} · "
            f"precision={profile_data.get('precision', '?')} · "
            f"{profile_data.get('prompt_tokens', '?')} prompt + "
            f"{profile_data.get('generated_tokens', '?')} generated tokens · "
            f"decode avg {profile_data.get('decode_avg_ms', 0):.2f} ms/tok")

    if trend_rows:
        meta_parts.append(f"优化历史 {len(trend_rows)} 条记录")

    # warmup 与 bench.py 口径一致
    warmup = 4

    # 读模板 + 内联 CSS/JS 为单文件
    import html as html_mod
    html = read_asset("viz/template.html")
    html = inline_assets(html)
    html = html.replace("__META__", html_mod.escape(" · ".join(meta_parts) or "—"))
    html = html.replace("__SECTIONS__",
                        build_sections(profile_data is not None, bool(trend_rows)))
    html = html.replace("__FLAME_DATA__", _esc_json_for_script(json.dumps(flame)))
    html = html.replace("__TOKENS_DATA__", _esc_json_for_script(json.dumps(tokens)))
    html = html.replace("__DONUT_DATA__", _esc_json_for_script(json.dumps(donut)))
    html = html.replace("__TREND_DATA__", _esc_json_for_script(json.dumps(trend_rows, ensure_ascii=False)))
    html = html.replace("__WARMUP__", str(warmup))

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"  ✓ 已生成 {out}（独立 HTML，浏览器直接打开）")


if __name__ == "__main__":
    main()
