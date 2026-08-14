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
    "matvec": "#4e79a7",
    "norm": "#59a14f",
    "attention": "#f28e2b",
    "activation": "#e15759",
    "embed": "#b07aa1",
    "lm_head": "#76b7b2",
    "topk": "#edc948",
    "other": "#bab0ac",
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
    """解析 optimization_log.md 的汇总表，返回按表内顺序（时间序）的记录。"""
    text = Path(log_path).read_text()
    rows = []
    for line in text.splitlines():
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
        # 标记已回退的配置（日志里加粗注明），趋势图上淡化
        rolled_back = "回退" in cells[0]
        rows.append({
            "label": re.sub(r"\*\*", "", cells[0]),  # 去掉 markdown 加粗
            "commit": cells[1],
            "median_ms": med,
            "p95_ms": p95,
            "note": cells[6],
            "rolled_back": rolled_back,
        })
    return rows


# ---------- HTML 生成 ----------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>tinyqwen 性能可视化</title>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px;
         background: #fafafa; color: #222; }
  h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 36px;
       border-bottom: 1px solid #ddd; padding-bottom: 6px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 12px; }
  .chart { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
           padding: 16px; margin-bottom: 20px; overflow-x: auto; }
  #tooltip { position: fixed; pointer-events: none; background: rgba(0,0,0,.85);
             color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 12px;
             display: none; z-index: 99; white-space: pre; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px;
            margin: 8px 0; }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .legend i { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
  button { padding: 4px 12px; cursor: pointer; margin-bottom: 8px; }
  text { font-size: 11px; }
  .flame-rect { cursor: pointer; stroke: #fff; stroke-width: 1; }
  .flame-rect:hover { stroke: #000; }
</style>
</head>
<body>
<h1>tinyqwen 性能可视化</h1>
<div class="meta">__META__</div>
<div id="tooltip"></div>
__SECTIONS__
<script>
"use strict";
// ---------- 工具 ----------
const tooltip = document.getElementById("tooltip");
function showTip(evt, text) {
  tooltip.textContent = text;
  tooltip.style.display = "block";
  tooltip.style.left = (evt.clientX + 14) + "px";
  tooltip.style.top = (evt.clientY + 14) + "px";
}
function hideTip() { tooltip.style.display = "none"; }
function fmt(ms) { return ms >= 100 ? ms.toFixed(0) : ms.toFixed(2); }
function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

// ---------- 火焰图 ----------
function renderFlame(containerId, root, totalForPct) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const W = Math.max(container.clientWidth - 32, 900);
  const ROW = 24, GAP = 2;
  let maxDepth = 0;
  (function depth(n, d) { maxDepth = Math.max(maxDepth, d);
    n.children.forEach(c => depth(c, d + 1)); })(root, 0);
  const H = (maxDepth + 1) * (ROW + GAP);
  const svg = svgEl("svg", {width: W, height: H});
  const zoomTotal = root.value;

  function layout(node, x0, x1, depth) {
    const px0 = x0 * W, px1 = x1 * W;
    if (px1 - px0 < 0.5) return;  // 太窄不画
    const y = H - (depth + 1) * (ROW + GAP);
    const rect = svgEl("rect", {
      x: px0, y: y, width: px1 - px0, height: ROW, rx: 2,
      class: "flame-rect", fill: node.color});
    const pctVsTotal = node.value / totalForPct * 100;
    const pctVsZoom = node.value / zoomTotal * 100;
    const tip = node.name + "\\n" + fmt(node.value) + " ms\\n" +
                pctVsTotal.toFixed(2) + "% of total" +
                (pctVsZoom < 99.9 ? "\\n" + pctVsZoom.toFixed(1) + "% of view" : "");
    rect.addEventListener("mousemove", e => showTip(e, tip));
    rect.addEventListener("mouseleave", hideTip);
    if (node.children.length) {
      rect.addEventListener("click", () => renderFlame(containerId, node, totalForPct));
    }
    svg.appendChild(rect);
    // 宽度足够时显示文字
    if (px1 - px0 > 55) {
      const t = svgEl("text", {x: px0 + 4, y: y + ROW / 2 + 4});
      t.textContent = (px1 - px0 > 90 ? node.name : node.name.slice(0, 8)) +
                      " " + pctVsZoom.toFixed(1) + "%";
      t.style.pointerEvents = "none";
      svg.appendChild(t);
    }
    let cx = x0;
    for (const child of node.children) {
      const w = (x1 - x0) * (child.value / node.value);
      layout(child, cx, cx + w, depth + 1);
      cx += w;
    }
  }
  layout(root, 0, 1, 0);
  container.appendChild(svg);
}

// ---------- 时序图 ----------
function renderTimeline(containerId, tokens, warmup) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const W = Math.max(container.clientWidth - 32, 900), H = 220;
  const padL = 55, padB = 30, padT = 10;
  const svg = svgEl("svg", {width: W, height: H});
  const maxMs = Math.max(...tokens.map(t => t.latency_ms)) * 1.1;
  const n = tokens.length;
  const bw = (W - padL - 10) / n;
  // y 轴
  for (let g = 0; g <= 4; g++) {
    const v = maxMs * g / 4;
    const y = padT + (H - padT - padB) * (1 - g / 4);
    svg.appendChild(svgEl("line", {x1: padL, y1: y, x2: W - 5, y2: y,
      stroke: "#eee"}));
    const t = svgEl("text", {x: 4, y: y + 4, fill: "#888"});
    t.textContent = fmt(v) + "ms";
    svg.appendChild(t);
  }
  tokens.forEach((tok, i) => {
    const h = (H - padT - padB) * tok.latency_ms / maxMs;
    let color;
    if (tok.is_prefill) color = "#f28e2b";
    else if (i < tokens.filter(t => t.is_prefill).length + warmup) color = "#aec7e8";
    else color = "#4e79a7";
    const rect = svgEl("rect", {
      x: padL + i * bw + 1, y: H - padB - h, width: bw - 2, height: h,
      fill: color, rx: 1});
    rect.addEventListener("mousemove", e => showTip(e,
      "token #" + i + " (pos " + tok.pos + ")\\n" +
      (tok.is_prefill ? "prefill" : "decode") + "\\n" +
      fmt(tok.latency_ms) + " ms"));
    rect.addEventListener("mouseleave", hideTip);
    svg.appendChild(rect);
    if (i % Math.ceil(n / 20) === 0) {
      const t = svgEl("text", {x: padL + i * bw + bw / 2, y: H - padB + 14,
        "text-anchor": "middle", fill: "#888"});
      t.textContent = i;
      svg.appendChild(t);
    }
  });
  container.appendChild(svg);
}

// ---------- 环形图 ----------
function renderDonut(containerId, items) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const S = 260, cx = S / 2, cy = S / 2, r = 95;
  const svg = svgEl("svg", {width: S, height: S});
  const total = items.reduce((a, b) => a + b.value, 0);
  let angle = -Math.PI / 2;
  items.forEach(it => {
    const frac = it.value / total;
    const a2 = angle + frac * Math.PI * 2;
    const large = frac > 0.5 ? 1 : 0;
    const x1 = cx + r * Math.cos(angle), y1 = cy + r * Math.sin(angle);
    const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2);
    const path = svgEl("path", {
      d: `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`,
      fill: it.color, stroke: "#fff", "stroke-width": 1});
    path.addEventListener("mousemove", e => showTip(e,
      it.name + "\\n" + fmt(it.value) + " ms (" + (frac * 100).toFixed(1) + "%)"));
    path.addEventListener("mouseleave", hideTip);
    svg.appendChild(path);
    angle = a2;
  });
  // 中间挖空
  svg.appendChild(svgEl("circle", {cx, cy, r: r * 0.55, fill: "#fff"}));
  const t = svgEl("text", {x: cx, y: cy, "text-anchor": "middle",
    "font-size": 14, fill: "#333"});
  t.textContent = fmt(total) + " ms";
  svg.appendChild(t);
  container.appendChild(svg);
  // 图例
  const legend = document.createElement("div");
  legend.className = "legend";
  items.forEach(it => {
    legend.innerHTML += `<span><i style="background:${it.color}"></i>${it.name}
      ${(it.value / total * 100).toFixed(1)}%</span>`;
  });
  container.appendChild(legend);
}

// ---------- 趋势图 ----------
function renderTrend(containerId, rows, logScale) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const W = Math.max(container.clientWidth - 32, 900), H = 300;
  const padL = 60, padB = 80, padT = 15, padR = 20;
  const svg = svgEl("svg", {width: W, height: H});
  const mids = rows.map(r => r.median_ms);
  let lo = Math.min(...mids) * 0.8, hi = Math.max(...mids) * 1.2;
  const tf = logScale ? Math.log : x => x;
  lo = tf(lo); hi = tf(hi);
  const n = rows.length;
  const xPos = i => padL + (W - padL - padR) * (n === 1 ? 0.5 : i / (n - 1));
  const yPos = v => padT + (H - padT - padB) * (1 - (tf(v) - lo) / (hi - lo));
  // y 轴刻度
  const ticks = logScale ?
    [1, 2, 5, 10, 20, 50, 100, 200, 500].filter(v => tf(v) >= lo && tf(v) <= hi) :
    [0, 1, 2, 3, 4].map(i => Math.exp(lo) + (Math.exp(hi) - Math.exp(lo)) * i / 4);
  ticks.forEach(v => {
    const y = yPos(v);
    svg.appendChild(svgEl("line", {x1: padL, y1: y, x2: W - padR, y2: y,
      stroke: "#eee"}));
    const t = svgEl("text", {x: 6, y: y + 4, fill: "#888"});
    t.textContent = v + " ms";
    svg.appendChild(t);
  });
  // p95 线（淡）+ 中位线
  function polyline(key, color, width, dash) {
    const pts = rows.map((r, i) => `${xPos(i)},${yPos(r[key])}`).join(" ");
    svg.appendChild(svgEl("polyline", {points: pts, fill: "none",
      stroke: color, "stroke-width": width, "stroke-dasharray": dash || ""}));
  }
  polyline("p95_ms", "#e0a0a0", 1.5, "4 3");
  polyline("median_ms", "#4e79a7", 2.5);
  // 数据点 + x 轴标签
  rows.forEach((r, i) => {
    const dot = svgEl("circle", {cx: xPos(i), cy: yPos(r.median_ms), r: 4,
      fill: r.rolled_back ? "#ccc" : "#4e79a7",
      stroke: r.rolled_back ? "#999" : "#2a5a8a"});
    dot.addEventListener("mousemove", e => showTip(e,
      r.label + "\\nmedian " + r.median_ms.toFixed(2) + " ms/tok" +
      "\\np95 " + r.p95_ms.toFixed(2) + " ms" +
      "\\ncommit " + r.commit + (r.rolled_back ? "\\n（已回退）" : "")));
    dot.addEventListener("mouseleave", hideTip);
    svg.appendChild(dot);
    const t = svgEl("text", {x: xPos(i), y: H - padB + 14, fill: "#666",
      "font-size": 10, "text-anchor": "end"});
    t.setAttribute("transform", `rotate(-40 ${xPos(i)} ${H - padB + 14})`);
    t.textContent = r.label.length > 22 ? r.label.slice(0, 22) + "…" : r.label;
    svg.appendChild(t);
  });
  container.appendChild(svg);
}

// ---------- 数据 & 入口 ----------
const FLAME = __FLAME_DATA__;
const TOKENS = __TOKENS_DATA__;
const DONUT = __DONUT_DATA__;
const TREND = __TREND_DATA__;
const WARMUP = __WARMUP__;

document.addEventListener("DOMContentLoaded", () => {
  if (FLAME) renderFlame("flame", FLAME, FLAME.value);
  if (TOKENS && TOKENS.length) renderTimeline("timeline", TOKENS, WARMUP);
  if (DONUT) renderDonut("donut", DONUT);
  if (TREND && TREND.length) renderTrend("trend", TREND, true);
  window.addEventListener("resize", () => {
    if (FLAME) renderFlame("flame", FLAME, FLAME.value);
    if (TREND && TREND.length) renderTrend("trend", TREND, true);
  });
});
</script>
</body>
</html>
"""


def build_sections(profile: bool, trend: bool) -> str:
    parts = []
    if profile:
        parts.append("""
<h2>Op 火焰图</h2>
<div class="meta">宽度 = 耗时占比。点击某块可放大其子树；颜色 = op 类别。
lm_head / 各层 matvec 通常是带宽瓶颈大头。</div>
<div class="legend">
  <span><i style="background:#4e79a7"></i>matvec</span>
  <span><i style="background:#f28e2b"></i>attention</span>
  <span><i style="background:#59a14f"></i>norm</span>
  <span><i style="background:#e15759"></i>activation</span>
  <span><i style="background:#76b7b2"></i>lm_head</span>
  <span><i style="background:#b07aa1"></i>embed</span>
  <span><i style="background:#edc948"></i>topk</span>
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
<h2>优化历史趋势（对数刻度）</h2>
<div class="meta">来自 optimization_log.md 汇总表。蓝线 = decode 中位，
红色虚线 = p95，灰点 = 已回退的配置。越靠右越快。</div>
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
        # 给每个节点填 color
        def paint(node):
            node["color"] = cat_color(node.get("cat", "other"))
            for c in node["children"]:
                paint(c)
        paint(flame)

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

    html = HTML_TEMPLATE
    html = html.replace("__META__", " · ".join(meta_parts) or "—")
    html = html.replace("__SECTIONS__",
                        build_sections(profile_data is not None, bool(trend_rows)))
    html = html.replace("__FLAME_DATA__", json.dumps(flame))
    html = html.replace("__TOKENS_DATA__", json.dumps(tokens))
    html = html.replace("__DONUT_DATA__", json.dumps(donut))
    html = html.replace("__TREND_DATA__", json.dumps(trend_rows, ensure_ascii=False))
    html = html.replace("__WARMUP__", str(warmup))

    out = Path(args.out)
    out.write_text(html)
    print(f"  ✓ 已生成 {out}（独立 HTML，浏览器直接打开）")


if __name__ == "__main__":
    main()
