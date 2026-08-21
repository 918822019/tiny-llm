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

输入：
    - profile 子命令：runtime --profile-out 输出的 JSON 文件
    - trend 子命令：docs/optimization_log.md 日志文件
    - all 子命令：两者都需要

输出：
    - 独立 HTML 文件（内嵌 CSS/JS/SVG），浏览器直接打开即可
"""

# 启用延迟注解求值
from __future__ import annotations

# ---- 标准库导入 ----
import argparse       # 命令行参数解析（含子命令）
import json           # JSON 序列化/反序列化
import re             # 正则表达式（资源内联、日志解析）
import sys            # 系统退出
from pathlib import Path  # 路径操作

# ---------- 前端共享资源 ----------

# webui 目录路径：存放 CSS/JS/HTML 模板等前端资源
WEBUI_DIR = Path(__file__).resolve().parent / "webui"


def read_asset(rel_path: str) -> str:
    """从 tools/webui/ 读取一个共享前端文件（CSS/JS/HTML）。

    Args:
        rel_path: 相对于 WEBUI_DIR 的文件路径

    Returns:
        str: 文件文本内容
    """
    return (WEBUI_DIR / rel_path).read_text(encoding="utf-8")


def _esc_json_for_script(s: str) -> str:
    """转义 JSON 字符串中的 < 和 </script>，安全内联到 <script> 块。

    防止 JSON 数据中包含 "</script>" 时提前关闭 script 标签导致 XSS 或语法错误。

    Args:
        s: 待转义的 JSON 字符串

    Returns:
        str: 转义后的安全字符串
    """
    return s.replace("<", "\\u003c").replace("-->", "--\\>")


def inline_assets(html: str) -> str:
    """把外部引用的 CSS/JS 资源内联到 HTML 中，生成单文件。

    将 <link href="assets/X"> 替换为 <style>...</style>，
    将 <script src="assets/X"> 替换为 <script>...</script>。
    输出仍是单文件，可 file:// 直接打开，无需 HTTP 服务器。

    Args:
        html: 包含外部资源引用的 HTML 文本

    Returns:
        str: 内联后的完整 HTML
    """
    def repl_link(m):
        """替换 <link> 标签为内联 <style>。"""
        return "<style>\n" + read_asset(m.group(1)) + "\n</style>"
    # 匹配 <link ... href="assets/xxx" ...> 并替换
    html = re.sub(r'<link[^>]*href="assets/([^"]+)"[^>]*>', repl_link, html)

    def repl_script(m):
        """替换 <script src="..."> 为内联 <script>。"""
        attr = m.group(1) or ""  # 保留其他属性（如 type="module"）
        return "<script" + attr + ">\n" + read_asset(m.group(2)) + "\n</script>"
    # 匹配 <script ... src="assets/xxx" ...></script> 并替换
    html = re.sub(r'<script([^>]*)\ssrc="assets/([^"]+)"[^>]*>\s*</script>', repl_script, html)
    return html


# ---------- op 分类（与 profile_diff.py 同口径）----------

# 算子类别定义：{类别名: [关键词列表]}
CATEGORIES = {
    "matvec": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
               "down_proj", "lm_head"],       # 矩阵向量乘法（带宽瓶颈大头）
    "norm": ["layernorm", "rmsnorm"],          # 归一化层
    "attention": ["attention", "rope", "kv_append"],  # 注意力机制相关
    "activation": ["swiglu", "silu"],          # 激活函数
    "other": ["embed", "topk_argmax", "residual"],   # 其他算子
}

# 各类别的 CSS 颜色变量名
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
    """根据算子名称判断其所属类别。

    优先匹配特殊类别（lm_head、embed、topk），再按 CATEGORIES 关键词匹配，
    都不命中则归入 "other"。

    Args:
        op_name: 算子名称（如 "layer_0.q_proj" 或 "lm_head"）

    Returns:
        str: 类别名称
    """
    lower = op_name.lower()
    # 去掉前缀（如 "layer_0."），只保留算子本体名
    if "." in lower:
        lower = lower.split(".", 1)[1]
    # 优先匹配特殊类别
    if "lm_head" in lower:
        return "lm_head"
    if "embed" in lower:
        return "embed"
    if "topk" in lower:
        return "topk"
    # 按关键词匹配通用类别
    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in lower:
                return cat
    return "other"


def cat_color(cat: str) -> str:
    """获取类别对应的 CSS 颜色变量。

    Args:
        cat: 类别名称

    Returns:
        str: CSS var() 颜色值
    """
    return CAT_COLORS.get(cat, CAT_COLORS["other"])


# ---------- 火焰图树构建 ----------

def build_flame_tree(op_totals: dict) -> dict:
    """把扁平 op_totals 组织成层级树：root → (embed / layer_N / lm_head / topk) → 层内 op。

    火焰图需要层级结构；profiler 输出的是扁平的 {op_name: {calls, total_ms}}，
    这里按 "layer_N.op" 的模式拆分成两层树。

    Args:
        op_totals: profiler 输出的 {op_name: {calls, total_ms}} 字典

    Returns:
        dict: 树形结构 {"name", "value", "cat", "children": [...]}
    """
    root_children: dict[str, dict] = {}  # 根节点的子节点集合
    layer_re = re.compile(r"^layer_(\d+)\.(.+)$")  # 匹配 layer_N.op_name 模式

    for op, info in op_totals.items():
        ms = info["total_ms"]  # 该算子的总耗时
        m = layer_re.match(op)
        if m:
            # 属于某层的算子：归入 layer_N 分组
            layer_key = f"layer_{int(m.group(1)):02d}"  # 补零保证排序正确
            layer = root_children.setdefault(layer_key, {
                "name": layer_key, "value": 0.0, "children": [],
                "cat": "matvec"})
            layer["value"] += ms  # 累加层总耗时
            layer["children"].append({
                "name": m.group(2), "value": ms, "children": [],
                "cat": categorize_op(op)})
        else:
            # 非层内算子（embed、lm_head、topk 等）作为根的直接子节点
            root_children[op] = {
                "name": op, "value": ms, "children": [],
                "cat": categorize_op(op)}

    # 层内 op 按耗时降序排列（火焰图中宽的先出现）
    for node in root_children.values():
        if node["children"]:
            node["children"].sort(key=lambda c: -c["value"])

    children = sorted(root_children.values(), key=lambda c: c["name"])
    # lm_head 通常是大头，排到最后（火焰图最右），便于肉眼定位
    children.sort(key=lambda c: (c["name"] != "lm_head", c["name"]))
    total = sum(c["value"] for c in children)  # 总耗时
    return {"name": "total", "value": total, "cat": "other", "children": children}


# ---------- 优化历史解析 ----------

def parse_opt_log(log_path: str) -> list[dict]:
    """解析 optimization_log.md 的汇总表，返回按表内顺序（时间序）的记录。

    只扫描 `## 汇总表` 到下一个 `## ` 之间的表格行，忽略嵌入的线程扫描表等。
    `rolled_back` 只对真正回退的配置（如 fp32-float 标注"已回退"）生效，
    不再宽匹配 "回退"（误判 restore-double）。
    commit 非标准 hash 或 "本次" 时标记 null，不在 tooltip 显示垃圾值。

    Args:
        log_path: optimization_log.md 文件路径

    Returns:
        list[dict]: 每条记录的字典，包含 label、commit、median_ms、p95_ms 等字段
    """
    text = Path(log_path).read_text(encoding="utf-8")
    # section 锚定：只取 `## 汇总表` 到下一个 `## ` 之间
    m = re.search(r"^## 汇总表\s*$", text, re.MULTILINE)
    if not m:
        return []  # 找不到汇总表标题
    section = text[m.end():]  # 截取汇总表之后的内容
    m2 = re.search(r"^## ", section, re.MULTILINE)  # 找下一个二级标题
    if m2:
        section = section[:m2.start()]  # 截断到下一个 section 之前

    rows = []
    # commit hash 的正则：7+ 位十六进制，可选尾部 +
    commit_re = re.compile(r"^[0-9a-f]{7,}\+?$")
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue  # 非表格行跳过
        cells = [c.strip() for c in line.strip("|").split("|")]  # 按 | 分割单元格
        if len(cells) < 7:
            continue  # 列数不足跳过
        if "配置" in cells[0] or set(cells[0]) <= {"-", " "}:
            continue  # 表头 / 分隔行跳过
        try:
            med = float(cells[2])   # decode 中位延迟
            p95 = float(cells[3])   # decode P95 延迟
        except ValueError:
            continue  # 数值解析失败跳过
        label = re.sub(r"\*\*", "", cells[0])  # 去掉 markdown 加粗标记
        # rolled_back: 只匹配"已回退"/"证伪归档"/"已弃用"，不匹配"回退"
        rolled_back = bool(re.search(r"已回退|证伪归档|已弃用", label))
        # commit 过滤：非标准 hash 标记空串
        raw_commit = cells[1].strip()
        commit = raw_commit if commit_re.match(raw_commit) else None
        rows.append({
            "label": label,
            "commit": commit or "",
            "median_ms": med,
            "p95_ms": p95,
            "note": cells[6],        # 归因说明
            "rolled_back": rolled_back,
        })
    return rows


# ---------- HTML 生成 ----------

# HTML 模板已移至 tools/webui/viz/template.html，运行时读盘 + inline_assets 内联为单文件。
# CSS/JS 共享源在 tools/webui/{tokens,base}.css + {util,chart,palette}.js + viz/{viz.css,viz.js}。


def build_sections(profile: bool, trend: bool) -> str:
    """根据启用的可视化类型构建 HTML section 片段。

    这些是插入到模板 __SECTIONS__ 占位符中的 HTML 内容，定义了图表容器和说明文字。

    Args:
        profile: 是否包含 profile 可视化（火焰图 + 时序 + 饼图）
        trend: 是否包含优化历史趋势图

    Returns:
        str: HTML section 片段拼接结果
    """
    parts = []
    if profile:
        # profile 可视化的三个图表区域（注意：这是 HTML 字符串，属于代码逻辑不可修改）
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
        # 趋势图区域
        parts.append("""
<h2>优化历史趋势</h2>
<div class="meta">来自 optimization_log.md 汇总表。蓝线 = decode 中位，
红色虚线 = p95，灰点 = 已回退的配置。越靠右越快。
可切换对数/线性刻度。</div>
<div class="chart" id="trend"></div>
""")
    return "\n".join(parts)


def main() -> None:
    """visualize.py 的主入口函数。

    解析子命令 → 加载数据 → 组装 HTML → 内联资源 → 写入文件。
    """
    # 创建带子命令的参数解析器
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # profile 子命令：单 profile 可视化
    p_prof = sub.add_parser("profile", help="单 profile 可视化")
    p_prof.add_argument("profile", help="runtime --profile-out 输出的 JSON")
    p_prof.add_argument("-o", "--out", default="profile_viz.html")

    # trend 子命令：优化历史趋势
    p_trend = sub.add_parser("trend", help="优化历史趋势")
    p_trend.add_argument("--log", default="docs/optimization_log.md")
    p_trend.add_argument("-o", "--out", default="trend.html")

    # all 子命令：profile + 历史合一
    p_all = sub.add_parser("all", help="profile + 历史，一个 HTML 全出")
    p_all.add_argument("profile", help="runtime --profile-out 输出的 JSON")
    p_all.add_argument("--log", default="docs/optimization_log.md")
    p_all.add_argument("-o", "--out", default="viz.html")

    args = ap.parse_args()

    # ---- 加载 profile 数据 ----
    profile_data = None
    if args.cmd in ("profile", "all"):
        pp = Path(args.profile)
        if not pp.exists():
            sys.exit(f"error: profile not found: {pp}")
        profile_data = json.loads(pp.read_text())

    # ---- 加载优化历史 ----
    trend_rows = []
    if args.cmd in ("trend", "all"):
        lp = Path(args.log)
        if not lp.exists():
            sys.exit(f"error: log not found: {lp}")
        trend_rows = parse_opt_log(str(lp))
        if not trend_rows:
            print(f"  ⚠️ {args.log} 里没解析出有效表格行", file=sys.stderr)

    # ---- 组装可视化数据 ----
    flame = None     # 火焰图树数据
    tokens = None    # token 时序数据
    donut = None     # 饼图数据
    meta_parts = []  # 元信息文本片段

    if profile_data:
        op_totals = profile_data.get("op_totals", {})
        # 构建火焰图层级树
        flame = build_flame_tree(op_totals)

        # color 由前端 viz.js 从 cat 派生（减少 JSON 体积 ~15KB），不在后端着色

        # 提取 token 时序数据（延迟、是否 prefill、位置）
        tokens = [{"latency_ms": t["latency_ms"], "is_prefill": t["is_prefill"],
                   "pos": t["pos"]} for t in profile_data.get("tokens", [])]

        # 计算 op 类别占比（用于饼图）
        cat_sum: dict[str, float] = {}
        for op, info in op_totals.items():
            cat = categorize_op(op)
            cat_sum[cat] = cat_sum.get(cat, 0) + info["total_ms"]
        # 按耗时降序排列
        donut = [{"name": k, "value": round(v, 2), "color": cat_color(k)}
                 for k, v in sorted(cat_sum.items(), key=lambda kv: -kv[1])]

        # 构建元信息文本
        meta_parts.append(
            f"model={profile_data.get('model', '?')} · "
            f"precision={profile_data.get('precision', '?')} · "
            f"{profile_data.get('prompt_tokens', '?')} prompt + "
            f"{profile_data.get('generated_tokens', '?')} generated tokens · "
            f"decode avg {profile_data.get('decode_avg_ms', 0):.2f} ms/tok")

    if trend_rows:
        meta_parts.append(f"优化历史 {len(trend_rows)} 条记录")

    # warmup 与 bench.py 口径一致（丢弃前 4 个 decode token）
    warmup = 4

    # ---- 读模板 + 内联 CSS/JS 为单文件 ----
    import html as html_mod  # 标准库 html 模块（escape 函数）
    html = read_asset("viz/template.html")  # 读取 HTML 模板
    html = inline_assets(html)              # 内联外部资源
    # 替换模板中的占位符
    html = html.replace("__META__", html_mod.escape(" · ".join(meta_parts) or "—"))
    html = html.replace("__SECTIONS__",
                        build_sections(profile_data is not None, bool(trend_rows)))
    html = html.replace("__FLAME_DATA__", _esc_json_for_script(json.dumps(flame)))
    html = html.replace("__TOKENS_DATA__", _esc_json_for_script(json.dumps(tokens)))
    html = html.replace("__DONUT_DATA__", _esc_json_for_script(json.dumps(donut)))
    html = html.replace("__TREND_DATA__", _esc_json_for_script(json.dumps(trend_rows, ensure_ascii=False)))
    html = html.replace("__WARMUP__", str(warmup))

    # 写入输出文件
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"  ✓ 已生成 {out}（独立 HTML，浏览器直接打开）")


# 脚本直接运行入口
if __name__ == "__main__":
    main()
