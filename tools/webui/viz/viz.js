"use strict";
// viz.html 四个图表:flamegraph / token timeline / op donut / 优化趋势
// 修复:flame 面包屑+resize 保状态+pruned 计数;timeline resize;donut createElement legend;trend logScale toggle+linear 数学修正+平台分线型

// ---------- 火焰图 ----------
let _flameCurrentView = null;  // 保留当前 zoom 状态

function renderFlame(containerId, root, totalForPct) {
  const container = document.getElementById(containerId);
  container.replaceChildren();

  // 面包屑
  const crumb = document.createElement("div");
  crumb.className = "flame-breadcrumb";
  if (_flameCurrentView && _flameCurrentView !== root) {
    const rootLink = document.createElement("a");
    rootLink.textContent = "root";
    rootLink.onclick = () => { _flameCurrentView = null; renderFlame(containerId, FLAME, FLAME.value); };
    crumb.appendChild(rootLink);
    let chain = [];
    let n = _flameCurrentView;
    while (n && n !== root && n._parent) { chain.unshift(n); n = n._parent; }
    chain.forEach(node => {
      const sep = document.createElement("span");
      sep.className = "sep"; sep.textContent = "›";
      crumb.appendChild(sep);
      const a = document.createElement("a");
      a.textContent = node.name;
      a.onclick = () => { _flameCurrentView = node; renderFlame(containerId, node, totalForPct); };
      crumb.appendChild(a);
    });
  }
  container.appendChild(crumb);

  const W = responsiveWidth(container);
  const ROW = 24, GAP = 2;
  let maxDepth = 0;
  let prunedCount = 0;
  (function depth(n, d) { maxDepth = Math.max(maxDepth, d);
    n.children.forEach(c => depth(c, d + 1)); })(root, 0);
  const H = (maxDepth + 1) * (ROW + GAP);
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  const zoomTotal = root.value;

  function layout(node, x0, x1, depth) {
    const px0 = x0 * W, px1 = x1 * W;
    if (px1 - px0 < 0.5) { prunedCount++; return; }
    const y = H - (depth + 1) * (ROW + GAP);
    const color = node.color || catColor(node.cat || "other");
    const rect = el("rect", {
      x: px0, y: y, width: px1 - px0, height: ROW, rx: 2,
      class: "flame-rect"});
    rect.style.fill = color;
    const pctVsTotal = node.value / totalForPct * 100;
    const pctVsZoom = node.value / zoomTotal * 100;
    const tip = node.name + "\n" + fmtMs(node.value) + " ms\n" +
                pctVsTotal.toFixed(2) + "% of total" +
                (pctVsZoom < 99.9 ? "\n" + pctVsZoom.toFixed(1) + "% of view" : "");
    attachTooltip(rect, () => tip);
    if (node.children && node.children.length) {
      rect.addEventListener("click", () => {
        node._parent = root;
        _flameCurrentView = node;
        renderFlame(containerId, node, totalForPct);
      });
    }
    svg.appendChild(rect);
    if (px1 - px0 > 55) {
      const t = el("text", {x: px0 + 4, y: y + ROW / 2 + 4});
      t.textContent = (px1 - px0 > 90 ? node.name : node.name.slice(0, 8)) +
                      " " + pctVsZoom.toFixed(1) + "%";
      t.style.pointerEvents = "none";
      svg.appendChild(t);
    }
    let cx = x0;
    for (const child of node.children || []) {
      const w = (x1 - x0) * (child.value / node.value);
      layout(child, cx, cx + w, depth + 1);
      cx += w;
    }
  }
  layout(root, 0, 1, 0);
  container.appendChild(svg);

  // pruned 计数提示
  if (prunedCount > 0) {
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = `共 ${maxDepth + 1} 层,已合并 ${prunedCount} 项 < 0.5px`;
    container.appendChild(hint);
  }
}

// ---------- 时序图 ----------
function renderTimeline(containerId, tokens, warmup) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  const W = responsiveWidth(container), H = 220;
  const padL = 55, padB = 30, padT = 10;
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  const maxMs = Math.max(...tokens.map(t => t.latency_ms)) * 1.1;
  const n = tokens.length;
  const bw = (W - padL - 10) / n;
  const prefillCount = tokens.filter(t => t.is_prefill).length;
  // y 轴
  for (let g = 0; g <= 4; g++) {
    const v = maxMs * g / 4;
    const y = padT + (H - padT - padB) * (1 - g / 4);
    svg.appendChild(el("line", {x1: padL, y1: y, x2: W - 5, y2: y, stroke: "var(--border-soft)"}));
    const t = el("text", {x: 4, y: y + 4, fill: "var(--muted-2)"});
    t.textContent = fmtMs(v) + "ms";
    svg.appendChild(t);
  }
  tokens.forEach((tok, i) => {
    const h = (H - padT - padB) * tok.latency_ms / maxMs;
    let color;
    if (tok.is_prefill) color = "var(--c-attention)";
    else if (i < prefillCount + warmup) color = "#aec7e8";
    else color = "var(--c-matvec)";
    const rect = el("rect", {
      x: padL + i * bw + 1, y: H - padB - h, width: bw - 2, height: h, rx: 1});
    rect.style.fill = color;
    attachTooltip(rect, () =>
      "token #" + i + " (pos " + tok.pos + ")\n" +
      (tok.is_prefill ? "prefill" : "decode") + "\n" +
      fmtMs(tok.latency_ms) + " ms");
    svg.appendChild(rect);
    if (i % Math.ceil(n / 20) === 0) {
      const t = el("text", {x: padL + i * bw + bw / 2, y: H - padB + 14,
        "text-anchor": "middle", fill: "var(--muted-2)"});
      t.textContent = i;
      svg.appendChild(t);
    }
  });
  container.appendChild(svg);
}

// ---------- 环形图 ----------
function renderDonut(containerId, items) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  const S = 260, cx = S / 2, cy = S / 2, r = 95;
  const svg = el("svg", {width: S, height: S, viewBox: `0 0 ${S} ${S}`});
  const total = items.reduce((a, b) => a + b.value, 0);
  let angle = -Math.PI / 2;
  items.forEach(it => {
    const frac = it.value / total;
    const a2 = angle + frac * Math.PI * 2;
    const large = frac > 0.5 ? 1 : 0;
    const x1 = cx + r * Math.cos(angle), y1 = cy + r * Math.sin(angle);
    const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2);
    const path = el("path", {
      d: `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`,
      stroke: "var(--surface)", "stroke-width": 1});
    path.style.fill = it.color;
    attachTooltip(path, () =>
      it.name + "\n" + fmtMs(it.value) + " ms (" + (frac * 100).toFixed(1) + "%)");
    svg.appendChild(path);
    angle = a2;
  });
  // 中间挖空
  svg.appendChild(el("circle", {cx, cy, r: r * 0.55, fill: "var(--surface)"}));
  const t = el("text", {x: cx, y: cy, "text-anchor": "middle", "font-size": 14, fill: "var(--text)"});
  t.textContent = fmtMs(total) + " ms";
  svg.appendChild(t);
  container.appendChild(svg);
  // 图例(createElement,消除 innerHTML 注入面)
  const legend = document.createElement("div");
  legend.className = "legend";
  items.forEach(it => {
    const span = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = it.color;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(
      it.name + " " + (it.value / total * 100).toFixed(1) + "%"));
    legend.appendChild(span);
  });
  container.appendChild(legend);
}

// ---------- 趋势图 ----------
function renderTrend(containerId, rows, logScale) {
  const container = document.getElementById(containerId);
  container.replaceChildren();

  // log/linear 切换按钮
  const controls = document.createElement("div");
  controls.className = "trend-controls";
  const btnLog = document.createElement("button");
  btnLog.textContent = "对数刻度"; btnLog.className = logScale ? "active" : "";
  const btnLin = document.createElement("button");
  btnLin.textContent = "线性刻度"; btnLin.className = !logScale ? "active" : "";
  btnLog.onclick = () => { window._trendLogScale = true; renderTrend(containerId, rows, true); };
  btnLin.onclick = () => { window._trendLogScale = false; renderTrend(containerId, rows, false); };
  controls.appendChild(btnLog); controls.appendChild(btnLin);
  container.appendChild(controls);

  const W = responsiveWidth(container), H = 300;
  const padL = 60, padB = 80, padT = 15, padR = 20;
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  const mids = rows.map(r => r.median_ms);
  let lo = Math.min(...mids) * 0.8, hi = Math.max(...mids) * 1.2;

  const tf = logScale ? (x => Math.log10(Math.max(x, 1e-9))) : (x => x);
  lo = tf(lo); hi = tf(hi);
  const n = rows.length;
  const xPos = i => padL + (W - padL - padR) * (n === 1 ? 0.5 : i / (n - 1));
  const yPos = v => padT + (H - padT - padB) * (1 - (tf(v) - lo) / (hi - lo || 1));

  // y 轴刻度
  let ticks;
  if (logScale) {
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter(v => tf(v) >= lo && tf(v) <= hi);
  } else {
    ticks = niceTicks(lo, hi, 5);
  }
  ticks.forEach(v => {
    const y = yPos(v);
    svg.appendChild(el("line", {x1: padL, y1: y, x2: W - padR, y2: y, stroke: "var(--border-soft)"}));
    const t = el("text", {x: 6, y: y + 4, fill: "var(--muted-2)"});
    t.textContent = fmtMs(v) + " ms";
    svg.appendChild(t);
  });

  // p95 线 + 中位线
  function polyline(key, color, width, dash) {
    const pts = rows.map((r, i) => `${xPos(i)},${yPos(r[key])}`).join(" ");
    const pl = el("polyline", {points: pts, "stroke-width": width, "stroke-dasharray": dash || ""});
    pl.style.fill = "none";
    pl.style.stroke = color;
    svg.appendChild(pl);
  }
  polyline("p95_ms", "#e0a0a0", 1.5, "4 3");
  polyline("median_ms", "var(--c-matvec)", 2.5);

  // 数据点 + x 轴标签
  rows.forEach((r, i) => {
    const dot = el("circle", {cx: xPos(i), cy: yPos(r.median_ms), r: 4});
    dot.style.fill = r.rolled_back ? "var(--muted-3)" : "var(--c-matvec)";
    dot.style.stroke = r.rolled_back ? "var(--muted-2)" : "var(--c-matvec)";
    attachTooltip(dot, () =>
      r.label + "\nmedian " + r.median_ms.toFixed(2) + " ms/tok" +
      "\np95 " + r.p95_ms.toFixed(2) + " ms" +
      (r.commit ? "\ncommit " + r.commit : "") +
      (r.rolled_back ? "\n（已回退）" : ""));
    svg.appendChild(dot);
    const t = el("text", {x: xPos(i), y: H - padB + 14, fill: "var(--muted)",
      "font-size": 10, "text-anchor": "end"});
    t.setAttribute("transform", `rotate(-40 ${xPos(i)} ${H - padB + 14})`);
    t.textContent = r.label.length > 22 ? r.label.slice(0, 22) + "…" : r.label;
    svg.appendChild(t);
  });
  container.appendChild(svg);
}
