// tinyqwen 前端共享 SVG 图表原语 —— el / responsiveWidth / linearScale / logScale / axisY / axisX / tooltip
// 合并自 web_console.el() + visualize.svgEl(),统一坐标轴/网格线/tooltip。

const SVG_NS = "http://www.w3.org/2000/svg";

// SVG 元素工厂(合并 el + svgEl)
function el(tag, attrs, children) {
  const e = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v != null) e.setAttribute(k, v);
  }
  if (children) for (const c of children) e.appendChild(c);
  return e;
}

// 响应式宽度:取容器宽度减 padding,不低于 min
function responsiveWidth(container, opts = {}) {
  const { min = 480, pad = 32 } = opts;
  return Math.max((container.clientWidth || 800) - pad, min);
}

// 线性比例尺
function linearScale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  const span = d1 - d0 || 1;
  return v => r0 + (r1 - r0) * ((v - d0) / span);
}

// 对数比例尺(底数统一 log10)
function logScale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range;
  const ld = [Math.log10(Math.max(d0, 1e-9)), Math.log10(Math.max(d1, 1e-9))];
  const span = ld[1] - ld[0] || 1;
  return v => r0 + (r1 - r0) * ((Math.log10(Math.max(v, 1e-9)) - ld[0]) / span);
}

// 生成 ~n 个"漂亮"的刻度值(轻量版 d3.ticks)
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (n * step) / span;
  let mult = 10;
  if (err <= 1) mult = 10;
  else if (err <= 2) mult = 5;
  else if (err <= 5) mult = 2;
  const tickStep = step * mult;
  const start = Math.ceil(lo / tickStep) * tickStep;
  const ticks = [];
  for (let v = start; v <= hi + 1e-9; v += tickStep) ticks.push(Number(v.toFixed(6)));
  return ticks;
}

// Y 轴:网格线 + 标签
function axisY(svg, scale, opts = {}) {
  const { ticks = 5, x1, x2, padT, padB, H, format = fmtMs } = opts;
  const vals = typeof ticks === "number" ? niceTicks(opts.domainLo, opts.domainHi, ticks) : ticks;
  for (const v of vals) {
    const y = scale(v);
    if (y < padT - 1 || y > H - padB + 1) continue;
    svg.appendChild(el("line", {x1, y1: y, x2, y2: y, stroke: "var(--border-soft)"}));
    const t = el("text", {x: 4, y: y + 4, fill: "var(--muted-2)"});
    t.textContent = format(v);
    svg.appendChild(t);
  }
}

// X 轴:刻度标签
function axisX(svg, xPos, opts = {}) {
  const { n, ticks = 5, y, format = v => v } = opts;
  const step = Math.max(1, Math.ceil(n / ticks));
  for (let i = 0; i < n; i += step) {
    const t = el("text", {x: xPos(i), y, "text-anchor": "middle", fill: "var(--muted-2)"});
    t.textContent = format(i);
    svg.appendChild(t);
  }
}

// Tooltip 绑定(mousemove + mouseleave)
function attachTooltip(node, textFn) {
  const tip = document.getElementById("tooltip");
  if (!tip) return;
  node.addEventListener("mousemove", e => {
    tip.textContent = textFn(e);
    tip.style.display = "block";
    tip.style.left = (e.clientX + 14) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  node.addEventListener("mouseleave", () => { tip.style.display = "none"; });
}

// 空数据/加载中/错误 三态
function renderState(container, state, message = "") {
  const div = document.createElement("div");
  div.className = "hint";
  div.textContent = message;
  container.replaceChildren(div);
}
