// tinyqwen 前端共享工具 —— esc / api / reconcile / setText / toast / fmtSize / fmtMs
// 被 web_console(console.js) 和 viz.html(内联) 共同引用。

// HTML 转义(扩展 / 与行分隔符,防止 XSS)
function esc(s) {
  return String(s ?? "").replace(/[&<>"'/\u2028\u2029]/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;","/":"&#47;","\u2028":"&#8232;","\u2029":"&#8233;"}[c]));
}

// JSON → 安全的 <script> 内联(转义 </script> 与 <)
function escJsonForScript(obj) {
  return JSON.stringify(obj).replace(/</g, "\\u003c").replace(/-->/g, "--\\>");
}

// fetch 封装(加超时)
async function api(path, opts, timeoutMs = 15000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(path, {...opts, signal: ctrl.signal});
    let d = {};
    try { d = await r.json(); } catch (e) {}
    if (!r.ok) throw new Error(d.error || (r.status + " " + r.statusText));
    return d;
  } finally { clearTimeout(timer); }
}

// 增量更新:只 patch 变化的节点,不整页重建
function setText(id, text) {
  const e = document.getElementById(id);
  const s = String(text);
  if (e && e.textContent !== s) e.textContent = s;
}

// keyed reconcile:已有节点原地 update,新增才 build,消失才移除
function reconcile(container, items, keyOf, build, update) {
  const byKey = new Map();
  for (const child of [...container.children]) byKey.set(child.dataset.key, child);
  const seen = new Set();
  let prev = null;
  for (const item of items) {
    const key = keyOf(item);
    seen.add(key);
    let node = byKey.get(key);
    if (node) update(node, item);
    else { node = build(item); node.dataset.key = key; }
    const want = prev ? prev.nextSibling : container.firstChild;
    if (node !== want) container.insertBefore(node, want);
    prev = node;
  }
  for (const [key, node] of byKey) if (!seen.has(key)) node.remove();
}

// Toast
function toast(msg) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg; t.style.display = "block";
  setTimeout(() => t.style.display = "none", 3500);
}

// 数字格式化
function fmtSize(b) {
  if (b == null) return "?";
  if (b > 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b > 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b > 1e3) return (b / 1e3).toFixed(0) + " KB";
  return (b ?? 0) + " B";
}

function fmtMs(v) {
  v = Number(v) || 0;
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);
}

// 标签归一化(去掉 (Android) 后缀,与 Python load_history 对齐)
function normLabel(s) {
  return String(s ?? "").replace(/（Android）$/, "").replace(/\(Android\)$/, "");
}
