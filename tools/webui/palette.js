// tinyqwen 前端共享色板 —— op 类别 + 精度/状态
// 所有 hex 值定义在 tokens.css 的 CSS 变量里,JS 只引 var 名。
// 这样浅/深色模式自动切换,且改一个色 = 两边同步变。

// Op 类别 → CSS 变量名
const OP_CATS = {
  matvec:    "var(--c-matvec)",
  norm:      "var(--c-norm)",
  attention: "var(--c-attention)",
  activation:"var(--c-activation)",
  embed:     "var(--c-embed)",
  lm_head:   "var(--c-lmhead)",
  topk:      "var(--c-topk)",
  other:     "var(--c-other)",
};

function catColor(cat) {
  return OP_CATS[cat] || OP_CATS.other;
}

// 精度/状态 → CSS 变量名
const PRECISION = {
  fp16:     "var(--c-fp16)",
  f16:      "var(--c-fp16)",
  fp32:     "var(--c-fp32)",
  i4:       "var(--c-i4)",
  int4:     "var(--c-i4)",
  dataset:  "var(--c-dataset)",
  baseline: "var(--c-baseline)",
};

function precisionColor(key) {
  return PRECISION[key] || "var(--muted-3)";
}

// 趋势图按平台分线型
function platformLineStyle(label) {
  if (/Android/i.test(label)) return { dash: "5 3", color: "var(--c-i4)" };
  if (/x86|A10|cuda|gpu/i.test(label)) return { dash: "2 3", color: "var(--c-fp32)" };
  return { dash: "", color: "var(--c-fp16)" };
}
