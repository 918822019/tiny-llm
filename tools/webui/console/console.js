"use strict";
// tinyqwen web_console 前端逻辑 —— 四 tab + 摘要条 + 趋势/曲线图
// 修复:inline onclick→addEventListener;定时器生命周期;activeTail 单例;
// openSeries XSS;auto-scroll gate;100KB 日志上限;CSS 变量;响应式 viewBox。

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// ---------- 全局状态 ----------
let store = { status: null, activeTab: "status" };
let activeTail = null;  // {jobId, intervalId, pre, offset, autoScroll}
let statusTimer = null;
let trendMetric = "topt", trendSource = "all", trendLogScale = true;

// ---------- Tab 切换 ----------
function switchTab(name) {
  // 停止旧 tab 的定时器
  if (store.activeTab === "status" && name !== "status") stopStatusTimer();
  if (store.activeTab === "jobs" && name !== "jobs") stopActiveTail();

  store.activeTab = name;
  $$("nav button").forEach(b => {
    const active = b.dataset.tab === name;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", active);
  });
  ["status", "run", "jobs", "history"].forEach(t =>
    $("#tab-" + t).style.display = (t === name) ? "" : "none");

  // 启动新 tab 的定时器
  if (name === "status") startStatusTimer();

  ({ status: renderStatus, run: renderRun, jobs: renderJobs,
    history: renderHistory })[name]();
}

$$("nav button").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

// ---------- 定时器管理 ----------
function startStatusTimer() {
  stopStatusTimer();
  statusTimer = setInterval(() => {
    if (document.hidden) return;
    renderStatus();
  }, 8000);
}
function stopStatusTimer() {
  if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
}
function stopActiveTail() {
  if (activeTail) {
    clearInterval(activeTail.intervalId);
    activeTail = null;
  }
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopStatusTimer();
    stopActiveTail();
  } else {
    if (store.activeTab === "status") startStatusTimer();
    // activeTail 不自动恢复(用户切回 jobs tab 会重新打开)
  }
});

// ---------- 摘要条 ----------
function renderSummaryBar() {
  const bar = $("#summary-bar");
  if (!bar) return;
  const s = store.status;
  if (!s) return;
  const items = [];
  // 设备
  const dv = s.device || {};
  items.push(`<div class="summary-item"><span class="label">设备</span><span class="value">${esc(dv.device_model || dv.serial || "—")}</span></div>`);
  // 温度
  if (dv.temp_mc != null) {
    const t = (dv.temp_mc / 1000).toFixed(1);
    const cls = dv.temp_mc / 1000 > 60 ? "warn" : "ok";
    items.push(`<div class="summary-item"><span class="label">温度</span><span class="value">${t}°C</span></div>`);
  }
  // 基线
  if (s.baseline) {
    items.push(`<div class="summary-item"><span class="label">基线</span><span class="value">${s.baseline.decode_median_ms} ms/tok</span></div>`);
  }
  // 当前任务
  const running = (s.jobs || []).find(j => j.status === "running" || j.status === "starting");
  if (running) {
    items.push(`<div class="summary-item"><span class="label">当前任务</span><span class="value">${esc(running.label || running.id)}</span><span class="badge running">${running.status}</span></div>`);
  }
  bar.innerHTML = items.join("");
}

// ---------- ① 设备状态 ----------
const CHECK_ICON = { ok: "✅", warn: "⚠️", fail: "❌" };

function buildStatusSkeleton() {
  const box = $("#tab-status");
  box.innerHTML =
    '<div class="card"><h2>预检看板（等价 scripts/doctor_android.sh）'
    + ' <button class="ghost" id="st-refresh">刷新</button>'
    + ' <span class="hint" id="st-updated"></span></h2><div class="checks" id="st-checks"></div></div>'
    + '<div class="card"><h2>设备</h2>'
    + '<div class="kv">序列号：<b id="st-serial">—</b>　型号：<b id="st-model">—</b>　SoC：<b id="st-soc">—</b>　Android <b id="st-android">—</b></div>'
    + '<div class="kv">温度：<b id="st-temp">—</b>　大核：<b id="st-bigcores">—</b>　大核最高频率：<b id="st-maxfreq">—</b></div>'
    + '<div class="kv">设备端已缓存模型：<span id="st-devmodels">—</span></div>'
    + '</div>'
    + '<div class="card"><h2>当前 Android 基线</h2><div id="st-baseline">—</div></div>';
  $("#st-refresh").addEventListener("click", renderStatus);
}

function updateCheckNode(node, c) {
  const cls = "check " + c.state;
  if (node.className !== cls) node.className = cls;
  const name = node.querySelector(".name");
  const label = (CHECK_ICON[c.state] || "❓") + " " + c.name;
  if (name.textContent !== label) name.textContent = label;
  const msg = node.querySelector(".msg");
  if (msg.textContent !== c.msg) msg.textContent = c.msg;
}

async function renderStatus() {
  const box = $("#tab-status");
  if (!$("#st-checks")) buildStatusSkeleton();
  let d;
  try { d = await api("/api/status"); }
  catch (e) { box.querySelector(".checks").innerHTML =
    '<div class="hint">加载失败：' + esc(e.message) + '</div>'; return; }
  store.status = d;
  renderSummaryBar();
  reconcile($("#st-checks"), d.checks, c => c.key, c => {
    const div = document.createElement("div");
    const nm = document.createElement("span"); nm.className = "name";
    const ms = document.createElement("span"); ms.className = "msg";
    div.appendChild(nm); div.appendChild(ms);
    updateCheckNode(div, c);
    return div;
  }, updateCheckNode);
  const dv = d.device || {};
  setText("st-serial", dv.serial || "—");
  setText("st-model", dv.device_model || "—");
  setText("st-soc", dv.soc || "—");
  setText("st-android", dv.android_version || "—");
  setText("st-temp", dv.temp_mc != null ? (dv.temp_mc / 1000).toFixed(1) + "°C" : "—");
  setText("st-bigcores", dv.big_cores && dv.big_cores.length ? "cpu" + dv.big_cores.join(",") : "—");
  setText("st-maxfreq", dv.big_core_max_khz && dv.big_core_max_khz !== "unknown" ? dv.big_core_max_khz + " kHz" : "—");
  setText("st-devmodels", d.device_models.length
    ? d.device_models.map(m => m.name + "（" + fmtSize(m.size) + "）").join("、") : "无");
  setText("st-updated", "更新于 " + new Date().toTimeString().slice(0, 8));
  // 基线(合并到 status 页脚)
  const bl = d.baseline;
  const blBox = $("#st-baseline");
  if (blBox) {
    if (!bl) blBox.innerHTML = '<div class="hint">尚未建立基线</div>';
    else blBox.innerHTML = `<div class="kv">label：<b>${esc(bl.label)}</b></div>
      <div class="kv">decode 中位：<b>${bl.decode_median_ms} ms/tok</b>　p95：<b>${bl.decode_p95_ms} ms/tok</b></div>
      <div class="kv">设备：<b>${esc(bl.device_model)} / ${esc(bl.soc)}</b>　commit：<b>${esc(bl.commit)}</b></div>`;
  }
}

// ---------- ② 发起测试 ----------
async function renderRun() {
  if (!store.status) { try { store.status = await api("/api/status"); } catch (e) {} }
  const models = (store.status && store.status.host_models || []).map(m => m.name);
  const box = $("#tab-run");
  box.innerHTML = `
  <div class="card"><h2>发起测试</h2>
  <form id="runform">
    <div class="row">
      <label>任务类型
        <select name="type">
          <option value="verify">verify（正确性门禁）</option>
          <option value="bench" selected>bench（测速）</option>
          <option value="record">record（门禁+测速+写日志）</option>
          <option value="baseline">baseline（重建基线）</option>
          <option value="dataset">dataset（数据集负载：TTFT 分桶 + TOPT 分布）</option>
          <option value="tokenize">tokenize（生成数据集输入 JSONL）</option>
        </select>
      </label>
      <label>模型
        <select name="model">${models.map(m => `<option ${m === "model_f16.tqwen" ? "selected" : ""}>${esc(m)}</option>`).join("")}</select>
      </label>
      <label>label
        <input name="label" placeholder="如 fp16-neon-android">
      </label>
      <label>runs
        <input name="runs" type="number" value="1" min="1" max="10" style="min-width:60px">
      </label>
    </div>
    <div class="row" style="align-items:center;gap:8px">
      <span style="font-size:13px;color:var(--muted)">常用配方（点击填入下方 extra-args）：</span>
      <button class="ghost" type="button" id="recipe-ref">ref 基线</button>
      <button class="ghost" type="button" id="recipe-fp16">fp16 满栈 NEON（推荐）</button>
      <button class="ghost" type="button" id="recipe-i4">INT4（W4A8 SDOT）</button>
    </div>
    <div class="row" id="dataset-row" style="display:none">
      <label>数据集输入（tokenize_batch.py 生成）
        <select name="jsonl">${(store.status && store.status.jsonls || []).length
          ? (store.status.jsonls.map(j => `<option value="${esc(j.name)}">${esc(j.name)}${j.num != null ? "（" + j.num + " 条" + (j.token_range ? "，" + j.token_range + " tok" : "") + "）" : ""}</option>`).join(""))
          : '<option value="">（无：先跑 tools/tokenize_batch.py）</option>'}</select>
      </label>
      <label>decode tok/条
        <input name="decode_tokens" type="number" value="32" min="1" max="128" style="min-width:70px">
      </label>
      <label>max-seq-len
        <input name="seq_len" type="number" value="128" min="8" max="1024" style="min-width:80px">
      </label>
    </div>
    <div class="row" id="tokenize-row" style="display:none">
      <label>条数
        <input name="tk_num" type="number" value="16" min="1" max="500" style="min-width:70px">
      </label>
      <label>token 下限
        <input name="tk_min" type="number" value="8" min="1" style="min-width:60px">
      </label>
      <label>token 上限
        <input name="tk_max" type="number" value="96" min="1" style="min-width:60px">
      </label>
      <label>输出名（benchmarks/&lt;名&gt;.jsonl）
        <input name="tk_name" placeholder="poetry16" value="poetry16" style="min-width:140px">
      </label>
      <span class="hint">数据源默认诗词数据集 test.csv；确定性取前 N，可复现</span>
    </div>
    <div class="row">
      <label>extra-args（传给设备端 binary，kernel 实现都在这里切换）
        <input name="extra_args" class="wide" placeholder="--matvec-impl neon_mt_kv_nt --ops-impl neon">
      </label>
    </div>
    <div class="row">
      <label>对照模型（A/B 用，可空）
        <select name="control_model"><option value="">（同模型）</option>${models.map(m => `<option>${esc(m)}</option>`).join("")}</select>
      </label>
      <label>对照参数（可空）
        <input name="control_args" placeholder="--matvec-impl ref">
      </label>
      <label>绑核
        <input name="pin_cores" value="big" style="min-width:90px">
      </label>
      <label>线程数（空=自动对齐绑核数）
        <input name="threads" type="number" min="0" style="min-width:70px">
      </label>
      <label style="flex-direction:row;align-items:center;gap:6px">
        <input name="no_thermal_gate" type="checkbox" style="min-width:auto"> 跳过热门禁
      </label>
    </div>
    <button class="primary" type="submit">启动任务</button>
    <div class="hint">同一时刻只允许一个设备任务（设备独占）。fp16 满栈配方：--matvec-impl neon_mt_kv_nt --ops-impl neon</div>
  </form></div>`;

  // addEventListener 替代 inline onclick
  $("#recipe-ref").addEventListener("click", () => applyRecipe("", ""));
  $("#recipe-fp16").addEventListener("click", () => applyRecipe("model_f16.tqwen", "--matvec-impl neon_mt_kv_nt --ops-impl neon"));
  $("#recipe-i4").addEventListener("click", () => applyRecipe("model_i4_hqq_lmh.tqwen", "--matvec-impl sdot_mt --ops-impl neon"));

  $("#runform").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = new FormData(ev.target);
    const spec = {
      type: f.get("type"), model: f.get("model"),
      label: (f.get("label") || "").trim(),
      runs: parseInt(f.get("runs") || "1", 10),
      extra_args: (f.get("extra_args") || "").trim(),
      control_model: f.get("control_model") || "",
      control_args: (f.get("control_args") || "").trim(),
      pin_cores: (f.get("pin_cores") || "").trim(),
      no_thermal_gate: !!f.get("no_thermal_gate"),
    };
    if (f.get("threads")) spec.threads = parseInt(f.get("threads"), 10);
    if (spec.type === "dataset") {
      spec.jsonl = (f.get("jsonl") || "").trim();
      spec.decode_tokens = parseInt(f.get("decode_tokens") || "32", 10);
      spec.seq_len = parseInt(f.get("seq_len") || "128", 10);
    }
    if (spec.type === "tokenize") {
      spec.num = parseInt(f.get("tk_num") || "16", 10);
      spec.min_tokens = parseInt(f.get("tk_min") || "8", 10);
      spec.max_tokens = parseInt(f.get("tk_max") || "96", 10);
      spec.out_name = (f.get("tk_name") || "").trim();
    }
    try {
      const r = await api("/api/jobs", {method: "POST",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify(spec)});
      switchTab("jobs");
      openJobLog(r.id);
    } catch (e) { toast(e.message); }
  });
  const typeSel = $("#runform select[name='type']");
  const toggleRows = () => {
    $("#dataset-row").style.display = typeSel.value === "dataset" ? "" : "none";
    $("#tokenize-row").style.display = typeSel.value === "tokenize" ? "" : "none";
  };
  typeSel.addEventListener("change", toggleRows);
  toggleRows();
}

function applyRecipe(model, args) {
  const f = $("#runform");
  if (!f) return;
  f.extra_args.value = args;
  if (model) {
    const hit = [...f.model.options].find(o => o.textContent === model);
    if (hit) f.model.value = hit.value;
    else toast("本机没有 " + model + "（先导出）");
  }
  toast(args ? "已填入 extra-args" : "已清空 extra-args（= ref 基线）");
}

// 基线页「重建」预填
async function prefillRun(vals) {
  switchTab("run");
  await renderRun();
  const f = $("#runform");
  for (const [k, v] of Object.entries(vals)) { if (f.elements[k]) f.elements[k].value = v; }
}

// ---------- ③ 任务与日志 ----------
function buildJobsSkeleton() {
  const box = $("#tab-jobs");
  box.innerHTML =
    '<div class="card"><h2>任务 <button class="ghost" id="jobs-refresh">刷新</button></h2>'
    + '<div id="jobs-body"><div class="hint">加载中…</div></div></div>'
    + '<div class="card" id="logpane" style="display:none">'
    + '<h2 id="logtitle"></h2>'
    + '<div class="log-controls"><button class="ghost" id="log-autoscroll">自动滚动：开</button></div>'
    + '<pre class="log" id="logpre"></pre></div>';
  $("#jobs-refresh").addEventListener("click", renderJobs);
  $("#log-autoscroll").addEventListener("click", (e) => {
    if (!activeTail) return;
    activeTail.autoScroll = !activeTail.autoScroll;
    e.target.textContent = "自动滚动：" + (activeTail.autoScroll ? "开" : "关");
    e.target.classList.toggle("active", activeTail.autoScroll);
  });
}

function buildJobRow(j) {
  const tr = document.createElement("tr");
  const id = document.createElement("td"); id.textContent = j.id;
  const type = document.createElement("td"); type.textContent = j.type;
  const label = document.createElement("td"); label.textContent = j.label || "—";
  const model = document.createElement("td"); model.textContent = j.model;
  const status = document.createElement("td"); const badge = document.createElement("span"); badge.className = "badge"; status.appendChild(badge);
  const exit = document.createElement("td"); exit.className = "c-exit";
  const started = document.createElement("td"); started.textContent = (j.started || "").slice(5, 16);
  const actions = document.createElement("td"); actions.className = "c-actions";
  const logBtn = document.createElement("button"); logBtn.className = "ghost"; logBtn.textContent = "日志";
  logBtn.addEventListener("click", () => openJobLog(j.id));
  actions.appendChild(logBtn);
  tr.append(id, type, label, model, status, exit, started, actions);
  updateJobRow(tr, j);
  return tr;
}

function updateJobRow(tr, j) {
  const badge = tr.querySelector(".badge");
  if (badge.textContent !== j.status) {
    badge.textContent = j.status;
    badge.className = "badge " + j.status;
  }
  const exit = tr.querySelector(".c-exit");
  const exitTxt = String(j.exit ?? "—");
  if (exit.textContent !== exitTxt) exit.textContent = exitTxt;
  const act = tr.querySelector(".c-actions");
  const killBtn = act.querySelector(".c-kill");
  if (j.status === "running" && !killBtn) {
    const b = document.createElement("button");
    b.className = "ghost c-kill"; b.textContent = "终止";
    b.addEventListener("click", () => killJob(j.id));
    act.appendChild(b);
  } else if (j.status !== "running" && killBtn) {
    killBtn.remove();
  }
}

async function renderJobs() {
  if (!$("#jobs-body")) buildJobsSkeleton();
  const body = $("#jobs-body");
  let d;
  try { d = await api("/api/jobs"); }
  catch (e) { body.innerHTML = '<div class="hint">加载失败：' + esc(e.message) + '</div>'; return; }
  renderSummaryBar();
  if (!d.jobs.length) {
    body.innerHTML = '<div class="hint">还没有任务，去「发起测试」跑一个</div>';
    return;
  }
  if (!body.querySelector("table")) {
    const tbl = document.createElement("table");
    tbl.innerHTML = "<tr><th>id</th><th>类型</th><th>label</th><th>模型</th><th>状态</th><th>exit</th><th>开始</th><th></th></tr>";
    const tb = document.createElement("tbody"); tb.id = "jobs-rows";
    tbl.appendChild(tb);
    body.replaceChildren(tbl);
  }
  reconcile($("#jobs-rows"), d.jobs, j => j.id, buildJobRow, updateJobRow);
}

async function killJob(id) {
  try { await api(`/api/jobs/${id}/kill`, {method: "POST"}); renderJobs(); }
  catch (e) { toast(e.message); }
}

function openJobLog(id) {
  if (!$("#logpane")) buildJobsSkeleton();
  renderJobs();
  // 单例 activeTail:关闭旧的再开新的
  stopActiveTail();
  const pane = $("#logpane");
  pane.style.display = "";
  $("#logtitle").textContent = "日志：" + id;
  const LOG_CAP = 100 * 1024;  // 100KB 上限
  let offset = 0;
  let pollInFlight = false;  // 防重叠
  let userScrolledUp = false;

  // 用 getElementById 每次重新取,防 DOM 重建后 stale 引用
  function getPre() { return document.getElementById("logpre"); }
  const pre = getPre();
  if (pre) {
    pre.textContent = "";
    // 跟踪用户是否手动往上翻页
    pre.addEventListener("scroll", () => {
      userScrolledUp = pre.scrollTop + pre.clientHeight < pre.scrollHeight - 10;
    });
  }

  const poll = async () => {
    if (pollInFlight) return;  // 上一次还没完就跳过
    pollInFlight = true;
    try {
      const d = await api(`/api/jobs/${id}/tail?from=${offset}`);
      const pre = getPre();
      if (!pre) return;  // DOM 被重建了,放弃这次
      if (d.data) {
        // 在添加新内容前记录滚动位置
        const wasAtBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 10;
        pre.textContent += d.data;
        if (pre.textContent.length > LOG_CAP) {
          pre.textContent = pre.textContent.slice(-LOG_CAP);
        }
        offset = d.size;
        // 自动滚动:用户在底部(或从未滚动)时才滚
        if (activeTail && activeTail.autoScroll && (!userScrolledUp || wasAtBottom)) {
          pre.scrollTop = pre.scrollHeight;
        }
      }
      if (d.status !== "running" && d.status !== "starting") {
        stopActiveTail();
        pre.textContent += `\n[结束：${d.status}，exit=${d.exit}]\n`;
        if (activeTail === null || !activeTail.autoScroll || !userScrolledUp) {
          pre.scrollTop = pre.scrollHeight;
        }
        renderJobs();
      }
    } catch (e) { /* 轮询瞬断忽略 */ }
    finally { pollInFlight = false; }
  };
  poll();
  activeTail = {
    jobId: id,
    intervalId: setInterval(poll, 1000),
    autoScroll: true,
  };
}

// ---------- ④ 历史与趋势 ----------
async function renderHistory() {
  const box = $("#tab-history");
  let d;
  try { d = await api("/api/history"); }
  catch (e) { box.innerHTML = '<div class="card">加载失败：' + esc(e.message) + '</div>'; return; }
  const rows = d.history.slice().reverse();

  // 趋势控制(addEventListener)
  const btn = (active, id, text) =>
    `<button class="${active ? "primary" : "ghost"}" id="${id}" style="padding:3px 10px">${text}</button>`;
  let html = '<div class="card"><h2>趋势（optimization_log.md 账本 + 测量历史）</h2>'
    + '<div class="trend-controls">'
    + '<span class="label">指标</span>'
    + btn(trendMetric === "topt", "tm-topt", "TOPT（decode ms/tok）")
    + btn(trendMetric === "ttft", "tm-ttft", "TTFT（首 token ms）")
    + '<span class="label" style="margin-left:12px">来源</span>'
    + btn(trendSource === "all", "ts-all", "全部")
    + btn(trendSource === "canonical", "ts-can", "canonical")
    + btn(trendSource === "dataset", "ts-ds", "dataset")
    + '<span style="margin-left:12px"></span>'
    + btn(false, "ts-scale", `y 轴：${trendLogScale ? "对数" : "线性"}`)
    + '</div>'
    + '<div class="legend"><span><i style="background:var(--c-fp16)"></i>fp16</span>'
    + '<span><i style="background:var(--c-i4)"></i>INT4</span>'
    + '<span><i style="background:var(--c-fp32)"></i>fp32</span>'
    + '<span><i style="background:var(--surface);border:2px solid var(--c-dataset)"></i>dataset 行（空心）</span>'
    + '<span><i style="background:var(--c-baseline);border-radius:0;height:2px;width:16px"></i>当前基线（仅 TOPT）</span></div>'
    + '<div id="trendchart"></div></div>';

  // 历史表
  html += '<div class="card"><h2>测量历史（benchmarks/history_android.jsonl）</h2>';
  if (!rows.length) html += '<div class="hint">暂无记录</div>';
  else {
    html += "<table><tr><th>时间</th><th>label</th><th>模型</th><th>TOPT median ms/tok</th><th>p95</th><th>TTFT ms</th><th>负载 prefill+decode</th><th>A/B×</th><th>设备</th><th>commit</th><th>来源</th><th>曲线</th></tr><tbody id='hist-rows'></tbody></table>";
  }
  html += "</div>";
  // seriespane 独立 mount 点(不被 renderHistory 重建销毁)
  html += '<div id="seriespane"></div>';
  box.innerHTML = html;

  // 绑定控制按钮
  $("#tm-topt").addEventListener("click", () => { trendMetric = "topt"; renderHistory(); });
  $("#tm-ttft").addEventListener("click", () => { trendMetric = "ttft"; renderHistory(); });
  $("#ts-all").addEventListener("click", () => { trendSource = "all"; renderHistory(); });
  $("#ts-can").addEventListener("click", () => { trendSource = "canonical"; renderHistory(); });
  $("#ts-ds").addEventListener("click", () => { trendSource = "dataset"; renderHistory(); });
  $("#ts-scale").addEventListener("click", () => { trendLogScale = !trendLogScale; renderHistory(); });

  // 历史表行(reconcile,不再 innerHTML 循环)
  if (rows.length) {
    const tbody = $("#hist-rows");
    const buildRow = (r) => {
      const tr = document.createElement("tr");
      const cells = [
        (r.ts || "").slice(5, 16),
        r.label || "",
        r.model || "—",
        r.median_ms ?? "—",
        r.p95_ms ?? "—",
        r.ttft_ms ?? "—",
        r.dataset_token_range ? r.dataset_token_range + " tok" : (r.prompt_tokens != null ? r.prompt_tokens + "+" + (r.generated_tokens ?? "?") : "—"),
        r.ab_ratio ? r.ab_ratio + "×" : "—",
        r.device || "—",
        r.commit || "—",
      ];
      for (const c of cells) {
        const td = document.createElement("td");
        td.textContent = c; tr.appendChild(td);
      }
      // source badge
      const srcTd = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = "badge " + (r.source || "");
      badge.textContent = r.source || "";
      srcTd.appendChild(badge); tr.appendChild(srcTd);
      // 曲线按钮(addEventListener,不再拼 onclick 字符串 — 修 XSS)
      const actTd = document.createElement("td");
      if (r.series_file || r.ttft_file) {
        const b = document.createElement("button");
        b.className = "ghost"; b.textContent = "曲线";
        b.addEventListener("click", () => openSeries(
          r.series_file ? r.series_file.split("/").pop() : "",
          r.ttft_file ? r.ttft_file.split("/").pop() : "",
          r.label, r.ttft_ms ?? null));
        actTd.appendChild(b);
      } else { actTd.textContent = "—"; }
      tr.appendChild(actTd);
      return tr;
    };
    reconcile(tbody, rows, r => (r.ts || "") + "|" + r.label, buildRow, () => {});
  }

  drawTrend(d);

  // 恢复已打开的 series pane
  if (store.openedSeries) {
    openSeries(store.openedSeries.seriesFile, store.openedSeries.ttftFile,
      store.openedSeries.label, store.openedSeries.ttftMs);
  }
}

function colorOf(p) {
  const s = ((p.model || "") + " " + (p.label || "")).toLowerCase();
  if (s.includes("i4") || s.includes("int4")) return "var(--c-i4)";
  if (s.includes("f16") || s.includes("fp16")) return "var(--c-fp16)";
  return "var(--c-fp32)";
}

function drawTrend(d) {
  const box = $("#trendchart");
  if (!box) return;
  box.replaceChildren();
  const norm = s => (s || "").replace(/（Android）$/, "").replace(/\(Android\)$/, "").trim();
  const val = p => trendMetric === "ttft" ? p.ttft_ms : p.median_ms;
  let pts = [];
  for (const r of d.log) pts.push({...r, _src: "log"});
  const seen = new Set(pts.map(p => norm(p.label) + "|" + p.median_ms));
  for (const r of d.history) {
    const key = norm(r.label) + "|" + r.median_ms;
    if (!seen.has(key)) { pts.push({...r, _src: "history"}); seen.add(key); }
  }
  if (trendSource === "canonical") pts = pts.filter(p => p.source !== "dataset");
  else if (trendSource === "dataset") pts = pts.filter(p => p.source === "dataset");
  pts = pts.filter(p => val(p) != null);
  if (!pts.length) { renderState(box, "empty", "暂无符合条件的数据"); return; }

  const W = responsiveWidth(box), H = 300, padL = 52, padR = 12, padT = 14, padB = 30;
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  const base = trendMetric === "topt" && store.status && store.status.baseline
    ? store.status.baseline.decode_median_ms : null;
  const ys = pts.map(val).concat(base ? [base] : []);
  let ymin = Math.min(...ys) * 0.85, ymax = Math.max(...ys) * 1.15;
  const lg = v => trendLogScale ? Math.log10(Math.max(v, 1e-9)) : v;
  const ymap = v => H - padB - (lg(v) - lg(ymin)) / (lg(ymax) - lg(ymin) || 1) * (H - padT - padB);
  const xmap = i => padL + (pts.length === 1 ? 0.5 : i / (pts.length - 1)) * (W - padL - padR);

  for (let t = 0; t <= 4; t++) {
    const v = trendLogScale
      ? Math.pow(10, lg(ymin) + t / 4 * (lg(ymax) - lg(ymin)))
      : ymin + t / 4 * (ymax - ymin);
    const y = ymap(v);
    svg.appendChild(el("line", {x1: padL, y1: y, x2: W - padR, y2: y, stroke: "var(--border-soft)"}));
    const tx = el("text", {x: 6, y: y + 4, "font-size": 11, fill: "var(--muted-2)"});
    tx.textContent = fmtMs(v);
    svg.appendChild(tx);
  }
  if (base) {
    const bl = el("line", {x1: padL, y1: ymap(base), x2: W - padR, y2: ymap(base),
      "stroke-dasharray": "5,4", "stroke-width": 1.5});
    bl.style.stroke = "var(--c-baseline)";
    svg.appendChild(bl);
  }
  // 趋势线
  for (let i = 1; i < pts.length; i++) {
    const ln = el("line", {x1: xmap(i - 1), y1: ymap(val(pts[i - 1])),
      x2: xmap(i), y2: ymap(val(pts[i])), "stroke-width": 1});
    ln.style.stroke = "var(--border)";
    svg.appendChild(ln);
  }
  const yl = el("text", {x: 6, y: 12, "font-size": 11, fill: "var(--muted)"});
  yl.textContent = trendMetric === "ttft" ? "TTFT ms" : "TOPT ms/tok";
  svg.appendChild(yl);
  pts.forEach((p, i) => {
    const isDs = p.source === "dataset";
    const c = el("circle", {cx: xmap(i), cy: ymap(val(p)), r: isDs ? 5 : 4.5,
      "stroke-width": isDs ? 2 : 0, opacity: p.rolled_back ? 0.3 : 1});
    if (isDs) { c.style.fill = "var(--surface)"; c.style.stroke = "var(--c-dataset)"; }
    else { c.style.fill = colorOf(p); }
    const tt = el("title", {});
    tt.textContent = `${p.label}\nmedian=${p.median_ms}ms  p95=${p.p95_ms ?? "?"}${p.ttft_ms != null ? "  TTFT=" + p.ttft_ms + "ms" : ""}${p.dataset_token_range ? "\n数据集负载 " + p.dataset_token_range + " tok" : (p.prompt_tokens != null ? "\nprefill=" + p.prompt_tokens + "tok  decode=" + (p.generated_tokens ?? "?") + "tok" : "")}\ncommit=${p.commit || "?"}  来源=${p.source || p._src}${p.note ? "\n" + p.note : ""}`;
    c.appendChild(tt);
    svg.appendChild(c);
    if (pts.length <= 14 || i % Math.ceil(pts.length / 14) === 0) {
      const tx = el("text", {x: xmap(i), y: H - 8, "font-size": 10, fill: "var(--muted-2)", "text-anchor": "middle"});
      tx.textContent = (p.label || "").slice(0, 14);
      svg.appendChild(tx);
    }
  });
  box.appendChild(svg);
}

// ---------- ④b 资源时序曲线 ----------
function pairUp(ts, vals) {
  const pts = [];
  for (let i = 0; i < ts.length && i < vals.length; i++)
    if (vals[i] != null) pts.push([ts[i], vals[i]]);
  return pts;
}

function lineChart(opts) {
  const W = responsiveWidth({clientWidth: 1000}), H = 180, padL = 56, padR = 12, padT = 8, padB = 24;
  const wrap = document.createElement("div");
  const title = document.createElement("div");
  title.className = "kv"; title.style.margin = "10px 0 2px";
  const b = document.createElement("b"); b.textContent = opts.title;
  title.appendChild(b); wrap.appendChild(title);
  const allPts = opts.lines.flatMap(l => l.pts);
  if (!allPts.length) { const h = document.createElement("div"); h.className = "hint"; h.textContent = "无数据"; wrap.appendChild(h); return wrap; }
  let ymin = opts.ymin ?? Math.min(...allPts.map(p => p[1]));
  let ymax = opts.ymax ?? Math.max(...allPts.map(p => p[1]));
  if (ymax === ymin) { ymax += 1; if (opts.ymin == null) ymin -= 1; }
  const xmax = Math.max(...allPts.map(p => p[0]), 0.001);
  const xmap = x => padL + x / xmax * (W - padL - padR);
  const ymap = v => H - padB - (v - ymin) / (ymax - ymin || 1) * (H - padT - padB);
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  for (let g = 0; g <= 3; g++) {
    const v = ymin + g / 3 * (ymax - ymin), y = ymap(v);
    svg.appendChild(el("line", {x1: padL, y1: y, x2: W - padR, y2: y, stroke: "var(--border-soft)"}));
    const tx = el("text", {x: 6, y: y + 4, "font-size": 10, fill: "var(--muted-2)"});
    tx.textContent = fmtMs(v);
    svg.appendChild(tx);
  }
  for (let g = 0; g <= 4; g++) {
    const x = g / 4 * xmax;
    const tx = el("text", {x: xmap(x), y: H - 8, "font-size": 10, fill: "var(--muted-2)", "text-anchor": "middle"});
    tx.textContent = x.toFixed(1) + "s";
    svg.appendChild(tx);
  }
  for (const l of opts.lines) {
    if (l.pts.length < 2) continue;
    let d = "";
    l.pts.forEach((p, i) => { d += (i ? "L" : "M") + xmap(p[0]).toFixed(1) + " " + ymap(p[1]).toFixed(1); });
    const path = el("path", {d, "stroke-width": l.width || 1.5});
    path.style.fill = "none"; path.style.stroke = l.color;
    svg.appendChild(path);
  }
  wrap.appendChild(svg);
  return wrap;
}

async function openSeries(seriesFile, ttftFile, label, ttftMs) {
  store.openedSeries = { seriesFile, ttftFile, label, ttftMs };
  const box = $("#tab-history");
  let pane = $("#seriespane");
  if (!pane) {
    pane = document.createElement("div");
    pane.id = "seriespane"; pane.className = "card";
    box.appendChild(pane);
  }
  pane.replaceChildren();
  const h2 = document.createElement("h2"); h2.textContent = "曲线：" + label;
  const closeBtn = document.createElement("button");
  closeBtn.className = "ghost"; closeBtn.textContent = "关闭";
  closeBtn.addEventListener("click", () => { pane.replaceChildren(); store.openedSeries = null; });
  h2.appendChild(closeBtn);
  pane.appendChild(h2);
  const cont = document.createElement("div"); cont.id = "seriescharts";
  pane.appendChild(cont);
  cont.innerHTML = "<div class='hint'>加载中…</div>";

  let s = null;
  if (seriesFile) {
    try { s = await api("/api/series?file=" + encodeURIComponent(seriesFile)); }
    catch (e) { /* 资源曲线缺失就只画有的部分 */ }
  }
  if (!s && !ttftFile) { cont.innerHTML = "<div class='hint'>没有可显示的曲线数据</div>"; return; }

  cont.replaceChildren();
  const bigCores = (store.status && store.status.device && store.status.device.big_cores) || [];
  const bigSet = new Set(bigCores.map(c => "cpu" + c));
  const hasBig = bigSet.size > 0;
  const hint = document.createElement("div"); hint.className = "hint";
  hint.textContent = (s ? s.samples + " 个采样点" : "")
    + (ttftMs != null ? "　TTFT=" + ttftMs + " ms" : "")
    + (s ? "　x 轴含启动加载阶段（RSS 走平处 = 加载完、forward 开始）" : "");
  cont.appendChild(hint);

  if (s) {
    const t = s.t_rel_s;
    cont.appendChild(lineChart({title: "常驻内存 RSS（MB）",
      lines: [{pts: pairUp(t, s.rss_mb), color: "var(--c-fp16)", width: 2}]}));
    cont.appendChild(lineChart({title: "CPU 温度（°C，cpu 类 zone 取 max）",
      lines: [{pts: pairUp(t, s.temp_c), color: "var(--fail)", width: 2}]}));
    cont.appendChild(lineChart({title: "各核频率（GHz，" + (hasBig ? "橙=大核" : "全部核") + "）",
      lines: Object.entries(s.freq_khz || {}).map(([cpu, vals]) => ({
        pts: pairUp(t, vals.map(v => v == null ? null : v / 1e6)),
        color: hasBig ? (bigSet.has(cpu) ? "var(--c-i4)" : "var(--muted-3)") : "var(--c-i4)",
        width: hasBig && bigSet.has(cpu) ? 2 : 1}))}));
    cont.appendChild(lineChart({title: "各核利用率（%，" + (hasBig ? "绿=大核" : "全部核") + "）",
      lines: Object.entries(s.util_pct || {}).map(([cpu, vals]) => ({
        pts: pairUp(t, vals), color: hasBig ? (bigSet.has(cpu) ? "var(--c-baseline)" : "var(--muted-3)") : "var(--c-baseline)",
        width: hasBig && bigSet.has(cpu) ? 2 : 1})), ymin: 0, ymax: 100}));
  }
  if (ttftFile) {
    try {
      const d = await api("/api/ttft?file=" + encodeURIComponent(ttftFile));
      cont.appendChild(drawTTFTScatter(d.points || []));
    } catch (e) {
      const err = document.createElement("div"); err.className = "hint";
      err.textContent = "TTFT 散点加载失败：" + e.message;
      cont.appendChild(err);
    }
  }
}

function drawTTFTScatter(pts) {
  const W = responsiveWidth({clientWidth: 1000}), H = 260, padL = 56, padR = 16, padT = 12, padB = 34;
  const wrap = document.createElement("div");
  const title = document.createElement("div");
  title.className = "kv"; title.style.margin = "10px 0 2px";
  const b = document.createElement("b"); b.textContent = "TTFT vs prompt 长度（数据集负载，prefill 扩展性）";
  title.appendChild(b); wrap.appendChild(title);
  if (!pts.length) { const h = document.createElement("div"); h.className = "hint"; h.textContent = "无数据"; wrap.appendChild(h); return wrap; }
  const xs = pts.map(p => p.prompt_tokens);
  const ys = pts.map(p => p.ttft_ms);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  if (xmin === xmax) { xmin -= 1; xmax += 1; }
  const ymax = Math.max(...ys) * 1.15;
  const xmap = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
  const ymap = v => H - padB - v / ymax * (H - padT - padB);
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  for (let g = 0; g <= 3; g++) {
    const v = ymax * g / 3, y = ymap(v);
    svg.appendChild(el("line", {x1: padL, y1: y, x2: W - padR, y2: y, stroke: "var(--border-soft)"}));
    const tx = el("text", {x: 6, y: y + 4, "font-size": 10, fill: "var(--muted-2)"});
    tx.textContent = fmtMs(v);
    svg.appendChild(tx);
  }
  for (let g = 0; g <= 4; g++) {
    const x = xmin + (xmax - xmin) * g / 4;
    const tx = el("text", {x: xmap(x), y: H - 10, "font-size": 10, fill: "var(--muted-2)", "text-anchor": "middle"});
    tx.textContent = Math.round(x) + " tok";
    svg.appendChild(tx);
  }
  const ks = pts.map(p => p.ttft_ms / p.prompt_tokens).sort((a, b) => a - b);
  const k = ks[Math.floor(ks.length / 2)];
  const sl = el("line", {x1: xmap(xmin), y1: ymap(k * xmin),
    x2: xmap(xmax), y2: Math.max(ymap(k * xmax), padT), "stroke-dasharray": "5,4"});
  sl.style.stroke = "var(--muted-3)";
  svg.appendChild(sl);
  const kl = el("text", {x: W - padR - 4, y: padT + 12, "font-size": 11, fill: "var(--muted-2)", "text-anchor": "end"});
  kl.textContent = "中位 " + k.toFixed(2) + " ms/tok";
  svg.appendChild(kl);
  for (const p of pts) {
    const c = el("circle", {cx: xmap(p.prompt_tokens), cy: ymap(p.ttft_ms), r: 4.5, opacity: 0.85});
    c.style.fill = "var(--c-dataset)";
    const tt = el("title", {});
    tt.textContent = `prompt=${p.prompt_tokens} tok\nTTFT=${p.ttft_ms} ms`
      + (p.topt_median_ms != null ? `\nTOPT 中位=${p.topt_median_ms} ms/tok` : "");
    c.appendChild(tt);
    svg.appendChild(c);
  }
  wrap.appendChild(svg);
  return wrap;
}

// ---------- 启动 ----------
renderStatus();
