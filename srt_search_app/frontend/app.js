const state = {
  currentJobId: null,
  pollingTimer: null,
  results: [],
  activeResultId: null,
};

const els = {
  rootPath: document.getElementById("root-path"),
  modelName: document.getElementById("model-name"),
  queryInput: document.getElementById("query-input"),
  limitInput: document.getElementById("limit-input"),
  chooseFolderBtn: document.getElementById("choose-folder-btn"),
  summaryBtn: document.getElementById("summary-btn"),
  reindexBtn: document.getElementById("reindex-btn"),
  searchBtn: document.getElementById("search-btn"),
  resultCount: document.getElementById("result-count"),
  results: document.getElementById("results"),
  detail: document.getElementById("detail"),
  summaryRoot: document.getElementById("summary-root"),
  summaryFiles: document.getElementById("summary-files"),
  summaryChunks: document.getElementById("summary-chunks"),
  summaryIndexed: document.getElementById("summary-indexed"),
  summaryTime: document.getElementById("summary-time"),
  jobState: document.getElementById("job-state"),
  jobMessage: document.getElementById("job-message"),
  jobProgressBar: document.getElementById("job-progress-bar"),
  jobProgressText: document.getElementById("job-progress-text"),
};

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "请求失败");
  }
  return data;
}

function updateSummary(summary) {
  els.summaryRoot.textContent = summary?.root_path || "未选择";
  els.summaryFiles.textContent = String(summary?.total_files ?? 0);
  els.summaryChunks.textContent = String(summary?.total_chunks ?? 0);
  els.summaryIndexed.textContent = String(summary?.indexed_files ?? 0);
  els.summaryTime.textContent = summary?.last_indexed_at || "-";
}

async function loadSummary() {
  const rootPath = els.rootPath.value.trim();
  if (!rootPath) {
    updateSummary(null);
    return;
  }
  const modelName = els.modelName.value.trim();
  const data = await request(`/api/summary?root_path=${encodeURIComponent(rootPath)}&model_name=${encodeURIComponent(modelName)}`);
  updateSummary(data.summary);
}

function renderResults() {
  els.resultCount.textContent = `${state.results.length} 条`;
  if (state.results.length === 0) {
    els.results.innerHTML = '<div class="empty-state">没有找到匹配结果。</div>';
    els.detail.className = "detail-card empty-state";
    els.detail.textContent = "可以尝试换一种描述方式，或者先更新索引。";
    return;
  }

  els.results.innerHTML = "";
  for (const result of state.results) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `result-item${state.activeResultId === result.chunk_id ? " active" : ""}`;
    item.innerHTML = `
      <div class="result-title">${escapeHtml(result.text)}</div>
      <div class="result-meta">
        <div>文件：${escapeHtml(result.rel_path)}</div>
        <div>时间：${result.start_time} - ${result.end_time}</div>
        <div>相关度：${result.score}%</div>
      </div>
    `;
    item.addEventListener("click", () => {
      state.activeResultId = result.chunk_id;
      renderResults();
      renderDetail(result);
    });
    els.results.appendChild(item);
  }

  const selected = state.results.find((item) => item.chunk_id === state.activeResultId) || state.results[0];
  state.activeResultId = selected.chunk_id;
  renderDetail(selected);
}

function renderDetail(result) {
  els.detail.className = "detail-card";
  els.detail.innerHTML = `
    <div class="block">
      <h3>${escapeHtml(result.rel_path)}</h3>
      <div class="label">完整路径</div>
      <div>${escapeHtml(result.abs_path)}</div>
    </div>
    <div class="block">
      <div class="label">时间范围</div>
      <div>${result.start_time} - ${result.end_time}</div>
    </div>
    <div class="block">
      <div class="label">命中片段</div>
      <div>${escapeHtml(result.text)}</div>
    </div>
    <div class="block">
      <div class="label">前文</div>
      <div>${escapeHtml(result.prev_text || "无")}</div>
    </div>
    <div class="block">
      <div class="label">后文</div>
      <div>${escapeHtml(result.next_text || "无")}</div>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function updateJob(job) {
  els.jobState.textContent = job.state;
  els.jobMessage.textContent = job.error || job.message || "等待中";
  const total = job.total_files || 0;
  const processed = job.processed_files || 0;
  const pct = total > 0 ? Math.min((processed / total) * 100, 100) : 0;
  els.jobProgressBar.style.width = `${pct}%`;
  els.jobProgressText.textContent = `${processed} / ${total}`;
}

async function pollJob(jobId) {
  const data = await request(`/api/index/jobs/${jobId}`);
  updateJob(data.job);
  if (data.job.state === "completed" || data.job.state === "failed") {
    clearInterval(state.pollingTimer);
    state.pollingTimer = null;
    state.currentJobId = null;
    await loadSummary();
  }
}

async function chooseFolder() {
  const data = await request("/api/folders/choose", { method: "POST" });
  if (data.path) {
    els.rootPath.value = data.path;
    await loadSummary();
  }
}

async function startIndex(forceRebuild = false) {
  const rootPath = els.rootPath.value.trim();
  const modelName = els.modelName.value.trim();
  if (!rootPath) {
    throw new Error("请先选择字幕目录");
  }

  const data = await request("/api/index/start", {
    method: "POST",
    body: JSON.stringify({ root_path: rootPath, model_name: modelName, force_rebuild: forceRebuild }),
  });
  state.currentJobId = data.job_id;
  if (state.pollingTimer) {
    clearInterval(state.pollingTimer);
  }
  state.pollingTimer = setInterval(() => {
    pollJob(state.currentJobId).catch((error) => {
      clearInterval(state.pollingTimer);
      state.pollingTimer = null;
      els.jobMessage.textContent = error.message;
    });
  }, 1000);
  await pollJob(state.currentJobId);
}

async function doSearch() {
  const rootPath = els.rootPath.value.trim();
  const modelName = els.modelName.value.trim();
  const query = els.queryInput.value.trim();
  const limit = Number(els.limitInput.value) || 10;

  if (!rootPath) {
    throw new Error("请先选择字幕目录");
  }
  if (!query) {
    throw new Error("请输入搜索内容");
  }

  const data = await request("/api/search", {
    method: "POST",
    body: JSON.stringify({ root_path: rootPath, model_name: modelName, query, limit }),
  });
  state.results = data.results || [];
  state.activeResultId = state.results[0]?.chunk_id || null;
  renderResults();
}

function bindActions() {
  els.chooseFolderBtn.addEventListener("click", () => runSafe(chooseFolder));
  els.summaryBtn.addEventListener("click", () => runSafe(loadSummary));
  els.reindexBtn.addEventListener("click", () => runSafe(() => startIndex(false)));
  els.searchBtn.addEventListener("click", () => runSafe(doSearch));
  els.queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      runSafe(doSearch);
    }
  });
}

async function init() {
  const data = await request("/api/models");
  if (!els.modelName.value) {
    els.modelName.value = data.default_model_name;
  }
}

async function runSafe(fn) {
  try {
    await fn();
  } catch (error) {
    els.jobState.textContent = "error";
    els.jobMessage.textContent = error.message;
  }
}

bindActions();
init().catch((error) => {
  els.jobState.textContent = "error";
  els.jobMessage.textContent = error.message;
});
