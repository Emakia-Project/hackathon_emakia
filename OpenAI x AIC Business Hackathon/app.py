"""
analyze.py — EmakiaEval Multi-LLM Agreement Dashboard
======================================================
7 perspectives on hate speech classification:
  Human · LLM0 · LLM3 · LLM4 · OpenAI · Gemini · Claude

No assumed ground truth — we measure agreement between all evaluators.

Run: python3 analyze.py → http://localhost:5001
"""

from flask import Flask, render_template_string, jsonify, request
from google.cloud import bigquery
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
import requests as req

load_dotenv()
app = Flask(__name__)

EVALUATORS = ["OpenAI", "Gemini", "Claude"]


# ─────────────────────────────────────────────────────────────────────────────
# BigQuery
# ─────────────────────────────────────────────────────────────────────────────

def get_bq_client():
    try:
        return bigquery.Client(project="emakia")
    except Exception as e:
        print(f"❌ BigQuery: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM classifiers — each returns "Harassment" | "Neutral" | "Unknown"
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """Classify this social media post as HARASSMENT or NEUTRAL.
Reply with exactly one word: Harassment or Neutral.

Post: "{text}" """


def classify_openai(text, key):
    try:
        r = req.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 5,
                "temperature": 0,
                "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:300])}]
            },
            timeout=20
        )
        word = r.json()["choices"][0]["message"]["content"].strip().split()[0]
        return "Harassment" if "harass" in word.lower() else "Neutral"
    except Exception as e:
        print(f"OpenAI classify error: {e}")
        return "Unknown"


def classify_gemini(text, key):
    try:
        r = req.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": CLASSIFY_PROMPT.format(text=text[:300])}]}],
                "generationConfig": {"maxOutputTokens": 5, "temperature": 0}
            },
            timeout=20
        )
        word = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().split()[0]
        return "Harassment" if "harass" in word.lower() else "Neutral"
    except Exception as e:
        print(f"Gemini classify error: {e}")
        return "Unknown"


def classify_claude(text, key):
    try:
        r = req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 10,
                "temperature": 0,
                "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:300])}]
            },
            timeout=20
        )
        if r.status_code != 200:
            print(f"Claude HTTP {r.status_code}: {r.text[:100]}")
            return "Unknown"
        resp = r.json()
        print(f"🔍 Claude raw response: {str(resp)[:200]}")
        # Anthropic response: {"content": [{"type": "text", "text": "..."}], ...}
        items = resp.get("content", [])
        if not items:
            print(f"Claude empty content: {str(resp)[:100]}")
            return "Unknown"
        text_out = items[0].get("text", "").strip().lower()
        return "Harassment" if "harass" in text_out else "Neutral"
    except Exception as e:
        print(f"Claude classify error: {e}")
        return "Unknown"


def classify_all_llms(text, openai_key, gemini_key, claude_key):
    """Call all 3 LLMs concurrently for one post."""
    results = {}
    tasks = {
        "OpenAI": (classify_openai, openai_key),
        "Gemini": (classify_gemini, gemini_key),
        "Claude": (classify_claude, claude_key),
    }
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fn, text, key): name for name, (fn, key) in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis LLM call (OpenAI writes the narrative)
# ─────────────────────────────────────────────────────────────────────────────

def get_analysis(prompt, openai_key):
    r = req.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "max_tokens": 450,
            "messages": [
                {"role": "system", "content": "You are an expert AI safety researcher specializing in hate speech detection. No evaluator is assumed to be ground truth — all are perspectives."},
                {"role": "user",   "content": prompt}
            ]
        },
        timeout=30
    )
    if r.status_code != 200:
        raise Exception(f"OpenAI analysis error {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Agreement helpers
# ─────────────────────────────────────────────────────────────────────────────

def ag(a, b):
    """Case-insensitive agreement."""
    return (a or "").lower().strip() == (b or "").lower().strip()


def agreement_pct(rows, key_a, key_b):
    count = sum(1 for r in rows if ag(r.get(key_a), r.get(key_b)))
    return round(count / len(rows) * 100, 1) if rows else 0


def all_agree_pct(rows, keys):
    def all_same(r):
        vals = [(r.get(k) or "").lower().strip() for k in keys]
        return len(set(v for v in vals if v and v != "unknown")) == 1
    count = sum(1 for r in rows if all_same(r))
    return round(count / len(rows) * 100, 1) if rows else 0


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EmakiaEval — Multi-LLM Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0a0c10; --surface:#111318; --border:#1e2230;
    --accent:#4f8cff; --green:#3ecf8e; --orange:#f59e42;
    --red:#f26d6d; --purple:#a78bfa; --cyan:#22d3ee;
    --yellow:#fbbf24; --pink:#f472b6;
    --muted:#4a5068; --text:#e8eaf2; --subtext:#8891b0;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Syne',sans-serif; min-height:100vh; }
  body::before {
    content:''; position:fixed; inset:0;
    background-image:linear-gradient(var(--border) 1px,transparent 1px),
                     linear-gradient(90deg,var(--border) 1px,transparent 1px);
    background-size:40px 40px; opacity:0.3; pointer-events:none; z-index:0;
  }
  .container { max-width:1200px; margin:0 auto; padding:48px 24px; position:relative; z-index:1; }

  .logo { font-family:'DM Mono',monospace; font-size:11px; color:var(--accent); letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }
  h1 { font-size:clamp(26px,4.5vw,50px); font-weight:800; letter-spacing:-1px; line-height:1.05; }
  h1 span { color:var(--accent); }
  .subtitle { font-family:'DM Mono',monospace; font-size:11px; color:var(--subtext); margin-top:6px; }
  header { display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:36px; flex-wrap:wrap; gap:16px; }

  /* Evaluator legend */
  .legend { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:28px; }
  .legend-item { font-family:'DM Mono',monospace; font-size:11px; padding:5px 12px; border-radius:20px; border:1px solid; }
  .leg-human   { color:#e2e8f0; border-color:#4a5068; background:rgba(74,80,104,0.15); }
  .leg-llm0    { color:var(--red);    border-color:rgba(242,109,109,0.4); background:rgba(242,109,109,0.08); }
  .leg-llm3    { color:var(--orange); border-color:rgba(245,158,66,0.4);  background:rgba(245,158,66,0.08); }
  .leg-llm4    { color:var(--green);  border-color:rgba(62,207,142,0.4);  background:rgba(62,207,142,0.08); }
  .leg-openai  { color:var(--accent); border-color:rgba(79,140,255,0.4);  background:rgba(79,140,255,0.08); }
  .leg-gemini  { color:var(--cyan);   border-color:rgba(34,211,238,0.4);  background:rgba(34,211,238,0.08); }
  .leg-claude  { color:var(--purple); border-color:rgba(167,139,250,0.4); background:rgba(167,139,250,0.08); }

  /* Controls */
  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  select, button { font-family:'DM Mono',monospace; font-size:12px; border-radius:8px; border:1px solid var(--border); background:var(--surface); color:var(--text); padding:10px 14px; cursor:pointer; transition:all 0.2s; }
  select:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); padding:10px 24px; font-weight:700; }
  button.primary:hover { background:#3a7aef; }
  button.primary:disabled { opacity:0.45; cursor:not-allowed; }

  /* Info box */
  .info-box { font-family:'DM Mono',monospace; font-size:11px; color:var(--subtext); padding:12px 16px; background:rgba(79,140,255,0.05); border:1px solid rgba(79,140,255,0.15); border-radius:8px; margin-bottom:28px; line-height:1.7; }
  .info-box strong { color:var(--accent); }

  /* Agreement grid */
  .stats-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:12px; margin-bottom:24px; }
  .stat-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px; position:relative; overflow:hidden; animation:fadeUp 0.4s ease both; }
  .stat-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:10px 10px 0 0; }
  .c0::before{background:var(--red)}    .c0 .sv{color:var(--red)}    .c0 .bf{background:var(--red)}
  .c1::before{background:var(--orange)} .c1 .sv{color:var(--orange)} .c1 .bf{background:var(--orange)}
  .c2::before{background:var(--green)}  .c2 .sv{color:var(--green)}  .c2 .bf{background:var(--green)}
  .c3::before{background:var(--purple)} .c3 .sv{color:var(--purple)} .c3 .bf{background:var(--purple)}
  .c4::before{background:var(--cyan)}   .c4 .sv{color:var(--cyan)}   .c4 .bf{background:var(--cyan)}
  .c5::before{background:var(--yellow)} .c5 .sv{color:var(--yellow)} .c5 .bf{background:var(--yellow)}
  .c6::before{background:var(--pink)}   .c6 .sv{color:var(--pink)}   .c6 .bf{background:var(--pink)}
  .c7::before{background:var(--accent)} .c7 .sv{color:var(--accent)} .c7 .bf{background:var(--accent)}
  .sl { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase; color:var(--subtext); margin-bottom:8px; }
  .sv { font-size:32px; font-weight:800; letter-spacing:-1px; line-height:1; }
  .ss { font-family:'DM Mono',monospace; font-size:10px; color:var(--muted); margin-top:4px; }
  .bw { margin-top:8px; height:3px; background:var(--border); border-radius:4px; overflow:hidden; }
  .bf { height:100%; border-radius:4px; transition:width 1.2s ease; }

  /* Section header */
  .section-title { font-size:13px; font-weight:700; color:var(--subtext); letter-spacing:2px; text-transform:uppercase; font-family:'DM Mono',monospace; margin:28px 0 14px; }

  /* Analysis panel */
  .panel { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:26px; margin-bottom:22px; animation:fadeUp 0.5s ease both; }
  .panel-header { display:flex; align-items:center; gap:10px; margin-bottom:18px; flex-wrap:wrap; }
  .panel-title { font-size:15px; font-weight:700; }
  .badge { font-family:'DM Mono',monospace; font-size:10px; padding:3px 9px; border-radius:20px; background:rgba(79,140,255,0.1); color:var(--accent); border:1px solid rgba(79,140,255,0.25); }
  .analysis-text { font-family:'DM Mono',monospace; font-size:12px; line-height:1.9; color:var(--text); white-space:pre-wrap; }

  /* Sample table */
  .table-wrap { background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:24px; animation:fadeUp 0.6s ease both; }
  .table-header { padding:16px 20px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
  table { width:100%; border-collapse:collapse; font-family:'DM Mono',monospace; font-size:11px; }
  th { padding:9px 12px; text-align:left; color:var(--subtext); font-size:10px; letter-spacing:1px; text-transform:uppercase; border-bottom:1px solid var(--border); background:rgba(255,255,255,0.02); white-space:nowrap; }
  td { padding:9px 12px; border-bottom:1px solid rgba(30,34,48,0.5); vertical-align:middle; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:rgba(79,140,255,0.03); }
  .text-cell { max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--subtext); }
  .pill { display:inline-block; padding:2px 7px; border-radius:20px; font-size:10px; white-space:nowrap; }
  .harassment,.Harassment { background:rgba(242,109,109,0.15); color:var(--red); border:1px solid rgba(242,109,109,0.25); }
  .neutral,.Neutral       { background:rgba(62,207,142,0.15); color:var(--green); border:1px solid rgba(62,207,142,0.25); }
  .unknown,.Unknown       { background:rgba(74,80,104,0.2); color:var(--muted); border:1px solid var(--border); }
  .disagree-row td { background:rgba(242,109,109,0.025); }

  /* Progress */
  .progress-bar { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:16px 20px; margin-bottom:20px; font-family:'DM Mono',monospace; font-size:12px; color:var(--subtext); display:none; }
  .progress-fill { height:3px; background:var(--border); border-radius:4px; margin-top:10px; overflow:hidden; }
  .progress-fill-inner { height:100%; background:var(--accent); border-radius:4px; transition:width 0.5s ease; }

  .spinner { display:inline-block; width:13px; height:13px; border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; vertical-align:middle; margin-right:6px; }
  .hidden { display:none; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin { to{transform:rotate(360deg)} }
</style>
</head>
<body>
<div class="container">

  <header>
    <div>
      <div class="logo">EmakiaTech · AI Safety Research</div>
      <h1>Hate Speech<br><span>Multi-LLM</span> Analysis</h1>
      <p class="subtitle">7 perspectives · No assumed ground truth</p>
    </div>
    <div class="controls">
      <select id="limitSelect">
        <option value="20">20 posts</option>
        <option value="30">30 posts</option>
        <option value="50" selected>50 posts</option>
      </select>
      <select id="focusSelect">
        <option value="disagreements">Where CoreML disagrees</option>
        <option value="all">All posts</option>
      </select>
      <button class="primary" id="analyzeBtn" onclick="runAnalysis()">▶ Analyze</button>
    </div>
  </header>

  <!-- Evaluator legend -->
  <div class="legend">
    <span class="legend-item leg-human">👤 Human Label</span>
    <span class="legend-item leg-llm0">🔴 LLM0 CoreML</span>
    <span class="legend-item leg-llm3">🟠 LLM3 CoreML</span>
    <span class="legend-item leg-llm4">🟢 LLM4 CoreML</span>
    <span class="legend-item leg-openai">🔵 OpenAI</span>
    <span class="legend-item leg-gemini">🩵 Gemini</span>
    <span class="legend-item leg-claude">🟣 Claude</span>
  </div>

  <!-- Dataset tabs -->
  <div style="display:flex;gap:8px;margin-bottom:16px">
    <div id="tab-kaggle" onclick="switchDataset(event,'kaggle')" style="font-family:'DM Mono',monospace;font-size:12px;padding:8px 18px;border-radius:8px;border:1px solid var(--accent);background:rgba(79,140,255,0.08);color:var(--accent);cursor:pointer">📊 Kaggle Hate Speech</div>
    <div id="tab-coreml" onclick="switchDataset(event,'coreml')" style="font-family:'DM Mono',monospace;font-size:12px;padding:8px 18px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--subtext);cursor:pointer">🏛 Politics 2024</div>
  </div>

  <div class="info-box" id="datasetDesc">
    <strong>Kaggle Hate Speech dataset</strong> — Posts have a human crowd-sourced label + CoreML predictions (LLM0, LLM3, LLM4). The human label is <strong>not assumed to be truth</strong> — it is one of 7 perspectives. LLMs (OpenAI, Gemini, Claude) evaluate each post live.
  </div>

  <!-- Progress -->
  <div class="progress-bar" id="progressBar">
    <span id="progressText">Calling LLMs...</span>
    <div class="progress-fill"><div class="progress-fill-inner" id="progressInner" style="width:0%"></div></div>
  </div>

  <!-- Agreement with Human -->
  <div id="statsSection" class="hidden">
    <div class="section-title">Agreement with Human Label</div>
    <div class="stats-grid" id="humanAgreementGrid"></div>

    <div class="section-title">CoreML Internal Agreement</div>
    <div class="stats-grid" id="coremlAgreementGrid"></div>

    <div class="section-title">LLM Agreement with each other</div>
    <div class="stats-grid" id="llmAgreementGrid"></div>
  </div>

  <!-- OpenAI Analysis -->
  <div class="panel hidden" id="analysisPanel">
    <div class="panel-header">
      <div class="panel-title">GPT-4o Research Analysis</div>
      <span class="badge" id="analysisBadge">—</span>
    </div>
    <div class="analysis-text" id="analysisText"></div>
  </div>

  <!-- Sample table -->
  <div class="table-wrap hidden" id="tableWrap">
    <div class="table-header">
      <div class="panel-title">All 7 Perspectives — Sample Posts</div>
      <span class="badge" id="tableBadge">—</span>
    </div>
    <div id="tableContent"></div>
  </div>

</div>

<script>
function pill(val) {
  if (!val || val === '—') return '<span class="pill unknown">—</span>';
  const cls = val.toLowerCase().includes('harass') ? 'harassment' : val.toLowerCase().includes('neutral') ? 'neutral' : 'unknown';
  return `<span class="pill ${cls}">${val}</span>`;
}

function makeCard(label, value, sub, colorIdx, delay) {
  return `<div class="stat-card c${colorIdx}" style="animation-delay:${delay}s">
    <div class="sl">${label}</div>
    <div class="sv">${value}%</div>
    <div class="ss">${sub}</div>
    <div class="bw"><div class="bf" id="bar_${label.replace(/[^a-z0-9]/gi,'_')}" style="width:0%"></div></div>
  </div>`;
}

function animateBars(stats) {
  setTimeout(() => {
    stats.forEach(s => {
      const el = document.getElementById('bar_' + s.label.replace(/[^a-z0-9]/gi,'_'));
      if (el) el.style.width = s.value + '%';
    });
  }, 150);
}

let currentDataset = 'kaggle';

function switchDataset(e, ds) {
  currentDataset = ds;
  const tabs = ['kaggle', 'coreml'];
  tabs.forEach(t => {
    const el = document.getElementById('tab-'+t);
    if (t === ds) {
      el.style.borderColor = 'var(--accent)'; el.style.color = 'var(--accent)'; el.style.background = 'rgba(79,140,255,0.08)';
    } else {
      el.style.borderColor = 'var(--border)'; el.style.color = 'var(--subtext)'; el.style.background = 'var(--surface)';
    }
  });
  const descs = {
    kaggle: '<strong>Kaggle Hate Speech dataset</strong> — Posts have a human crowd-sourced label + CoreML predictions (LLM0, LLM3, LLM4). The human label is <strong>not assumed to be truth</strong> — it is one of 7 perspectives. LLMs (OpenAI, Gemini, Claude) evaluate each post live.',
    coreml: '<strong>Politics 2024 tweets</strong> — No human label. CoreML predictions (LLM0, LLM3, LLM4) already stored. <strong>No ground truth exists</strong> — LLMs (OpenAI, Gemini, Claude) evaluate each post live to measure AI-to-AI agreement on political content.'
  };
  document.getElementById('datasetDesc').innerHTML = descs[ds];
  document.getElementById('statsSection').classList.add('hidden');
  document.getElementById('analysisPanel').classList.add('hidden');
  document.getElementById('tableWrap').classList.add('hidden');
}

async function runAnalysis() {
  const btn   = document.getElementById('analyzeBtn');
  const limit = document.getElementById('limitSelect').value;
  const focus = document.getElementById('focusSelect').value;

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Analyzing...';

  document.getElementById('statsSection').classList.add('hidden');
  document.getElementById('analysisPanel').classList.add('hidden');
  document.getElementById('tableWrap').classList.add('hidden');

  const prog = document.getElementById('progressBar');
  prog.style.display = 'block';
  document.getElementById('progressText').textContent = `Fetching ${limit} posts from BigQuery...`;
  document.getElementById('progressInner').style.width = '10%';

  // Simulate progress
  const steps = [
    [20, 'Calling OpenAI GPT-4o-mini...'],
    [45, 'Calling Google Gemini...'],
    [70, 'Calling Anthropic Claude...'],
    [88, 'Computing agreement stats...'],
    [95, 'Generating analysis...'],
  ];
  let si = 0;
  const interval = setInterval(() => {
    if (si < steps.length) {
      document.getElementById('progressInner').style.width = steps[si][0] + '%';
      document.getElementById('progressText').textContent = steps[si][1];
      si++;
    }
  }, 2500);

  try {
    const res  = await fetch(`/api/analyze?limit=${limit}&focus=${focus}&dataset=${currentDataset}`);
    const data = await res.json();
    clearInterval(interval);
    prog.style.display = 'none';

    if (data.error) { alert('Error: ' + data.error); return; }

    // Human agreement section
    const hg = document.getElementById('humanAgreementGrid');
    hg.innerHTML = data.human_agreement.map((s,i) => makeCard(s.label, s.value, s.sub, i, i*0.05)).join('');
    animateBars(data.human_agreement);

    // CoreML section
    const cg = document.getElementById('coremlAgreementGrid');
    cg.innerHTML = data.coreml_agreement.map((s,i) => makeCard(s.label, s.value, s.sub, i, i*0.05)).join('');
    animateBars(data.coreml_agreement);

    // LLM section
    const lg = document.getElementById('llmAgreementGrid');
    lg.innerHTML = data.llm_agreement.map((s,i) => makeCard(s.label, s.value, s.sub, i, i*0.05)).join('');
    animateBars(data.llm_agreement);

    document.getElementById('statsSection').classList.remove('hidden');

    // Analysis
    document.getElementById('analysisBadge').textContent = `${data.total} posts · ${focus}`;
    document.getElementById('analysisText').textContent = data.analysis;
    document.getElementById('analysisPanel').classList.remove('hidden');

    // Table
    document.getElementById('tableBadge').textContent = data.samples.length + ' posts shown';
    const isCoreml = data.dataset === 'coreml';
    const refLabel = isCoreml ? '⚖️ Ensemble' : '👤 Human';
    const refKey   = isCoreml ? 'ensemble' : 'human';
    let html = '<table><thead><tr><th>Post</th>';
    html += '<th>' + refLabel + '</th>';
    html += '<th>🔴 LLM0</th><th>🟠 LLM3</th><th>🟢 LLM4</th><th>🔵 OpenAI</th><th>🩵 Gemini</th><th>🟣 Claude</th>';
    if (isCoreml) html += '<th>🟡 Grok</th><th>🟤 LLaMA</th><th>⚫ DeepSeek</th>';
    html += '<th>All?</th></tr></thead><tbody>';
    data.samples.forEach(r => {
      const cls = r.all_agree ? '' : 'disagree-row';
      html += '<tr class="' + cls + '">';
      html += '<td class="text-cell" title="' + r.text + '">' + r.text.substring(0,50) + '…</td>';
      html += '<td>' + pill(r[refKey]) + '</td>';
      html += '<td>' + pill(r.llm0) + '</td>';
      html += '<td>' + pill(r.llm3) + '</td>';
      html += '<td>' + pill(r.llm4) + '</td>';
      html += '<td>' + pill(r.openai) + '</td>';
      html += '<td>' + pill(r.gemini) + '</td>';
      html += '<td>' + pill(r.claude) + '</td>';
      if (isCoreml) {
        html += '<td>' + pill(r.grok) + '</td>';
        html += '<td>' + pill(r.llama) + '</td>';
        html += '<td>' + pill(r.deepseek) + '</td>';
      }
      html += '<td style="color:' + (r.all_agree ? 'var(--green)' : 'var(--red)') + '">' + (r.all_agree ? '✓' : '✗') + '</td>';
      html += '</tr>';
    });
    html += '</tbody></table>';
    document.getElementById('tableContent').innerHTML = html;
    document.getElementById('tableWrap').classList.remove('hidden');

  } catch(e) {
    clearInterval(interval);
    prog.style.display = 'none';
    alert('Request failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ Analyze';
  }
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTE
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze")
def analyze():
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    claude_key = os.environ.get("ANTHROPIC_API_KEY")

    missing = [k for k, v in {"OPENAI_API_KEY": openai_key, "GEMINI/GOOGLE_API_KEY": gemini_key, "ANTHROPIC_API_KEY": claude_key}.items() if not v]
    if missing:
        return jsonify({"error": f"Missing keys: {', '.join(missing)}"}), 500

    focus = request.args.get("focus", "disagreements")
    limit = min(request.args.get("limit", default=20, type=int), 50)

    client = get_bq_client()
    if not client:
        return jsonify({"error": "BigQuery connection failed"}), 500

    dataset = request.args.get("dataset", "kaggle")

    if dataset == "coreml":
        # Read ALL stored predictions — no live API calls needed
        # vote_score already includes possibly_sensitive (2x) + Gemini Vision (2x)
        # Use vote_score >= 5 as ensemble ground truth (matches paper threshold theta=5)
        if focus == "disagreements":
            where = """WHERE prediction_openai IS NOT NULL
                          AND (LOWER(prediction_llm0) != LOWER(prediction_openai)
                            OR LOWER(prediction_llm3) != LOWER(prediction_openai)
                            OR LOWER(prediction_llm4) != LOWER(prediction_openai))"""
        else:
            where = "WHERE prediction_openai IS NOT NULL"
        query = f"""
            SELECT tweet_id as post_id, text,
                   prediction_llm0, score_llm0,
                   prediction_llm3, score_llm3,
                   prediction_llm4, score_llm4,
                   prediction_openai,
                   prediction_gemini,
                   prediction_claude,
                   prediction_grok,
                   prediction_llama,
                   prediction_deepseek,
                   vote_score,
                   CASE WHEN vote_score >= 5 THEN 'Harassment' ELSE 'Neutral' END as ensemble_label
            FROM `emakia.politics2024.CoreMLpredictionEvaluation`
            {where}
            ORDER BY created_at DESC
            LIMIT {limit}
        """
    else:
        where = "WHERE models_agree = FALSE" if focus == "disagreements" else ""
        query = f"""
            SELECT post_id, text, human_label,
                   prediction_llm0, score_llm0,
                   prediction_llm3, score_llm3,
                   prediction_llm4, score_llm4,
                   models_agree
            FROM `emakia.kaggle_eval.predictions`
            {where}
            ORDER BY evaluated_at DESC
            LIMIT {limit}
        """

    bq_rows = list(client.query(query).result())
    total   = len(bq_rows)

    if total == 0:
        return jsonify({"error": "No data found", "focus": focus}), 404

    if dataset == "coreml":
        # Politics 2024 — use ALL stored predictions, no live API calls
        # vote_score already includes possibly_sensitive (2×) + Gemini Vision (2×)
        print(f"📊 Reading {total} stored predictions from CoreMLpredictionEvaluation...")
        enriched = []
        for r in bq_rows:
            enriched.append({
                "text":         r.text,
                "human":        getattr(r, "ensemble_label", "Unknown"),  # ensemble as reference
                "llm0":         r.prediction_llm0 or "Unknown",
                "llm3":         r.prediction_llm3 or "Unknown",
                "llm4":         r.prediction_llm4 or "Unknown",
                "openai":       r.prediction_openai or "Unknown",
                "gemini":       r.prediction_gemini or "Unknown",
                "claude":       r.prediction_claude or "Unknown",
                "grok":         r.prediction_grok or "Unknown",
                "llama":        r.prediction_llama or "Unknown",
                "deepseek":     r.prediction_deepseek or "Unknown",
                "vote_score":   int(r.vote_score or 0),
                "ensemble":     getattr(r, "ensemble_label", "Unknown"),
            })
    else:
        # Kaggle — call live APIs since no stored LLM predictions exist
        print(f"📊 Evaluating {total} posts with 3 LLMs concurrently...")
        enriched = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(classify_all_llms, r.text, openai_key, gemini_key, claude_key): r
                for r in bq_rows
            }
            for future in as_completed(futures):
                r      = futures[future]
                llms   = future.result()
                enriched.append({
                    "text":    r.text,
                    "human":   r.human_label,
                    "llm0":    r.prediction_llm0,
                    "llm3":    r.prediction_llm3,
                    "llm4":    r.prediction_llm4,
                    "openai":  llms.get("OpenAI", "Unknown"),
                    "gemini":  llms.get("Gemini", "Unknown"),
                    "claude":  llms.get("Claude", "Unknown"),
                })

    n = len(enriched)
    print(f"✅ All LLM calls complete — {n} posts enriched")

    # ── Agreement stats ───────────────────────────────────────────────────────
    def pct(key_a, key_b):
        c = sum(1 for r in enriched if ag(r.get(key_a), r.get(key_b)))
        return round(c / n * 100, 1)

    def all_pct(keys):
        def same(r):
            vals = list(set((r.get(k) or "").lower().strip() for k in keys if r.get(k) and r.get(k).lower() != "unknown"))
            return len(vals) == 1
        return round(sum(1 for r in enriched if same(r)) / n * 100, 1)

    is_coreml = dataset == "coreml"

    if is_coreml:
        # Politics 2024 — compare all 9 evaluators vs ensemble (vote_score>=5)
        # This matches paper Table 6: Individual LLM Adjudicator vs Ensemble Ground Truth
        human_agreement = [
            {"label": "LLM0 ↔ Ensemble",     "value": pct("llm0",     "ensemble"), "sub": "CoreML original vs θ≥5"},
            {"label": "LLM3 ↔ Ensemble",     "value": pct("llm3",     "ensemble"), "sub": "CoreML ≥3LLM vs θ≥5"},
            {"label": "LLM4 ↔ Ensemble",     "value": pct("llm4",     "ensemble"), "sub": "CoreML ≥4LLM vs θ≥5"},
            {"label": "OpenAI ↔ Ensemble",   "value": pct("openai",   "ensemble"), "sub": "GPT-4 vs θ≥5"},
            {"label": "Gemini ↔ Ensemble",   "value": pct("gemini",   "ensemble"), "sub": "Gemini vs θ≥5"},
            {"label": "Claude ↔ Ensemble",   "value": pct("claude",   "ensemble"), "sub": "Claude vs θ≥5"},
            {"label": "Grok ↔ Ensemble",     "value": pct("grok",     "ensemble"), "sub": "Grok vs θ≥5"},
            {"label": "LLaMA ↔ Ensemble",    "value": pct("llama",    "ensemble"), "sub": "LLaMA vs θ≥5"},
            {"label": "DeepSeek ↔ Ensemble", "value": pct("deepseek", "ensemble"), "sub": "DeepSeek vs θ≥5"},
        ]
    else:
        human_agreement = [
            {"label": "LLM0 ↔ Human",   "value": pct("llm0",   "human"),  "sub": "Original CoreML vs crowd"},
            {"label": "LLM3 ↔ Human",   "value": pct("llm3",   "human"),  "sub": "≥3 LLM corrected vs crowd"},
            {"label": "LLM4 ↔ Human",   "value": pct("llm4",   "human"),  "sub": "≥4 LLM corrected vs crowd"},
            {"label": "OpenAI ↔ Human",  "value": pct("openai", "human"),  "sub": "GPT-4o-mini vs crowd"},
            {"label": "Gemini ↔ Human",  "value": pct("gemini", "human"),  "sub": "Gemini vs crowd"},
            {"label": "Claude ↔ Human",  "value": pct("claude", "human"),  "sub": "Claude Haiku vs crowd"},
        ]
    has_human = not is_coreml

    coreml_agreement = [
        {"label": "LLM0 ↔ LLM3",    "value": pct("llm0", "llm3"),   "sub": "CoreML internal"},
        {"label": "LLM0 ↔ LLM4",    "value": pct("llm0", "llm4"),   "sub": "CoreML internal"},
        {"label": "LLM3 ↔ LLM4",    "value": pct("llm3", "llm4"),   "sub": "CoreML internal"},
        {"label": "All CoreML",      "value": all_pct(["llm0","llm3","llm4"]), "sub": "3-way CoreML consensus"},
    ]

    if is_coreml:
        llm_agreement = [
            {"label": "OpenAI ↔ Gemini",   "value": pct("openai",   "gemini"),   "sub": "LLM vs LLM"},
            {"label": "OpenAI ↔ Claude",   "value": pct("openai",   "claude"),   "sub": "LLM vs LLM"},
            {"label": "OpenAI ↔ Grok",     "value": pct("openai",   "grok"),     "sub": "LLM vs LLM"},
            {"label": "Gemini ↔ Grok",     "value": pct("gemini",   "grok"),     "sub": "LLM vs LLM"},
            {"label": "Claude ↔ Grok",     "value": pct("claude",   "grok"),     "sub": "LLM vs LLM"},
            {"label": "LLaMA ↔ DeepSeek",  "value": pct("llama",    "deepseek"), "sub": "LLM vs LLM"},
            {"label": "All 6 LLMs",        "value": all_pct(["openai","gemini","claude","grok","llama","deepseek"]), "sub": "6-way LLM consensus"},
        ]
    else:
        llm_agreement = [
            {"label": "OpenAI ↔ Gemini", "value": pct("openai", "gemini"), "sub": "LLM vs LLM"},
            {"label": "OpenAI ↔ Claude", "value": pct("openai", "claude"), "sub": "LLM vs LLM"},
            {"label": "Gemini ↔ Claude", "value": pct("gemini", "claude"), "sub": "LLM vs LLM"},
            {"label": "All 3 LLMs",      "value": all_pct(["openai","gemini","claude"]), "sub": "3-way LLM consensus"},
            {"label": "All 7 Agree",      "value": all_pct(["human","llm0","llm3","llm4","openai","gemini","claude"]), "sub": "Full consensus"},
        ]

    # ── Build analysis prompt ─────────────────────────────────────────────────
    sample_txt = "\n".join([
        f'Post {i+1}: "{r["text"][:100]}"\n'
        f'  Human:{r["human"]} | LLM0:{r["llm0"]} | LLM3:{r["llm3"]} | LLM4:{r["llm4"]} | OpenAI:{r["openai"]} | Gemini:{r["gemini"]} | Claude:{r["claude"]}'
        for i, r in enumerate(enriched[:10])
    ])

    prompt = f"""You are an AI safety researcher analyzing hate speech classification across 7 evaluators.
IMPORTANT: No evaluator is ground truth — all are perspectives on the same content.

Evaluators: Human crowd label, CoreML LLM0 (original), LLM3 (≥3 LLM corrected), LLM4 (≥4 LLM corrected), OpenAI GPT-4o-mini, Google Gemini, Anthropic Claude Haiku.

Dataset: Kaggle hate speech posts, focus: {focus}, {n} posts analyzed.

Key agreement rates:
- LLM0↔Human: {pct("llm0","human")}% | LLM3↔Human: {pct("llm3","human")}% | LLM4↔Human: {pct("llm4","human")}%
- OpenAI↔Human: {pct("openai","human")}% | Gemini↔Human: {pct("gemini","human")}% | Claude↔Human: {pct("claude","human")}%
- OpenAI↔Gemini: {pct("openai","gemini")}% | OpenAI↔Claude: {pct("openai","claude")}% | Gemini↔Claude: {pct("gemini","claude")}%
- All 7 agree: {all_pct(["human","llm0","llm3","llm4","openai","gemini","claude"])}%

Sample predictions:
{sample_txt}

Analyze (no assumed ground truth):
1. Which evaluator clusters with which others — are LLMs more aligned with each other or with human labels?
2. What does low all-7 agreement tell us about the subjectivity of hate speech?
3. Where CoreML models diverge from LLMs — what does that reveal about training data differences?
4. Key insight for AI safety researchers on using multiple evaluators without a ground truth.

Max 300 words."""

    try:
        analysis = get_analysis(prompt, openai_key)
    except Exception as e:
        analysis = f"Analysis unavailable: {e}"

    # ── Samples for table ─────────────────────────────────────────────────────
    if is_coreml:
        all_keys = ["ensemble","llm0","llm3","llm4","openai","gemini","claude","grok","llama","deepseek"]
    else:
        all_keys = ["human","llm0","llm3","llm4","openai","gemini","claude"]

    samples = [
        {
            **r,
            "all_agree": len(set((r.get(k) or "").lower().strip() for k in all_keys if r.get(k) and (r.get(k) or "").lower() not in ("unknown",""))) == 1
        }
        for r in enriched[:20]
    ]

    # For CoreML, add extra columns to table
    if is_coreml:
        table_cols = ["Post","Ensemble(θ≥5)","LLM0","LLM3","LLM4","OpenAI","Gemini","Claude","Grok","LLaMA","DeepSeek","All?"]
        for s in samples:
            s["preds_extra"] = [s.get("grok","?"), s.get("llama","?"), s.get("deepseek","?")]
    else:
        table_cols = ["Post","Human","LLM0","LLM3","LLM4","OpenAI","Gemini","Claude","All?"]

    return jsonify({
        "total":             n,
        "focus":             focus,
        "dataset":           dataset,
        "human_agreement":   human_agreement,
        "coreml_agreement":  coreml_agreement,
        "llm_agreement":     llm_agreement,
        "analysis":          analysis,
        "samples":           samples,
        "table_cols":        table_cols,
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 EmakiaEval Multi-LLM Dashboard → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
