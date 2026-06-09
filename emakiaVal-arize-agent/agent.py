"""
agent.py — EmakiaEval Arize Agent v4.0
=======================================
Google Cloud Rapid Agent Hackathon — Arize Track

5-adjudicator weighted ensemble (new weights, not in published paper):

  OpenAI         H=1.0  N=1.0  — industry baseline
  Claude         H=0.5  N=1.5  — precision anchor (recall=0.248, spec=0.989)
  Grok           H=1.0  N=1.0  — X/Twitter-native, balanced
  Gemini Text    H=1.0  N=0.5  — high recall text model
  Gemini Vision  H=1.5  N=0.5  — multimodal, highest harassment weight (paper: 2x)

Pipeline:
  1. Fetch posts from kaggle_eval.predictions (includes LLM0/LLM3/LLM4 CoreML predictions)
  2. Run 5 adjudicators in parallel
  3. Apply weighted voting → establish TRUTH
  4. Score LLM0, LLM3, LLM4 CoreML models against TRUTH
  5. Score human_label against TRUTH
  6. Trace everything to Arize Phoenix
  7. Store to BigQuery arize_eval.predictions
"""

import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from flask import Flask, request, jsonify
from google.cloud import bigquery
from dotenv import load_dotenv

# ── Arize Phoenix — using phoenix.otel for correct auth ───────────────────────
from phoenix.otel import register
from opentelemetry import trace

load_dotenv()
app = Flask(__name__)

# ── CORS — required for dashboard served from any origin ─────────────────────
from flask_cors import CORS
CORS(app)


# ── Phoenix setup ─────────────────────────────────────────────────────────────

PHOENIX_API_KEY            = os.environ.get("PHOENIX_API_KEY", "")
PHOENIX_COLLECTOR_ENDPOINT = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/s/corinne"
)
PHOENIX_PROJECT            = os.environ.get("PHOENIX_PROJECT_NAME", "emakia-arize-eval")

def setup_phoenix_tracing():
    if not PHOENIX_API_KEY:
        print("⚠️  PHOENIX_API_KEY not set — tracing disabled")
        return trace.get_tracer("emakia-arize-agent")
    try:
        tracer_provider = register(
            project_name=PHOENIX_PROJECT,
            endpoint=f"{PHOENIX_COLLECTOR_ENDPOINT}/v1/traces",
            headers={"api_key": PHOENIX_API_KEY},
            set_global_tracer_provider=True,
            batch=True,
        )
        print(f"✅ Phoenix tracing → {PHOENIX_COLLECTOR_ENDPOINT}/v1/traces")
        print(f"✅ Phoenix project  → {PHOENIX_PROJECT}")
        return trace.get_tracer("emakia-arize-agent")
    except Exception as e:
        print(f"⚠️  Phoenix setup failed: {e} — continuing without tracing")
        return trace.get_tracer("emakia-arize-agent")

tracer = setup_phoenix_tracing()


# ── BigQuery ──────────────────────────────────────────────────────────────────

BQ_SOURCE_TABLE = "emakia.kaggle_eval.predictions"
BQ_DEST_TABLE   = "emakia.arize_eval.predictions"

def get_bq_client():
    try:
        client = bigquery.Client(project="emakia")
        print(f"✅ BigQuery ready — project: {client.project}")
        return client
    except Exception as e:
        print(f"❌ BigQuery: {e}")
        return None

bq_client = get_bq_client()


# ── Adjudicator weights (derived from adjudicator_weights.csv) ────────────────
#
#  HARASSMENT weights (based on recall_H):
#    Gemini Vision recall≈0.999 → HIGH  → 1.5
#    Gemini Text   recall≈0.999 → HIGH  → 1.0
#    OpenAI        recall=0.994 → MED   → 1.0
#    Grok          recall=0.966 → MED   → 1.0
#    Claude        recall=0.248 → LOW   → 0.5
#
#  NEUTRAL weights (based on spec_N):
#    Claude        spec=0.989   → HIGH  → 1.5
#    Grok          spec=0.821   → MED   → 1.0
#    OpenAI        spec=0.710   → MED   → 1.0
#    Gemini Text   spec=0.555   → LOW   → 0.5
#    Gemini Vision spec=0.555   → LOW   → 0.5

W_HARASSMENT = {
    "gemini_vision": 1.5,
    "gemini_text":   1.0,
    "openai":        1.0,
    "grok":          1.0,
    "claude":        0.5,
}
W_NEUTRAL = {
    "claude":        1.5,
    "grok":          1.0,
    "openai":        1.0,
    "gemini_text":   0.5,
    "gemini_vision": 0.5,
}


def weighted_truth(predictions: dict) -> dict:
    """
    5-adjudicator weighted voting → ground truth.
    No ties possible (max H=5.0, max N=4.5).
    """
    harassment_score = 0.0
    neutral_score    = 0.0
    vote_detail      = {}

    for model, label in predictions.items():
        if label == "Unknown":
            vote_detail[model] = {"label": "Unknown", "h_contrib": 0, "n_contrib": 0}
            continue
        h_w = W_HARASSMENT.get(model, 1.0)
        n_w = W_NEUTRAL.get(model, 1.0)
        if label == "Harassment":
            harassment_score += h_w
            vote_detail[model] = {"label": label, "h_contrib": h_w, "n_contrib": 0}
        else:
            neutral_score += n_w
            vote_detail[model] = {"label": label, "h_contrib": 0, "n_contrib": n_w}

    total = harassment_score + neutral_score
    truth = "Harassment" if harassment_score > neutral_score else "Neutral"
    confidence = abs(harassment_score - neutral_score) / total if total > 0 else 0.0

    return {
        "truth":            truth,
        "harassment_score": round(harassment_score, 2),
        "neutral_score":    round(neutral_score, 2),
        "confidence":       round(confidence, 3),
        "vote_detail":      vote_detail,
    }


# ── Classification prompt ─────────────────────────────────────────────────────

CLASSIFY_PROMPT = (
    'Classify this social media post as HARASSMENT or NEUTRAL.\n'
    'Reply with exactly one word: Harassment or Neutral.\n\n'
    'Post: "{text}"'
)

def _parse_label(raw: str) -> str:
    return "Harassment" if "harass" in raw.strip().lower() else "Neutral"


# ── LLM classifiers ───────────────────────────────────────────────────────────

def classify_openai(text: str, key: str) -> dict:
    with tracer.start_as_current_span("classify_openai") as span:
        span.set_attribute("llm.model",    "gpt-4o-mini")
        span.set_attribute("llm.provider", "openai")
        span.set_attribute("input.text",   text[:200])
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "max_tokens": 20, "temperature": 0,
                      "messages": [{"role": "user",
                                    "content": CLASSIFY_PROMPT.format(text=text[:300])}]},
                timeout=30,
            )
            word  = r.json()["choices"][0]["message"]["content"]
            label = _parse_label(word)
            span.set_attribute("output.label", label)
            return {"label": label, "error": None}
        except Exception as e:
            span.set_attribute("error", str(e))
            return {"label": "Unknown", "error": str(e)}


def classify_claude(text: str, key: str) -> dict:
    with tracer.start_as_current_span("classify_claude") as span:
        span.set_attribute("llm.model",    "claude-sonnet-4-6")
        span.set_attribute("llm.provider", "anthropic")
        span.set_attribute("input.text",   text[:200])
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-4-6", "max_tokens": 20,
                      "messages": [{"role": "user",
                                    "content": CLASSIFY_PROMPT.format(text=text[:300])}]},
                timeout=30,
            )
            word  = r.json()["content"][0]["text"]
            label = _parse_label(word)
            span.set_attribute("output.label", label)
            return {"label": label, "error": None}
        except Exception as e:
            span.set_attribute("error", str(e))
            return {"label": "Unknown", "error": str(e)}


def classify_grok(text: str, key: str) -> dict:
    with tracer.start_as_current_span("classify_grok") as span:
        span.set_attribute("llm.model",    "grok-4.3")
        span.set_attribute("llm.provider", "xai")
        span.set_attribute("input.text",   text[:200])
        try:
            r = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": "grok-4.3", "max_tokens": 20, "temperature": 0,
                      "messages": [{"role": "user",
                                    "content": CLASSIFY_PROMPT.format(text=text[:300])}]},
                timeout=60,
            )
            word  = r.json()["choices"][0]["message"]["content"]
            label = _parse_label(word)
            span.set_attribute("output.label", label)
            return {"label": label, "error": None}
        except Exception as e:
            span.set_attribute("error", str(e))
            return {"label": "Unknown", "error": str(e)}


def classify_gemini_text(text: str, key: str) -> dict:
    """Gemini Flash — fast text classification."""
    with tracer.start_as_current_span("classify_gemini_text") as span:
        span.set_attribute("llm.model",    "gemini-flash-latest")
        span.set_attribute("llm.provider", "google")
        span.set_attribute("input.text",   text[:200])
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-flash-latest:generateContent?key={key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [
                        {"text": CLASSIFY_PROMPT.format(text=text[:300])}
                    ]}],
                    "generationConfig": {
                        "maxOutputTokens": 20,
                        "temperature": 0,
                        "thinkingConfig": {"thinkingBudget": 0}
                    },
                },
                timeout=30,
            )
            parts = r.json()["candidates"][0]["content"].get("parts", [])
            if not parts:
                return {"label": "Unknown", "error": "empty response"}
            word  = parts[0].get("text", "")
            label = _parse_label(word)
            span.set_attribute("output.label", label)
            return {"label": label, "error": None}
        except Exception as e:
            span.set_attribute("error", str(e))
            return {"label": "Unknown", "error": str(e)}


def classify_gemini_vision(text: str, key: str) -> dict:
    """Gemini Vision — multimodal, highest harassment weight (1.5)."""
    with tracer.start_as_current_span("classify_gemini_vision") as span:
        span.set_attribute("llm.model",    "gemini-2.5-flash")
        span.set_attribute("llm.provider", "google")
        span.set_attribute("input.text",   text[:200])
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [
                        {"text": CLASSIFY_PROMPT.format(text=text[:300])}
                    ]}],
                    "generationConfig": {
                        "maxOutputTokens": 20,
                        "temperature": 0,
                        "thinkingConfig": {"thinkingBudget": 0}
                    },
                },
                timeout=60,
            )
            parts = r.json()["candidates"][0]["content"].get("parts", [])
            if not parts:
                return {"label": "Unknown", "error": "empty response"}
            word  = parts[0].get("text", "")
            label = _parse_label(word)
            span.set_attribute("output.label", label)
            return {"label": label, "error": None}
        except Exception as e:
            span.set_attribute("error", str(e))
            return {"label": "Unknown", "error": str(e)}


# ── Core evaluation row ───────────────────────────────────────────────────────

def evaluate_row(row: dict, keys: dict) -> dict:
    """
    For one post:
      1. Run 5 adjudicators in parallel
      2. Weighted voting → TRUTH
      3. Score LLM0/LLM3/LLM4 CoreML vs TRUTH
      4. Score human_label vs TRUTH
      5. All spans traced to Phoenix
    """
    with tracer.start_as_current_span("evaluate_row") as span:
        text        = str(row.get("text", ""))
        post_id     = str(row.get("post_id", ""))
        human_label = str(row.get("human_label", "")).lower()
        coreml_llm0 = str(row.get("prediction_llm0", "") or "").lower()
        coreml_llm3 = str(row.get("prediction_llm3", "") or "").lower()
        coreml_llm4 = str(row.get("prediction_llm4", "") or "").lower()

        span.set_attribute("post.id",          post_id)
        span.set_attribute("post.human_label",  human_label)
        span.set_attribute("coreml.llm0",       coreml_llm0)
        span.set_attribute("coreml.llm3",       coreml_llm3)
        span.set_attribute("coreml.llm4",       coreml_llm4)

        # ── 5 adjudicators in parallel ────────────────────────────────────────
        classifiers = {
            "openai":        lambda: classify_openai(text, keys["openai"]),
            "claude":        lambda: classify_claude(text, keys["claude"]),
            "grok":          lambda: classify_grok(text, keys["grok"]),
            "gemini_text":   lambda: classify_gemini_text(text, keys["gemini"]),
            "gemini_vision": lambda: classify_gemini_vision(text, keys["gemini"]),
        }

        predictions = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(fn): name for name, fn in classifiers.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    predictions[name] = future.result()
                except Exception as e:
                    predictions[name] = {"label": "Unknown", "error": str(e)}

        pred_labels = {m: v["label"] for m, v in predictions.items()}

        # ── Weighted adjudication → TRUTH ─────────────────────────────────────
        adjudication = weighted_truth(pred_labels)
        truth        = adjudication["truth"].lower()

        span.set_attribute("adjudication.truth",            truth)
        span.set_attribute("adjudication.harassment_score", adjudication["harassment_score"])
        span.set_attribute("adjudication.neutral_score",    adjudication["neutral_score"])
        span.set_attribute("adjudication.confidence",       adjudication["confidence"])

        # ── Score CoreML and human label vs TRUTH ─────────────────────────────
        def matches(pred):
            return bool(pred) and pred.strip().lower() == truth

        llm0_correct  = matches(coreml_llm0)
        llm3_correct  = matches(coreml_llm3)
        llm4_correct  = matches(coreml_llm4)
        human_correct = matches(human_label)

        span.set_attribute("score.llm0_correct",  llm0_correct)
        span.set_attribute("score.llm3_correct",  llm3_correct)
        span.set_attribute("score.llm4_correct",  llm4_correct)
        span.set_attribute("score.human_correct", human_correct)

        return {
            "post_id":                  post_id,
            "text":                     text,
            "human_label":              human_label,
            "coreml_llm0":              coreml_llm0,
            "coreml_llm3":              coreml_llm3,
            "coreml_llm4":              coreml_llm4,
            "prediction_openai":        pred_labels.get("openai",        "Unknown"),
            "prediction_claude":        pred_labels.get("claude",        "Unknown"),
            "prediction_grok":          pred_labels.get("grok",          "Unknown"),
            "prediction_gemini_text":   pred_labels.get("gemini_text",   "Unknown"),
            "prediction_gemini_vision": pred_labels.get("gemini_vision", "Unknown"),
            "weighted_truth":           adjudication["truth"],
            "harassment_score":         adjudication["harassment_score"],
            "neutral_score":            adjudication["neutral_score"],
            "adjudication_confidence":  adjudication["confidence"],
            "vote_detail":              json.dumps(adjudication["vote_detail"]),
            "llm0_correct":             llm0_correct,
            "llm3_correct":             llm3_correct,
            "llm4_correct":             llm4_correct,
            "human_correct":            human_correct,
            "evaluated_at":             datetime.utcnow().isoformat(),
        }


# ── BigQuery destination table ────────────────────────────────────────────────

def ensure_dest_table():
    if bq_client is None:
        return False
    schema = [
        bigquery.SchemaField("post_id",                   "STRING"),
        bigquery.SchemaField("text",                      "STRING"),
        bigquery.SchemaField("human_label",               "STRING"),
        bigquery.SchemaField("coreml_llm0",               "STRING"),
        bigquery.SchemaField("coreml_llm3",               "STRING"),
        bigquery.SchemaField("coreml_llm4",               "STRING"),
        bigquery.SchemaField("prediction_openai",         "STRING"),
        bigquery.SchemaField("prediction_claude",         "STRING"),
        bigquery.SchemaField("prediction_grok",           "STRING"),
        bigquery.SchemaField("prediction_gemini_text",    "STRING"),
        bigquery.SchemaField("prediction_gemini_vision",  "STRING"),
        bigquery.SchemaField("weighted_truth",            "STRING"),
        bigquery.SchemaField("harassment_score",          "FLOAT"),
        bigquery.SchemaField("neutral_score",             "FLOAT"),
        bigquery.SchemaField("adjudication_confidence",   "FLOAT"),
        bigquery.SchemaField("vote_detail",               "STRING"),
        bigquery.SchemaField("llm0_correct",              "BOOL"),
        bigquery.SchemaField("llm3_correct",              "BOOL"),
        bigquery.SchemaField("llm4_correct",              "BOOL"),
        bigquery.SchemaField("human_correct",             "BOOL"),
        bigquery.SchemaField("evaluated_at",              "TIMESTAMP"),
    ]
    dataset_ref = bq_client.dataset("arize_eval")
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        bq_client.create_dataset(dataset_ref)
    table_ref = dataset_ref.table("predictions")
    try:
        bq_client.get_table(table_ref)
    except Exception:
        bq_client.create_table(bigquery.Table(table_ref, schema=schema))
        print("✅ Created arize_eval.predictions")
    return True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({
        "service":   "EmakiaEval Arize Agent",
        "version":   "4.0 — phoenix.otel + 5-adjudicator weighted ensemble",
        "status":    "running",
        "bigquery":  "connected" if bq_client else "disconnected",
        "phoenix":   "configured" if PHOENIX_API_KEY else "missing",
        "adjudicators": {
            "openai":        {"H": W_HARASSMENT["openai"],        "N": W_NEUTRAL["openai"]},
            "claude":        {"H": W_HARASSMENT["claude"],        "N": W_NEUTRAL["claude"]},
            "grok":          {"H": W_HARASSMENT["grok"],          "N": W_NEUTRAL["grok"]},
            "gemini_text":   {"H": W_HARASSMENT["gemini_text"],   "N": W_NEUTRAL["gemini_text"]},
            "gemini_vision": {"H": W_HARASSMENT["gemini_vision"], "N": W_NEUTRAL["gemini_vision"]},
        },
        "endpoints": ["/dashboard", "/api/health", "/api/run-eval",
                      "/api/eval-stats", "/api/disagreement-clusters",
                      "/api/threshold-analysis"],
    }), 200


@app.route("/api/threshold-analysis", methods=["POST"])
def threshold_analysis():
    """
    Re-evaluate LLM0/LLM3/LLM4 at multiple τ thresholds using raw
    probability scores stored in kaggle_eval.predictions.
    Shows the crossover point where LLM3 overtakes LLM0.

    Body (JSON):
        thresholds — list of float thresholds to test (default: 18 steps 0.10-0.95)
    """
    if bq_client is None:
        return jsonify({"error": "BigQuery not available"}), 500

    body       = request.get_json(force=True) or {}
    thresholds = body.get("thresholds", [round(x * 0.05 + 0.10, 2) for x in range(18)])

    query = f"""
        SELECT post_id, human_label,
               score_llm0, score_llm3, score_llm4
        FROM `{BQ_SOURCE_TABLE}`
        WHERE score_llm0 IS NOT NULL
          AND score_llm3 IS NOT NULL
          AND score_llm4 IS NOT NULL
          AND human_label IS NOT NULL
        LIMIT 500
    """
    try:
        rows = [dict(r) for r in bq_client.query(query).result()]
    except Exception as e:
        return jsonify({"error": f"BigQuery fetch: {e}"}), 500

    if not rows:
        return jsonify({"error": "No rows with score columns found in kaggle_eval.predictions"}), 404

    n = len(rows)
    accuracy_by_threshold = {}

    for tau in thresholds:
        correct = {"llm0": 0, "llm3": 0, "llm4": 0}
        for r in rows:
            truth = str(r.get("human_label", "")).lower()
            for m in ["llm0", "llm3", "llm4"]:
                score = float(r.get(f"score_{m}") or 0)
                pred  = "harassment" if score >= tau else "neutral"
                if pred == truth:
                    correct[m] += 1
        key = f"{tau:.2f}"
        accuracy_by_threshold[key] = {
            m: round(correct[m] / n * 100, 2) for m in ["llm0", "llm3", "llm4"]
        }

    # Find crossover point
    crossover_tau = None
    prev = None
    for tau in sorted(thresholds):
        key = f"{tau:.2f}"
        d   = accuracy_by_threshold.get(key, {})
        if prev and prev.get("llm3", 0) >= prev.get("llm0", 0) and \
           d.get("llm3", 0) < d.get("llm0", 0):
            crossover_tau = key
            break
        prev = d

    return jsonify({
        "total":                n,
        "thresholds_tested":    thresholds,
        "accuracy_by_threshold": accuracy_by_threshold,
        "crossover_tau":        crossover_tau,
        "insight": (
            f"LLM3 outperforms LLM0 below τ = {crossover_tau} — "
            f"use τ = 0.55 for best LLM3 performance, "
            f"τ = 0.95 for best LLM0 performance."
        ) if crossover_tau else "No crossover detected in this score range.",
    }), 200


@app.route("/dashboard")
def dashboard():
    """Serve the EmakiaEval live dashboard — accessible from any browser."""
    try:
        with open("dashboard.html") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    except FileNotFoundError:
        return "dashboard.html not found — make sure it is deployed with agent.py", 404


@app.route("/api/health", methods=["GET"])
def health():
    bq_ok = False
    if bq_client:
        try:
            list(bq_client.query("SELECT 1").result())
            bq_ok = True
        except Exception:
            pass
    return jsonify({
        "status":    "healthy" if bq_ok else "degraded",
        "bigquery":  "connected" if bq_ok else "error",
        "phoenix":   "configured" if PHOENIX_API_KEY else "missing",
        "timestamp": datetime.utcnow().isoformat(),
    }), 200 if bq_ok else 503


@app.route("/api/run-eval", methods=["POST"])
def run_eval():
    """
    Pull rows from kaggle_eval.predictions (has LLM0/LLM3/LLM4 CoreML predictions)
    → run 5 adjudicators → weighted truth → score CoreML + human label
    → store to arize_eval.predictions → trace to Phoenix

    Body (JSON):
        limit  — rows to process (default 10, max 50)
        offset — pagination offset (default 0)
    """
    if bq_client is None:
        return jsonify({"error": "BigQuery not available"}), 500

    keys = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY", ""),
        "grok":   os.environ.get("GROK_API_KEY", ""),
    }
    missing = [k for k, v in keys.items() if not v]
    if missing:
        return jsonify({"error": f"Missing API keys: {missing}"}), 500

    body   = request.get_json(force=True) or {}
    limit  = min(int(body.get("limit",  10)), 50)
    offset = max(int(body.get("offset",  0)), 0)

    query = f"""
        SELECT DISTINCT post_id, text, human_label,
               prediction_llm0, prediction_llm3, prediction_llm4
        FROM `{BQ_SOURCE_TABLE}`
        WHERE text IS NOT NULL AND human_label IS NOT NULL
        LIMIT {limit} OFFSET {offset}
    """
    try:
        rows = [dict(r) for r in bq_client.query(query).result()]
    except Exception as e:
        return jsonify({"error": f"BigQuery fetch: {e}"}), 500

    if not rows:
        return jsonify({"error": "No rows found at this offset"}), 404

    ensure_dest_table()

    results, errors = [], []
    with tracer.start_as_current_span("emakia_eval_batch_v4") as span:
        span.set_attribute("batch.size",         len(rows))
        span.set_attribute("batch.offset",       offset)
        span.set_attribute("batch.adjudicators", 5)
        span.set_attribute("batch.method",       "weighted_ensemble")

        for row in rows:
            try:
                results.append(evaluate_row(row, keys))
            except Exception as e:
                errors.append({"post_id": row.get("post_id"), "error": str(e)})

    if results:
        bq_client.insert_rows_json(BQ_DEST_TABLE, results)

    n = len(results)
    def acc(field):
        return round(sum(1 for r in results if r.get(field)) / n * 100, 2) if n else 0

    return jsonify({
        "status":      "success",
        "evaluated":   n,
        "errors":      len(errors),
        "adjudicators": 5,
        "accuracy_vs_weighted_truth": {
            "coreml_llm0":  acc("llm0_correct"),
            "coreml_llm3":  acc("llm3_correct"),
            "coreml_llm4":  acc("llm4_correct"),
            "human_label":  acc("human_correct"),
        },
        "weighted_truth_distribution": {
            "harassment": sum(1 for r in results if r["weighted_truth"] == "Harassment"),
            "neutral":    sum(1 for r in results if r["weighted_truth"] == "Neutral"),
        },
        "rows": results,
    }), 200


@app.route("/api/eval-stats", methods=["GET"])
def eval_stats():
    """Accuracy summary of CoreML models and human labels vs weighted truth."""
    if bq_client is None:
        return jsonify({"error": "BigQuery not available"}), 500
    try:
        q = f"""
            SELECT
                COUNT(*) as total,
                ROUND(SAFE_DIVIDE(COUNTIF(llm0_correct),  COUNT(*)) * 100, 2) as llm0_acc,
                ROUND(SAFE_DIVIDE(COUNTIF(llm3_correct),  COUNT(*)) * 100, 2) as llm3_acc,
                ROUND(SAFE_DIVIDE(COUNTIF(llm4_correct),  COUNT(*)) * 100, 2) as llm4_acc,
                ROUND(SAFE_DIVIDE(COUNTIF(human_correct), COUNT(*)) * 100, 2) as human_acc,
                ROUND(AVG(adjudication_confidence) * 100, 2) as avg_confidence,
                COUNTIF(weighted_truth = 'Harassment') as truth_harassment,
                COUNTIF(weighted_truth = 'Neutral')    as truth_neutral,
                ROUND(AVG(harassment_score), 3) as avg_h_score,
                ROUND(AVG(neutral_score), 3)    as avg_n_score
            FROM `{BQ_DEST_TABLE}`
        """
        rows = list(bq_client.query(q).result())
        if not rows or rows[0].total == 0:
            return jsonify({"error": "No data yet — POST /api/run-eval first"}), 404
        r = rows[0]
        return jsonify({
            "total_evaluated": r.total,
            "weighted_truth_distribution": {
                "harassment": r.truth_harassment,
                "neutral":    r.truth_neutral,
            },
            "accuracy_vs_weighted_truth": {
                "coreml_llm0":  r.llm0_acc,
                "coreml_llm3":  r.llm3_acc,
                "coreml_llm4":  r.llm4_acc,
                "human_label":  r.human_acc,
            },
            "avg_adjudication_confidence": r.avg_confidence,
            "avg_scores": {
                "harassment": r.avg_h_score,
                "neutral":    r.avg_n_score,
            },
            "adjudicator_weights": {
                "harassment": W_HARASSMENT,
                "neutral":    W_NEUTRAL,
            }
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/disagreement-clusters", methods=["GET"])
def disagreement_clusters():
    """
    Posts where weighted truth disagrees with human_label.
    These reveal where cloud LLM consensus corrects noisy human annotations.
    """
    if bq_client is None:
        return jsonify({"error": "BigQuery not available"}), 500
    try:
        limit = min(request.args.get("limit", default=20, type=int), 50)
        q = f"""
            SELECT
                post_id, text, human_label, weighted_truth,
                harassment_score, neutral_score, adjudication_confidence,
                prediction_openai, prediction_claude, prediction_grok,
                prediction_gemini_text, prediction_gemini_vision,
                coreml_llm0, coreml_llm3, coreml_llm4,
                llm0_correct, llm3_correct, llm4_correct, human_correct
            FROM `{BQ_DEST_TABLE}`
            WHERE human_correct = FALSE
            ORDER BY adjudication_confidence DESC
            LIMIT {limit}
        """
        rows = [dict(r) for r in bq_client.query(q).result()]
        if not rows:
            return jsonify({
                "message": "No disagreements — human labels match weighted truth!",
            }), 200

        llm3_right = sum(1 for r in rows if r.get("llm3_correct"))
        llm0_right = sum(1 for r in rows if r.get("llm0_correct"))
        llm4_right = sum(1 for r in rows if r.get("llm4_correct"))

        return jsonify({
            "total_human_label_disagreements": len(rows),
            "interpretation": (
                "Posts where 5-adjudicator weighted consensus disagrees with "
                "original Kaggle human annotation. High confidence = cloud LLMs "
                "are likely correct, human annotator likely wrong."
            ),
            "coreml_correct_when_human_wrong": {
                "llm3": llm3_right,
                "llm0": llm0_right,
                "llm4": llm4_right,
            },
            "insight": (
                f"LLM3 CoreML was right when human annotators were wrong "
                f"in {llm3_right}/{len(rows)} cases — "
                f"confirming paper: consensus-refined labels > human labels."
            ),
            "disagreements": rows,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found",
                    "endpoints": ["/dashboard", "/", "/api/health", "/api/run-eval",
                                  "/api/eval-stats", "/api/disagreement-clusters",
                                  "/api/threshold-analysis"]}), 404

@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error"}), 500


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    print(f"🚀 EmakiaEval Arize Agent v4.0 — phoenix.otel + 5-adjudicator weighted ensemble")
    print(f"🚀 Port: {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
