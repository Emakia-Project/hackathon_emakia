"""
kaggle_gemma4_integration.py
============================
REAL Gemma 4 inference using OPEN WEIGHTS — no Gemini API, no GOOGLE_API_KEY.

Two backends are supported. The active one is chosen by env var
`GEMMA4_BACKEND` ("llama" or "transformers"). Default is "llama".

  • llama  → llama-cpp-python loading a GGUF quantized model.
             Lowest RAM footprint. Recommended for Streamlit Community Cloud.
             E2B at Q4_K_M ≈ 1.2–1.5 GB.

  • transformers → Hugging Face transformers loading the full model.
             Higher quality, much higher RAM. Recommended for Hugging Face
             Spaces (free 16 GB tier), Cloud Run with 4+ GB, or local GPU.

Public API (unchanged from your previous version, so app.py keeps working):
    load_kaggle_tweets(limit) -> pd.DataFrame
    classify_with_gemma4(text) -> dict
    classify_batch_with_gemma4(texts, progress_bar=None) -> list[dict]
    analyze_image_with_gemma4(image_url, tweet_text) -> dict
    render_kaggle_data_section() -> pd.DataFrame | None
    render_gemma4_classifier_section(df) -> None

Setup (Streamlit Cloud secrets.toml or local env):
    KAGGLE_USERNAME = "your_kaggle_username"
    KAGGLE_API_TOKEN = "your_kaggle_api_token"
    # Optional overrides:
    GEMMA4_BACKEND   = "llama"          # or "transformers"
    GEMMA_GGUF_REPO  = "..."            # HF repo containing GGUF file
    GEMMA_GGUF_FILE  = "...q4_k_m.gguf" # filename within that repo
    GEMMA_HF_REPO    = "google/gemma-4-e2b-it"
"""

import os
import json
import pandas as pd
import streamlit as st

# ── Configuration ────────────────────────────────────────────────────────────
GEMMA4_BACKEND = os.environ.get("GEMMA4_BACKEND", "llama").lower()

# llama.cpp / GGUF defaults — VERIFIED working repos as of May 2026.
# Default: bartowski's E2B Q4_K_M (~1.5 GB, runs on any laptop with 4+ GB RAM).
#
# Alternates if you want to swap (set GEMMA_GGUF_REPO + GEMMA_GGUF_FILE in env):
#   • unsloth/gemma-4-E2B-it-GGUF          → gemma-4-E2B-it-Q4_K_M.gguf  (~1.5 GB)
#   • ggml-org/gemma-4-E2B-it-GGUF         → gemma-4-E2B-it-Q8_0.gguf    (~5 GB, higher quality)
#   • lmstudio-community/gemma-4-E2B-it-GGUF → gemma-4-E2B-it-Q4_K_M.gguf
#   • unsloth/gemma-4-26B-A4B-it-GGUF      → gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf (~18 GB, needs lots of RAM)
#
# Google did NOT publish official Gemma 4 GGUFs — community maintainers do.
GEMMA_GGUF_REPO = os.environ.get(
    "GEMMA_GGUF_REPO",
    "bartowski/google_gemma-4-E2B-it-GGUF",
)
GEMMA_GGUF_FILE = os.environ.get(
    "GEMMA_GGUF_FILE",
    "google_gemma-4-E2B-it-Q4_K_M.gguf",
)

# transformers backend — full precision weights from HF
GEMMA_HF_REPO = os.environ.get("GEMMA_HF_REPO", "google/gemma-4-e2b-it")

# Kaggle dataset (unchanged from your version)
KAGGLE_DATASET_SLUG = os.environ.get(
    "KAGGLE_DATASET_SLUG", "corinnedavid/emakia-dataset"
)
KAGGLE_CSV_FILENAME = os.environ.get("KAGGLE_CSV_FILENAME", "tweets.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  KAGGLE DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_kaggle_tweets(limit: int = 200) -> pd.DataFrame:
    """Download the Emakia tweet dataset from Kaggle and return as a DataFrame."""
    import kaggle

    download_path = "/tmp/kaggle_emakia"
    os.makedirs(download_path, exist_ok=True)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET_SLUG, path=download_path, unzip=True, quiet=False,
    )

    csv_path = os.path.join(download_path, KAGGLE_CSV_FILENAME)
    if not os.path.exists(csv_path):
        files = os.listdir(download_path)
        raise FileNotFoundError(
            f"Expected '{KAGGLE_CSV_FILENAME}' in dataset but found: {files}\n"
            f"Update KAGGLE_CSV_FILENAME in the env or this file."
        )

    df = pd.read_csv(csv_path)
    return df.head(limit)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  GEMMA 4 — llama.cpp BACKEND (GGUF quantized, low RAM)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Gemma 4 (GGUF, llama.cpp)…")
def _load_llama_model():
    """Download (once) and load a quantized Gemma 4 GGUF via llama-cpp-python."""
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is not installed. Run:\n"
            "    pip install llama-cpp-python\n"
            "On macOS with Apple Silicon, you may want Metal acceleration:\n"
            "    CMAKE_ARGS='-DGGML_METAL=on' pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir"
        ) from e

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        ) from e

    st.info(
        f"📦 Downloading {GEMMA_GGUF_FILE} from {GEMMA_GGUF_REPO} "
        f"(first run only, cached afterward)…"
    )
    model_path = hf_hub_download(
        repo_id=GEMMA_GGUF_REPO,
        filename=GEMMA_GGUF_FILE,
        cache_dir=os.environ.get("HF_HOME", "/tmp/hf_cache"),
    )

    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=os.cpu_count() or 2,
        n_gpu_layers=0,        # CPU-only — required on Streamlit Cloud, fine on Cloud Run
        verbose=False,
    )


def _classify_llama(text: str) -> dict:
    llm = _load_llama_model()
    # Gemma 4 supports the system role natively (Apr 11 chat template update).
    # Splitting role from instruction gives slightly more reliable JSON output.
    out = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a civic content moderation AI. "
                    "You always respond with valid JSON only — no prose, no markdown."
                ),
            },
            {"role": "user", "content": _build_prompt(text)},
        ],
        max_tokens=220,
        temperature=0.1,
    )
    raw = out["choices"][0]["message"]["content"]
    return _parse_json_response(raw)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  GEMMA 4 — transformers BACKEND (full weights, higher RAM)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Gemma 4 (transformers)…")
def _load_hf_model():
    """Load full-precision Gemma 4 weights via Hugging Face transformers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_HF_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_HF_REPO,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def _classify_transformers(text: str) -> dict:
    import torch

    tokenizer, model = _load_hf_model()
    messages = [{"role": "user", "content": _build_prompt(text)}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=220,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        out[0][inputs.shape[-1]:], skip_special_tokens=True
    )
    return _parse_json_response(raw)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  SHARED PROMPT + JSON PARSING
# ═════════════════════════════════════════════════════════════════════════════

def _build_prompt(text: str) -> str:
    return f"""You are a civic content moderation AI. Analyze this social media post.

Post: "{text}"

Reply with ONLY valid JSON (no markdown, no commentary):
{{
  "label": "harassment" or "neutral",
  "score": <float 0.0-1.0 confidence>,
  "reason": "<one sentence explanation>",
  "severity": "low" or "medium" or "high"
}}"""


def _parse_json_response(raw: str) -> dict:
    """Robust JSON extraction — tolerates code fences and stray prose."""
    if not raw:
        return _err("empty model response", raw)

    clean = raw.replace("```json", "").replace("```", "").strip()
    start, end = clean.find("{"), clean.rfind("}")
    if start >= 0 and end > start:
        clean = clean[start : end + 1]

    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        return _err(f"Could not parse JSON: {raw[:120]}", raw)

    # Normalize fields and types
    result.setdefault("label", "error")
    result.setdefault("score", 0.0)
    result.setdefault("reason", "")
    result.setdefault("severity", "low")
    try:
        result["score"] = float(result["score"])
    except (TypeError, ValueError):
        result["score"] = 0.0
    return result


def _err(reason: str, raw: str = "") -> dict:
    return {"label": "error", "score": 0.0, "reason": reason, "severity": "low"}


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PUBLIC CLASSIFICATION API
# ═════════════════════════════════════════════════════════════════════════════

def classify_with_gemma4(text: str) -> dict:
    """Classify one post locally with Gemma 4. Backend chosen by env var."""
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0,
                "reason": "empty input", "severity": "low"}

    try:
        if GEMMA4_BACKEND == "transformers":
            return _classify_transformers(text)
        return _classify_llama(text)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


def classify_batch_with_gemma4(texts: list, progress_bar=None) -> list:
    """Classify a list of posts; optional Streamlit progress bar."""
    results = []
    total = max(len(texts), 1)
    for i, text in enumerate(texts):
        results.append(classify_with_gemma4(text))
        if progress_bar is not None:
            progress_bar.progress(
                (i + 1) / total, text=f"Gemma 4: {i + 1}/{total}"
            )
    return results


def analyze_image_with_gemma4(image_url: str, tweet_text: str = "") -> dict:
    """
    Vision classification with Gemma 4.

    NOTE: GGUF multimodal support is still limited; vision requires the
    transformers backend. If running under the llama backend, this returns
    an informative error so the caller can degrade gracefully.
    """
    if GEMMA4_BACKEND != "transformers":
        return {
            "has_harmful_content": False,
            "label": "skipped",
            "description": "vision requires GEMMA4_BACKEND=transformers",
            "reason": "current backend is llama.cpp (text only)",
        }

    try:
        import httpx
        import torch
        from PIL import Image
        from io import BytesIO
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # Vision-capable Gemma 4 variant
        vision_repo = os.environ.get(
            "GEMMA_HF_VISION_REPO", "google/gemma-4-e4b-it"
        )
        processor = AutoProcessor.from_pretrained(vision_repo)
        model = AutoModelForImageTextToText.from_pretrained(
            vision_repo,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        img = Image.open(BytesIO(httpx.get(image_url, timeout=15).content))
        prompt = (
            f"Tweet text: \"{tweet_text}\"\n\n"
            "Analyze this image for harassment, threats, hate symbols, or "
            "harmful content. Reply ONLY with JSON:\n"
            "{\"has_harmful_content\": true|false, "
            "\"label\": \"harassment\"|\"neutral\", "
            "\"description\": \"...\", \"reason\": \"...\"}"
        )

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        inputs = processor.apply_chat_template(
            messages, images=[img], return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        raw = processor.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return _parse_json_response(raw)

    except Exception as e:
        return {
            "has_harmful_content": False,
            "label": "error",
            "description": "vision call failed",
            "reason": f"{type(e).__name__}: {e}",
        }


# ═════════════════════════════════════════════════════════════════════════════
# 6.  STREAMLIT UI HELPERS  (drop-in for app.py)
# ═════════════════════════════════════════════════════════════════════════════

def render_kaggle_data_section():
    """Drop-in section to load the Kaggle dataset."""
    st.subheader("📊 Dataset — Emakia Political Tweets (Kaggle)")
    limit = st.slider("Number of tweets to load", 10, 500, 100)

    if st.button("Load Dataset from Kaggle"):
        with st.spinner("Downloading from Kaggle..."):
            try:
                df = load_kaggle_tweets(limit=limit)
                st.success(f"✅ Loaded {len(df)} tweets")
                st.dataframe(df.head(20))
                return df
            except Exception as e:
                st.error(f"❌ {e}")
                st.info(
                    "Set KAGGLE_USERNAME and KAGGLE_API_TOKEN in Streamlit secrets "
                    "(get them at kaggle.com → Settings → API)."
                )
    return None


def render_gemma4_classifier_section(df):
    """Drop-in section to run Gemma 4 classification."""
    backend_label = GEMMA4_BACKEND
    model_label = (
        GEMMA_GGUF_REPO if backend_label == "llama" else GEMMA_HF_REPO
    )

    st.subheader("🤖 Gemma 4 Classification — open weights, on-device")
    st.caption(f"Backend: `{backend_label}` · Model: `{model_label}`")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Test a single post**")
        test_text = st.text_area("Enter post text:", height=100)
        if st.button("Classify with Gemma 4"):
            if not test_text.strip():
                st.warning("Enter some text first.")
            else:
                with st.spinner("Running Gemma 4 locally…"):
                    result = classify_with_gemma4(test_text)
                label = result.get("label", "error")
                score = result.get("score", 0.0)
                color = "🚩" if label == "harassment" else (
                    "✅" if label == "neutral" else "⚠️"
                )
                st.markdown(f"**{color} {label.upper()}** — {score:.0%} confidence")
                st.caption(result.get("reason", ""))

    with col2:
        st.markdown("**Batch classify from dataset**")
        if df is None or (hasattr(df, "empty") and df.empty):
            st.info("Load the dataset first.")
            return
        n = st.number_input("How many posts?", 1, 50, 10)
        if st.button("Run Gemma 4 on batch"):
            texts = df["text"].head(int(n)).tolist()
            bar = st.progress(0.0)
            results = classify_batch_with_gemma4(texts, bar)
            bar.empty()

            result_df = pd.DataFrame({
                "text": texts,
                "label": [r.get("label") for r in results],
                "score": [r.get("score") for r in results],
                "severity": [r.get("severity") for r in results],
                "reason": [r.get("reason") for r in results],
            })
            harassed = (result_df["label"] == "harassment").sum()
            st.metric("🚩 Flagged", f"{harassed}/{len(result_df)}")
            st.dataframe(result_df)
