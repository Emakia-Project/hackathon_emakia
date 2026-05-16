import os
import sys
import uuid
import json
import time
import base64
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.adk.sessions import InMemorySessionService
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.genai import types as genai_types
# ── Path resolution ──────────────────────────────────────────────────────────
import glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _name in ("ingest", "tools"):
    candidates = glob.glob(os.path.join(_ROOT, "*", _name)) + [os.path.join(_ROOT, _name)]
    for _p in candidates:
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
from reddit_fetcher import get_reddit_posts
# ── Vision import ────────────────────────────────────────────────────────────
if os.getenv("K_SERVICE"):
    from gemini_vision_no_key import classify_media
else:
    from gemini_vision_with_key import classify_media
import re
import requests as _requests
import db_dtypes
# Local Gemma 4 backend removed — Cloud Run cannot support llama.cpp (2 GiB RAM limit).
# All inference goes through Cloud (Google AI Studio · gemma-4-26b).
# Eval harness (sampling, checkpointing, Kaggle publishing)
import eval_pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Unified secret bridging — works for local dev (.env, secrets.toml, ~/.kaggle),
# Streamlit Community Cloud (dashboard), and Cloud Run (Secret Manager).
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
def _bridge_secret(name: str, required: bool = False) -> str | None:
    value = os.environ.get(name)
    if not value:
        try:
            if name in st.secrets:
                value = st.secrets[name]
                os.environ[name] = value
        except Exception:
            value = None
    if required and not value:
        st.error(
            f"❌ Missing {name}. Set it in .streamlit/secrets.toml, "
            f".env, or as an environment variable."
        )
        st.stop()
    return value
GOOGLE_API_KEY  = _bridge_secret("GOOGLE_API_KEY", required=True)
KAGGLE_USERNAME = _bridge_secret("KAGGLE_USERNAME")
KAGGLE_API_TOKEN = _bridge_secret("KAGGLE_API_TOKEN")
genai.configure(api_key=GOOGLE_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────
# Kaggle data loader — registry of public tweet datasets
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_PUBLIC_DATASETS = {
    "Trump & Musk Inauguration (458K tweets)": {
        "slug": "bwandowando/tweets-on-trump-and-musk-potus-2025-inauguration",
        "text_col": "text",
        "label_col": None,
        "label_map": {},
        "id_col": "pseudo_id",
    },
    "Davidson Hate Speech (25K tweets)": {
        "slug": "mrmorj/hate-speech-and-offensive-language-dataset",
        "text_col": "tweet",
        "label_col": "class",
        "label_map": {0: "harassment", 1: "harassment", 2: "neutral"},
        "id_col": None,
    },
    "Hate Speech for Social Media (1.8K posts)": {
        "slug": "ziya07/hate-speech-detection-dataset-for-social-media",
        "text_col": "text",
        "label_col": "label",
        "label_map": {"hateful": "harassment", "offensive": "harassment", "neutral": "neutral"},
        "id_col": "post_id",
    },
    "Custom Kaggle Slug": {
        "slug": None,
        "text_col": None,
        "label_col": None,
        "label_map": {},
        "id_col": None,
    },
}
# ─────────────────────────────────────────────────────────────────────────────
# Encoding-tolerant CSV reader.
#
# Real-world Kaggle datasets — especially older Facebook / Twitter scrapes —
# are NOT always UTF-8. Common offenders:
#   * UTF-16 LE (BOM = 0xff 0xfe at byte 0)         ← "byte 0xff in position 0"
#   * UTF-16 BE (BOM = 0xfe 0xff)
#   * UTF-8 with BOM (Excel exports save this way)
#   * Windows-1252 / Latin-1 (legacy Western dumps)
#
# Order matters:
#   - utf-8 first    → clean files don't get a misleading "decoded as X" notice
#   - latin-1 last   → it accepts ANY byte sequence, so anywhere else in the
#                      list it would mask real encoding mismatches and silently
#                      produce corrupted text for non-Western data.
# ─────────────────────────────────────────────────────────────────────────────
_CSV_ENCODINGS_TO_TRY = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "latin-1",
)
def _read_csv_robust(path: str, **kwargs):
    """
    Drop-in replacement for pd.read_csv that cycles through likely encodings.
    Caller may pass any pd.read_csv kwarg EXCEPT `encoding` (managed here).
    Returns:  (DataFrame, encoding_used)
    Raises :  RuntimeError if no encoding decodes the file.
    """
    kwargs.pop("encoding", None)
    last_err = None
    for enc in _CSV_ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            return df, enc
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as e:
            # Wrong encoding can manifest as either UnicodeDecodeError (raw
            # bytes won't decode) OR ParserError (garbled bytes look like
            # malformed CSV). Try the next encoding either way.
            last_err = e
            continue
    raise RuntimeError(
        f"Could not decode CSV at {path} with any of "
        f"{_CSV_ENCODINGS_TO_TRY}. Last error: {last_err}"
    )
def _sniff_format(path: str):
    """Best-effort format detection for files with missing/unrecognised
    extensions. Returns 'csv'|'tsv'|'json'|'jsonl'|'parquet' or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(2048)
    except Exception:
        return None
    if not head:
        return None
    if head[:4] == b"PAR1":
        return "parquet"
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            text = head.decode("utf-16", errors="replace").lstrip()
        except Exception:
            return None
    else:
        try:
            text = head.decode("utf-8", errors="replace").lstrip()
        except Exception:
            return None
    if not text:
        return None
    if text[0] in "{[":
        non_empty = [l for l in text.splitlines() if l.strip()]
        if len(non_empty) >= 2 and all(
            l.lstrip().startswith("{") for l in non_empty[:3]
        ):
            return "jsonl"
        return "json"
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if first_line.count(",") >= 2:
        return "csv"
    if first_line.count("\t") >= 2:
        return "tsv"
    return None
def _scan_data_files(download_path: str):
    """Return list of (abs_path, format) for every loadable file."""
    fmt_globs = {
        "csv": "*.csv", "json": "*.json", "jsonl": "*.jsonl",
        "tsv": "*.tsv", "txt": "*.txt", "parquet": "*.parquet",
        "xlsx": "*.xlsx", "xls": "*.xls",
    }
    found = []
    matched = set()
    for fmt_name, pattern in fmt_globs.items():
        for path in glob.glob(os.path.join(download_path, "**", pattern), recursive=True):
            if os.path.isfile(path):
                found.append((path, fmt_name))
                matched.add(path)
    for path in glob.glob(os.path.join(download_path, "**", "*"), recursive=True):
        if not os.path.isfile(path) or path in matched:
            continue
        sniffed = _sniff_format(path)
        if sniffed:
            found.append((path, sniffed))
    return found
@st.cache_data(ttl=3600)
def download_and_list_kaggle_files(slug: str) -> list:
    """Download a Kaggle dataset (cached) and return file metadata tuples."""
    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        st.error(
            f"\u274c Kaggle auth failed: {e}\n\n"
            "Set KAGGLE_USERNAME and KAGGLE_API_TOKEN environment variables, "
            "or place a kaggle.json in ~/.kaggle/."
        )
        return []
    safe_name = slug.replace("/", "_")
    download_path = f"/tmp/kaggle_{safe_name}"
    os.makedirs(download_path, exist_ok=True)
    try:
        kaggle.api.dataset_download_files(slug, path=download_path, unzip=True, quiet=True)
    except Exception as e:
        err_str = str(e)
        kaggle_url = f"https://www.kaggle.com/datasets/{slug}"
        if "404" in err_str or "Not Found" in err_str:
            st.error(f"\u2753 **Dataset not found** at `{slug}`. Verify at {kaggle_url}.")
        elif "403" in err_str or "Forbidden" in err_str:
            st.error(
                f"\u26d4 **Access denied for `{slug}`.** Visit {kaggle_url} "
                "and click 'I Understand and Accept' first."
            )
        elif "401" in err_str or "Unauthorized" in err_str:
            st.error("\U0001f511 **Kaggle authentication failed.** Check your credentials.")
        else:
            st.error(f"\u274c Kaggle download failed: {err_str}")
        return []
    all_data_files = _scan_data_files(download_path)
    entries = []
    for abs_path, fmt in all_data_files:
        rel_path = os.path.relpath(abs_path, download_path)
        parts = rel_path.split(os.sep)
        group_label = os.sep.join(parts[:-1]) if len(parts) > 1 else "(root)"
        file_name = parts[-1]
        size_mb = os.path.getsize(abs_path) / 1024 / 1024
        entries.append((rel_path, group_label, file_name, size_mb, abs_path, fmt))
    entries.sort(key=lambda e: (e[1].lower(), -e[3]))
    return entries
def _normalize_kaggle_slug(raw: str) -> str:
    """Normalise any URL/path/slug to canonical 'owner/dataset-name'."""
    s = raw.strip()
    for prefix in ("https://www.kaggle.com/datasets/",
                   "https://kaggle.com/datasets/",
                   "http://www.kaggle.com/datasets/",
                   "http://kaggle.com/datasets/",
                   "[www.kaggle.com/datasets/](https://www.kaggle.com/datasets/)",
                   "kaggle.com/datasets/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.split("?")[0].split("#")[0].strip("/")
    if s.startswith("datasets/"):
        s = s[len("datasets/"):]
    parts = s.split("/")
    if len(parts) > 2:
        s = "/".join(parts[:2])
    return s
# ─────────────────────────────────────────────────────────────────────────────
# Chunked reader for large CSV/TSV/JSONL files.
# ─────────────────────────────────────────────────────────────────────────────
LARGE_FILE_THRESHOLD_MB = 200
def _count_rows_streaming(path: str, sep: str = ",") -> int:
    """Cheap row count: count newlines minus header. Works across UTF-8,
    UTF-16 (LE/BE), Latin-1, CP1252 — in all of them, byte 0x0A appears
    exactly once per newline."""
    total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            total += block.count(b"\n")
    return max(0, total - 1)
def _read_window_chunked(path: str, start_row: int, limit: int, fmt: str):
    """
    Read rows [start_row, start_row+limit) from a possibly huge file.
    Returns:
      4-tuple (window_df, total_rows, full_df_or_None, encoding) for tabular formats
      None for unsupported formats (caller falls back to whole-load)
    """
    size_mb = os.path.getsize(path) / 1024 / 1024
    is_large = size_mb > LARGE_FILE_THRESHOLD_MB
    if fmt in ("csv", "tsv", "txt"):
        sep = "\t" if fmt == "tsv" else (None if fmt == "txt" else ",")
        engine = "python" if sep is None else "c"
        # `low_memory` only applies to engine='c'
        common_kw = dict(sep=sep, engine=engine, on_bad_lines="skip")
        if engine == "c":
            common_kw["low_memory"] = False
        if is_large:
            total = _count_rows_streaming(path, sep=sep or ",")
            skip = range(1, start_row + 1) if start_row > 0 else None
            window, enc = _read_csv_robust(path, skiprows=skip, nrows=limit, **common_kw)
            return window, total, None, enc
        else:
            full, enc = _read_csv_robust(path, **common_kw)
            window = full.iloc[start_row:start_row + limit]
            return window, len(full), full, enc
    if fmt == "jsonl":
        if is_large:
            total = _count_rows_streaming(path)
            total += 1
            window = pd.read_json(path, lines=True)
            window = window.iloc[start_row:start_row + limit]
            return window, total, None, "utf-8"
        else:
            full = pd.read_json(path, lines=True)
            return full.iloc[start_row:start_row + limit], len(full), full, "utf-8"
    if fmt == "parquet":
        full = pd.read_parquet(path)
        return full.iloc[start_row:start_row + limit], len(full), full, "binary"
    return None
@st.cache_data(ttl=3600)
def load_kaggle_tweets(
    dataset_key: str,
    custom_slug: str = "",
    limit: int = 100,
    filter_label: str = None,
    selected_file: str = "",
    start_row: int = 0,
) -> list:
    """Download a public Kaggle dataset and return rows compatible with the
    Emakia multi-agent pipeline. Handles non-UTF-8 CSVs via _read_csv_robust."""
    try:
        import kaggle
        kaggle.api.authenticate()
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        st.error(
            f"\u274c Kaggle auth failed: {e}\n\n"
            "Set KAGGLE_USERNAME and KAGGLE_API_TOKEN environment variables.\n"
            "Get them at kaggle.com -> profile photo -> Settings -> API -> Create New Token"
        )
        return []
    ds_info = KAGGLE_PUBLIC_DATASETS[dataset_key]
    slug = ds_info["slug"] or custom_slug.strip()
    if not slug:
        st.error("\u274c Please enter a Kaggle dataset slug (owner/dataset-name).")
        return []
    slug = _normalize_kaggle_slug(slug)
    text_col = ds_info.get("text_col")
    label_col = ds_info.get("label_col")
    label_map = ds_info.get("label_map") or {}
    id_col = ds_info.get("id_col")
    safe_name = slug.replace("/", "_")
    download_path = f"/tmp/kaggle_{safe_name}"
    os.makedirs(download_path, exist_ok=True)
    already_downloaded = bool(_scan_data_files(download_path))
    if not already_downloaded:
        try:
            with st.spinner(f"\U0001f4e5 Downloading `{slug}` from Kaggle (first time only)..."):
                kaggle.api.dataset_download_files(slug, path=download_path, unzip=True, quiet=True)
        except Exception as e:
            err_str = str(e)
            kaggle_url = f"https://www.kaggle.com/datasets/{slug}"
            if "404" in err_str or "Not Found" in err_str:
                st.error(
                    f"\u2753 **Dataset not found.** No public Kaggle dataset exists at `{slug}`.\n\n"
                    f"- Visit {kaggle_url} to verify it's reachable.\n"
                    f"- The slug should be `owner/dataset-name` only."
                )
            elif "403" in err_str or "Forbidden" in err_str:
                st.error(
                    f"\u26d4 **Access denied for `{slug}`.**\n\n"
                    f"Visit {kaggle_url}, click **'I Understand and Accept'** or "
                    f"download once via the website, then retry."
                )
            elif "401" in err_str or "Unauthorized" in err_str:
                st.error(
                    "\U0001f511 **Kaggle authentication failed.**\n\n"
                    "Verify `~/.kaggle/kaggle.json` (chmod 600) OR "
                    "`KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` env vars are set"
                )
            else:
                st.error(f"\u274c Kaggle download failed: {err_str}")
            return []
    all_data_files = _scan_data_files(download_path)
    if not all_data_files:
        all_files = glob.glob(os.path.join(download_path, "**", "*"), recursive=True)
        files_found = [os.path.relpath(f, download_path) for f in all_files if os.path.isfile(f)]
        st.error(
            f"\u274c No supported data files found in dataset `{slug}`.\n\n"
            f"Looked for: csv, json, jsonl, tsv, txt, parquet, xlsx, xls\n\n"
            f"Files in archive: `{files_found[:20]}`"
        )
        return []
    entries = []
    for abs_path, fmt in all_data_files:
        rel_path = os.path.relpath(abs_path, download_path)
        parts = rel_path.split(os.sep)
        group_label = os.sep.join(parts[:-1]) if len(parts) > 1 else "(root)"
        file_name = parts[-1]
        size_mb = os.path.getsize(abs_path) / 1024 / 1024
        entries.append((rel_path, group_label, file_name, size_mb, abs_path, fmt))
    entries.sort(key=lambda e: (e[1].lower(), -e[3]))
    if selected_file:
        matched = [e for e in entries if e[0] == selected_file]
        if matched:
            data_file, chosen_fmt = matched[0][4], matched[0][5]
        else:
            data_file, chosen_fmt = entries[0][4], entries[0][5]
    else:
        data_file, chosen_fmt = entries[0][4], entries[0][5]
    rel_loaded = os.path.relpath(data_file, download_path)
    file_size_mb = os.path.getsize(data_file) / 1024 / 1024
    is_large = file_size_mb > LARGE_FILE_THRESHOLD_MB
    st.caption(
        f"\U0001f4c2 {chosen_fmt.upper()}: `{rel_loaded}` ({file_size_mb:.1f} MB)"
        + ("  \u2192 streaming (chunk-read just the requested window)" if is_large else "")
    )
    full_df_for_filtering = None
    encoding_used = "utf-8"
    try:
        if chosen_fmt in ("csv", "tsv", "txt", "jsonl", "parquet"):
            with st.spinner(f"\U0001f4d6 Reading rows {start_row}\u2013{start_row + limit - 1}..."):
                result = _read_window_chunked(data_file, start_row, limit, chosen_fmt)
            if result is None:
                st.error(f"\u274c Streaming reader returned nothing for `{rel_loaded}`.")
                return []
            df, total_rows, full_df_for_filtering, encoding_used = result
        elif chosen_fmt in ("xlsx", "xls"):
            with st.spinner("\U0001f4d6 Reading Excel file..."):
                full_df_for_filtering = pd.read_excel(data_file)
            df = full_df_for_filtering.iloc[start_row:start_row + limit]
            total_rows = len(full_df_for_filtering)
            encoding_used = "binary"
        elif chosen_fmt == "json":
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                for key in ("data", "root", "records", "rows", "items", "results"):
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                else:
                    records = [data]
            else:
                records = []
            if not records:
                st.error(f"\u274c JSON file `{rel_loaded}` loaded but contains no records.")
                return []
            full_df_for_filtering = pd.DataFrame(records)
            df = full_df_for_filtering.iloc[start_row:start_row + limit]
            total_rows = len(full_df_for_filtering)
        else:
            st.error(f"\u274c Unsupported file format: {chosen_fmt}")
            return []
    except Exception as e:
        st.error(f"\u274c Failed to read {chosen_fmt.upper()} file `{rel_loaded}`: {e}")
        return []
    # Surface non-default encoding so user knows the file wasn't UTF-8
    if encoding_used not in ("utf-8", "binary"):
        st.caption(
            f"\U0001f4c4 File decoded as `{encoding_used}` "
            "(not UTF-8 — Kaggle uploader saved it in that encoding)."
        )
    if df is None or len(df) == 0:
        if total_rows == 0:
            st.warning(f"\u26a0\ufe0f File `{rel_loaded}` appears to be empty.")
        else:
            st.warning(
                f"\u26a0\ufe0f Start row {start_row} is beyond the available "
                f"{total_rows} rows. Reset start row to 0."
            )
        return []
    st.caption(f"\U0001f4ca Columns: {list(df.columns)}")
    if text_col is None or text_col not in df.columns:
        for cand in ["tweet", "text", "content", "body", "message",
                     "comment_text", "post", "preprocessed_text"]:
            if cand in df.columns:
                text_col = cand
                break
        if text_col is None:
            obj_cols = df.select_dtypes(include="object").columns.tolist()
            text_col = obj_cols[0] if obj_cols else None
        if text_col is None:
            st.error("\u274c Could not auto-detect a text column.")
            return []
        st.caption(f"\U0001f50d Auto-detected text column: '{text_col}'")
    has_labels = False
    if label_col is None or label_col not in df.columns:
        for cand in ["label", "class", "target", "sentiment",
                     "is_toxic", "hate_speech", "category"]:
            if cand in df.columns:
                label_col = cand
                break
    if label_col and label_col in df.columns:
        has_labels = True
        st.caption(f"\U0001f50d Auto-detected label column: '{label_col}'")
    else:
        st.info(
            "\u2139\ufe0f This dataset has **no ground-truth label column** "
            "(message text only). The Gemma 4 multi-agent pipeline will still "
            "classify each row — there's just no reference label to compare "
            "against. For accuracy/F1 scores, use a labeled dataset like "
            "Davidson Hate Speech in the Benchmark Eval mode."
        )
    if id_col is None or id_col not in df.columns:
        for cand in ["id", "tweet_id", "post_id", "comment_id", "pseudo_id", "index"]:
            if cand in df.columns:
                id_col = cand
                break
    if has_labels:
        df = df.copy()
        df["_norm_label"] = (
            df[label_col].map(label_map) if label_map
            else df[label_col].astype(str)
        )
        numeric_label_map = {"harassment": 0, "neutral": 1}
        df["_numeric_label"] = df["_norm_label"].map(numeric_label_map)
    else:
        df = df.copy()
        df["_norm_label"] = None
        df["_numeric_label"] = None
    if filter_label and has_labels:
        if full_df_for_filtering is not None:
            full_df_for_filtering = full_df_for_filtering.copy()
            full_df_for_filtering["_norm_label"] = (
                full_df_for_filtering[label_col].map(label_map) if label_map
                else full_df_for_filtering[label_col].astype(str)
            )
            filtered = full_df_for_filtering[
                full_df_for_filtering["_norm_label"] == filter_label
            ]
            total_rows = len(filtered)
            df = filtered.iloc[start_row:start_row + limit].copy()
            df["_numeric_label"] = df["_norm_label"].map({"harassment": 0, "neutral": 1})
            if len(df) == 0 and total_rows > 0:
                st.warning(
                    f"\u26a0\ufe0f Start row {start_row} beyond filtered total "
                    f"({total_rows}). Reset to 0."
                )
                return []
        else:
            st.info(
                f"\u2139\ufe0f Label filter applied to visible window only "
                f"(file is large — global filter would require loading "
                f"{file_size_mb:.0f} MB into memory)."
            )
            df = df[df["_norm_label"] == filter_label]
    end_idx = start_row + len(df) - 1 if len(df) else start_row
    st.caption(
        f"\U0001f522 Showing rows **{start_row}\u2013{end_idx}** of "
        f"~**{total_rows}** total{' (filtered)' if filter_label else ''}."
    )
    tweets = []
    for i, row in df.iterrows():
        content = str(row.get(text_col, ""))
        tweet_id = str(row.get(id_col, f"row_{i}")) if id_col else f"row_{i}"
        label_numeric = row.get("_numeric_label")
        tweets.append({
            "content": content,
            "title": tweet_id,
            "label": int(label_numeric) if pd.notna(label_numeric) else None,
            "tweet_id": tweet_id,
        })
    return tweets
# ─────────────────────────────────────────────────────────────────────────────
# Gemma 4 classifier (cloud + local backends)
# ─────────────────────────────────────────────────────────────────────────────
GEMMA_MODEL = "gemma-4-26b-a4b-it"
def _classify_with_gemma4_cloud(text: str) -> dict:
    """Classify a single tweet with Gemma 4 via Google AI Studio."""
    from google import genai as _genai
    from google.genai import types as _types
    client = _genai.Client(api_key=GOOGLE_API_KEY)
    prompt = f"""You are a civic content moderation AI built for the Emakia project.
Analyze this social media post and classify it for harassment.
Post: "{text}"
Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "label": "harassment" or "neutral",
  "score": <float 0.0-1.0 representing harassment confidence>,
  "reason": "<one sentence explaining your classification>",
  "severity": "low" or "medium" or "high"
}}"""
    try:
        response = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=prompt,
            config=_types.GenerateContentConfig(temperature=0.1, max_output_tokens=200)
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"label": "error", "score": 0.0,
                "reason": f"Parse error: {response.text[:80]}", "severity": "low"}
    except Exception as e:
        return {"label": "error", "score": 0.0, "reason": str(e), "severity": "low"}
def classify_with_gemma4(text: str) -> dict:
    """Classify a single tweet with Gemma 4 via Google AI Studio (cloud only)."""
    return _classify_with_gemma4_cloud(text)
def render_gemma4_result(result: dict):
    """Render a Gemma 4 classification result in Streamlit."""
    label = result.get("label", "error")
    score = result.get("score", 0.0)
    reason = result.get("reason", "")
    severity = result.get("severity", "")
    icon = {"harassment": "🚩", "neutral": "✅", "error": "❌"}.get(label, "❓")
    st.markdown(f"**Gemma 4:** {icon} `{label.upper()}`  —  {score:.0%} confidence  `{severity}`")
    if reason:
        st.caption(f"↳ {reason}")
# ─────────────────────────────────────────────────────────────────────────────
# Media helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
def _resolve_url(short_url, timeout=10):
    try:
        resp = _requests.head(short_url, allow_redirects=True, timeout=timeout,
                              headers={"User-Agent": "Mozilla/5.0"})
        return resp.url
    except Exception:
        return short_url
def _fetch_image_b64(url):
    try:
        resp = _requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode()
    except Exception:
        return None
def _download_video(url, dest):
    try:
        resp = _requests.get(url, timeout=60, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
        return dest
    except Exception:
        return None
_YTDLP_SKIP_ERRORS = (
    "no video could be found", "video #1 is unavailable", "unable to download webpage",
    "404", "not found", "this tweet", "suspended", "does not exist", "deleted",
)
def _ytdlp_download(url, timeout=15):
    try:
        import yt_dlp
    except ImportError:
        return {}
    tmp_path = f"/tmp/ytdlp_{abs(hash(url))}.mp4"
    info_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "socket_timeout": 8, "retries": 0, "extractor_retries": 0,
        "extractor_args": {"twitter": {"api": ["syndication"]}},
    }
    try:
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            ydl.extract_info(url, download=False)
    except Exception as e:
        if any(s in str(e).lower() for s in _YTDLP_SKIP_ERRORS):
            return {}
    ydl_opts = {
        "outtmpl": tmp_path, "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "socket_timeout": timeout, "retries": 0, "extractor_retries": 0,
        "extractor_args": {"twitter": {"api": ["syndication"]}}, "ignoreerrors": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        candidates = glob.glob(f"{tmp_path}*")
        if candidates:
            return {"video_path": candidates[0], "media_url": url, "media_type": "video"}
        thumb = info.get("thumbnail")
        if thumb:
            b64 = _fetch_image_b64(thumb)
            if b64:
                return {"image_b64": b64, "media_url": thumb, "media_type": "image"}
    except Exception as e:
        if not any(s in str(e).lower() for s in _YTDLP_SKIP_ERRORS):
            print(f"yt-dlp failed for {url}: {e}")
    return {}
def resolve_tco_media(text):
    urls = re.findall(r"https://t\.co/\S+", text)
    if not urls:
        return {}
    short_url = urls[0]
    final_url = _resolve_url(short_url)
    path_lower = final_url.split("?")[0].lower()
    if any(path_lower.endswith(ext) for ext in _IMAGE_EXTS):
        b64 = _fetch_image_b64(final_url)
        if b64:
            return {"image_b64": b64, "media_url": final_url, "media_type": "image"}
    if any(path_lower.endswith(ext) for ext in _VIDEO_EXTS):
        tmp = f"/tmp/tweet_{abs(hash(final_url))}.mp4"
        saved = _download_video(final_url, tmp)
        if saved:
            return {"video_path": saved, "media_url": final_url, "media_type": "video"}
    for attempt_url in [short_url, final_url]:
        result = _ytdlp_download(attempt_url)
        if result:
            return result
    try:
        from bs4 import BeautifulSoup
        resp = _requests.get(final_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.content, "html.parser")
        og_image = (soup.find("meta", property="og:image")
                    or soup.find("meta", attrs={"name": "og:image"}))
        if og_image and og_image.get("content"):
            b64 = _fetch_image_b64(og_image["content"])
            if b64:
                return {"image_b64": b64, "media_url": og_image["content"], "media_type": "image"}
    except Exception:
        pass
    return {"media_url": final_url, "media_type": "link"}
# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
APP_NAME = "toxicity_misinformation_analysis"
USER_ID = f"user_{uuid.uuid4()}"
SESSION_ID = f"session_{uuid.uuid4()}"
GEMINI_MODEL = "gemini-2.0-flash"
GEMMA_MODEL = "gemma-4-26b-a4b-it"
debug_logs = []
# ─────────────────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────────────────
toxicity_agent = LlmAgent(
    name="ToxicityAnalyst",
    model=GEMINI_MODEL,
    instruction="Classify the following statement as 'toxic' or 'non-toxic'.",
    description="Detects toxic language.",
    output_key="toxicity_analysis"
)
bias_agent = LlmAgent(
    name="BiasAnalyst",
    model=GEMINI_MODEL,
    instruction="Classify the statement as 'biased' or 'neutral'. Explain why.",
    description="Assesses bias.",
    output_key="bias_analysis"
)
misinfo_agent = LlmAgent(
    name="MisinformationAnalyst",
    model=GEMINI_MODEL,
    instruction="Determine if this statement contains 'misinformation' or is 'accurate'. Include rationale.",
    description="Detects misinformation.",
    output_key="misinformation_analysis"
)
parallel_agent = ParallelAgent(
    name="ParallelAnalysisAgent",
    sub_agents=[toxicity_agent, bias_agent, misinfo_agent]
)
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def results_to_df(results):
    df = pd.DataFrame(results)
    if "original_label" in df.columns:
        df["original_label_text"] = df["original_label"].map({0: "harassment", 1: "neutral"})
        df["match"] = df.apply(
            lambda r: "✅" if (
                (r["original_label"] == 0 and "toxic" in str(r.get("toxicity", "")).lower()
                 and "non-toxic" not in str(r.get("toxicity", "")).lower())
                or (r["original_label"] == 1 and "non-toxic" in str(r.get("toxicity", "")).lower())
            ) else "❌",
            axis=1,
        )
    if "gemma4_label" in df.columns:
        df["gemma4_match"] = df.apply(
            lambda r: "✅" if (
                (r.get("original_label") == 0 and r.get("gemma4_label") == "harassment")
                or (r.get("original_label") == 1 and r.get("gemma4_label") == "neutral")
            ) else "❌",
            axis=1,
        )
    return df
async def _run_analysis_async(items):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    results = []
    for item in items:
        if "content" not in item:
            debug_logs.append(f"⚠️ Skipping item with missing content: {item}")
            continue
        try:
            runner = Runner(agent=parallel_agent, app_name=APP_NAME, session_service=session_service)
            msg = genai_types.Content(role="user", parts=[genai_types.Part(text=item["content"])])
            result = {
                "title": item.get("title", "Untitled"),
                "content": item["content"],
                "toxicity": "No result",
                "bias": "No result",
                "misinformation": "No result",
                "original_label": item.get("label", None),
                "tweet_id": item.get("tweet_id", None),
            }
            async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=msg):
                if event and event.content and event.content.parts:
                    output = event.content.parts[0].text or ""
                    if "toxic" in output.lower():
                        result["toxicity"] = output
                    elif "bias" in output.lower():
                        result["bias"] = output
                    elif "misinformation" in output.lower() or "accurate" in output.lower():
                        result["misinformation"] = output
            results.append(result)
        except Exception as e:
            debug_logs.append(f"Runner error: {e}")
            st.error(f"🔥 Error during analysis: {e}")
    return results
def run_analysis(items):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run_analysis_async(items))
                return future.result()
        else:
            return loop.run_until_complete(_run_analysis_async(items))
    except RuntimeError:
        return asyncio.run(_run_analysis_async(items))
def render_vision_result(result: dict):
    flag = result.get("flag")
    confidence = result.get("confidence")
    report = result.get("report")
    col1, col2 = st.columns(2)
    col1.metric("🚩 Flagged", "Yes" if flag else "No")
    if confidence is not None:
        col2.metric("📊 Confidence", f"{confidence:.0%}")
    if report:
        st.subheader("📋 Detailed Report")
        toxicity = report.get("toxicity", {})
        misinfo = report.get("misinformation", {})
        col_t, col_m = st.columns(2)
        with col_t:
            st.markdown("**🧪 Toxicity**")
            st.metric("Score", f"{toxicity.get('score', 0)} / 10")
            if toxicity.get("findings"):
                st.write("Findings:", ", ".join(toxicity["findings"]))
            if toxicity.get("timestamps"):
                st.write("Timestamps:", ", ".join(toxicity["timestamps"]))
        with col_m:
            st.markdown("**🚫 Misinformation**")
            st.metric("Score", f"{misinfo.get('score', 0)} / 10")
            if misinfo.get("claims"):
                st.write("Claims:", ", ".join(misinfo["claims"]))
            if misinfo.get("timestamps"):
                st.write("Timestamps:", ", ".join(misinfo["timestamps"]))
        verdict = report.get("overall_verdict", "—")
        verdict_color = {"SAFE": "🟢", "REVIEW": "🟡", "REMOVE": "🔴"}.get(verdict, "⚪")
        st.markdown(f"**Verdict:** {verdict_color} `{verdict}`")
        st.info(report.get("summary", ""))
    else:
        reason = result.get("reason")
        if reason:
            st.write(f"**Reason:** {reason}")
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — cloud-only info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Gemma 4 Backend")
    st.session_state["gemma_backend"] = "cloud"
    st.info("☁️ Cloud (Google AI Studio · gemma-4-26b)")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        Each post is analyzed by **three independent specialist agents** in
        parallel — toxicity, bias, and misinformation — plus a **direct
        Gemma 4 zero-shot classifier** for cross-validation.
        **Agreement metrics** show whether the agents and the zero-shot
        classifier converge. Disagreement is itself a signal: it usually
        means the post is borderline or context-dependent.
        **Label normalization:** the Davidson dataset's `0=hate, 1=offensive`
        both map to `harassment`; `2=neither` maps to `neutral`. Other
        datasets are normalized in the same `KAGGLE_PUBLIC_DATASETS` registry.
        """)
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧠 Emakia — Multi-Agent Content Moderation")
st.caption(f"Powered by Gemma 4 · Backend: **Cloud ({GEMMA_MODEL})**")
input_mode = st.radio(
    "Choose Input Mode",
    ["Reddit Posts", "Paste Text", "Upload Image", "Upload Video",
     "📊 Public Kaggle Tweets",
     "🧪 Benchmark Eval"],
)
# ── Mode 1: Reddit Posts ─────────────────────────────────────────────────────
if input_mode == "Reddit Posts":
    subreddit = st.text_input("Subreddit", value="politics")
    limit = st.slider("Number of posts", 1, 10, 3)
    if st.button("Analyze Reddit Posts"):
        with st.spinner("Fetching posts from Reddit..."):
            try:
                posts = get_reddit_posts(subreddit, limit=limit)
            except Exception as e:
                st.error(f"❌ Reddit fetch failed: {e}")
                posts = []
        if not posts:
            st.warning("⚠️ No posts returned from Reddit.")
        else:
            st.info(f"Fetched {len(posts)} posts. Running analysis...")
            with st.expander("📥 Raw posts fetched"):
                for p in posts:
                    st.markdown(f"**{p.get('title', '(no title)')}**")
                    st.caption(p.get("content", "")[:300])
                    if p.get("image_url"):
                        st.image(p["image_url"], width=200)
                    st.write("---")
            with st.spinner("Running Gemma 4 multi-agent analysis..."):
                results = run_analysis(posts)
            vision_results = []
            media_posts = [p for p in posts if p.get("image_b64") or p.get("video_path")]
            if media_posts:
                with st.spinner(f"Running Gemini Vision on {len(media_posts)} media item(s)..."):
                    for p in media_posts:
                        try:
                            if p.get("image_b64"):
                                v = classify_media(image_b64=p["image_b64"])
                            else:
                                v = classify_media(video_path=p["video_path"])
                            vision_results.append({"title": p.get("title", "Media"), "result": v})
                        except Exception as e:
                            st.warning(f"Vision error: {e}")
            if results:
                st.header("📰 Text Analysis Results")
                for post in results:
                    st.subheader(post["title"])
                    st.write(post["content"][:500])
                    st.markdown(f"🧪 **Toxicity:** `{post['toxicity']}`")
                    st.markdown(f"🎯 **Bias:** `{post['bias']}`")
                    st.markdown(f"🚫 **Misinformation:** `{post['misinformation']}`")
                    st.write("---")
            if vision_results:
                st.header("🔬 Vision Results")
                for vr in vision_results:
                    st.subheader(vr["title"])
                    render_vision_result(vr["result"])
                    st.write("---")
# ── Mode 2: Paste Text ───────────────────────────────────────────────────────
elif input_mode == "Paste Text":
    text_input = st.text_area("Paste your text here (Facebook post, tweet, etc.)")
    if st.button("Analyze Text"):
        if text_input.strip():
            with st.spinner("Analyzing with Gemma 4..."):
                results = run_analysis([{"content": text_input, "title": "Manual Input"}])
                gemma_result = classify_with_gemma4(text_input)
            for r in results:
                st.markdown(f"🧪 **Toxicity (agent):** `{r['toxicity']}`")
                st.markdown(f"🎯 **Bias (agent):** `{r['bias']}`")
                st.markdown(f"🚫 **Misinformation (agent):** `{r['misinformation']}`")
            st.divider()
            st.markdown("### Gemma 4 Direct Classifier")
            render_gemma4_result(gemma_result)
        else:
            st.warning("Please enter some text first.")
# ── Mode 3: Upload Image ─────────────────────────────────────────────────────
elif input_mode == "Upload Image":
    st.caption("Gemini Vision scans images for harassment, hate speech, or threats.")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze Image with Gemini Vision"):
            with st.spinner("Calling Gemini Vision..."):
                try:
                    image_b64 = base64.b64encode(uploaded.read()).decode()
                    result = classify_media(image_b64=image_b64)
                    st.success("✅ Analysis complete")
                    render_vision_result(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
# ── Mode 4: Upload Video ─────────────────────────────────────────────────────
elif input_mode == "Upload Video":
    st.caption("Gemini Vision analyzes videos for toxicity and misinformation.")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded:
        st.video(uploaded)
        if st.button("Analyze Video with Gemini Vision"):
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            with st.spinner("Uploading to Gemini & analyzing…"):
                try:
                    result = classify_media(video_path=tmp_path)
                    st.success("✅ Analysis complete")
                    render_vision_result(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
# ── Mode 5: Public Kaggle Tweets ─────────────────────────────────────────────
elif input_mode == "\U0001f4ca Public Kaggle Tweets":
    st.markdown("""
    **Public tweet / social-media datasets from Kaggle.**
    Choose a pre-configured dataset or enter any Kaggle slug.
    Classified by the Gemma 4 multi-agent pipeline.
    """)
    dataset_key = st.selectbox(
        "Select a public dataset",
        list(KAGGLE_PUBLIC_DATASETS.keys()),
        index=0,
    )
    custom_slug = ""
    if dataset_key == "Custom Kaggle Slug":
        custom_slug = st.text_input(
            "Enter Kaggle dataset slug (owner/dataset-name) — full URLs also accepted",
            placeholder="e.g. mrmorj/hate-speech-and-offensive-language-dataset",
            help=(
                "Paste either:\n"
                "- The bare slug: `owner/dataset-name`, OR\n"
                "- The full URL: `https://www.kaggle.com/datasets/owner/dataset-name`\n\n"
                "Both formats are auto-normalized."
            ),
        )
        if custom_slug.strip():
            _preview_slug = _normalize_kaggle_slug(custom_slug)
            preview_url = f"https://www.kaggle.com/datasets/{_preview_slug}"
            st.caption(
                f"\U0001f50d Preview: [{preview_url}]({preview_url})  "
                "← click to verify the dataset exists on Kaggle"
            )
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        limit = st.slider("Number of posts per page", 3, 50, 5, key="kaggle_limit")
    with col2:
        start_row = st.number_input(
            "Start at row",
            min_value=0,
            value=0,
            step=limit,
            key="kaggle_start_row",
            help=(
                "Row offset into the dataset. Use the up/down arrows or type a "
                "number, then click Load again to see the new window."
            ),
        )
    with col3:
        _preset_has_labels = bool(
            KAGGLE_PUBLIC_DATASETS[dataset_key].get("label_col")
            and KAGGLE_PUBLIC_DATASETS[dataset_key].get("label_map")
        )
        if _preset_has_labels:
            filter_label = st.selectbox(
                "Filter by normalised label", ["All", "harassment", "neutral"]
            )
        else:
            filter_label = "All"
            st.markdown("&nbsp;")
            st.caption("_No ground-truth labels — Gemma will classify each row from scratch._")
    analyze_media = st.checkbox(
        "🔬 Also resolve & analyze attached media (slower; uses Gemini Vision)",
        value=False,
        help="Off by default — t.co URL resolution can take up to 20s per post."
    )
    _ds_info = KAGGLE_PUBLIC_DATASETS[dataset_key]
    _resolved_slug = _ds_info["slug"] or _normalize_kaggle_slug(custom_slug)
    selected_file = ""
    if _resolved_slug:
        with st.spinner(f"\U0001f4e5 Inspecting Kaggle dataset `{_resolved_slug}`..."):
            entries = download_and_list_kaggle_files(_resolved_slug)
        if entries:
            if len(entries) > 1:
                options_idx = list(range(len(entries)))
                options_labels = []
                for rel_path, group_label, file_name, size_mb, _, fmt in entries:
                    label = (
                        f"\U0001f4c1 {group_label}/   "
                        f"\U0001f4c4 {file_name}  ({size_mb:.1f} MB · {fmt.upper()})"
                    )
                    options_labels.append(label)
                unique_groups = sorted({e[1] for e in entries})
                fmt_counts = {}
                for e in entries:
                    fmt_counts[e[5]] = fmt_counts.get(e[5], 0) + 1
                fmt_summary = [f"{n} {fmt.upper()}" for fmt, n in sorted(fmt_counts.items())]
                top_folders = sorted({g.split(os.sep)[0] for g in unique_groups})[:8]
                st.info(
                    f"\U0001f4c2 Found **{' + '.join(fmt_summary)} files** "
                    f"across **{len(unique_groups)} location(s)**.  "
                    f"Top folders: `{', '.join(top_folders)}`"
                )
                chosen = st.selectbox(
                    "Choose which data file to load",
                    options=options_idx,
                    format_func=lambda i: options_labels[i],
                    index=0,
                    key="kaggle_data_picker",
                )
                selected_file = entries[chosen][0]
            else:
                selected_file = entries[0][0]
                st.caption(
                    f"\U0001f4c2 Single data file in this dataset: "
                    f"`{selected_file}` "
                    f"({entries[0][3]:.1f} MB, {entries[0][5].upper()})"
                )
    if st.button("\U0001f680 Load from Kaggle & Analyze with Gemma 4"):
        with st.spinner("\U0001f4e5 Loading data..."):
            _canonical_custom = _normalize_kaggle_slug(custom_slug) if custom_slug else ""
            tweets = load_kaggle_tweets(
                dataset_key=dataset_key,
                custom_slug=_canonical_custom,
                limit=limit,
                filter_label=None if filter_label == "All" else filter_label,
                selected_file=selected_file,
                start_row=int(start_row),
            )
        if not tweets:
            st.warning("\u274c No posts loaded. Check Kaggle credentials and dataset slug.")
            st.info(
                "Make sure these env vars are set:\n"
                "  export KAGGLE_USERNAME=your_username\n"
                "  export KAGGLE_API_TOKEN=your_key\n\n"
                "For custom datasets, enter the slug as owner/dataset-name"
            )
        else:
            st.success(f"\u2705 Loaded {len(tweets)} posts from Kaggle")
            progress = st.progress(0)
            results = []
            vision_map = {}
            gemma4_map = {}
            for i, tweet in enumerate(tweets):
                batch = run_analysis([tweet])
                results.extend(batch)
                gemma4_result = classify_with_gemma4(tweet["content"])
                gemma4_map[i] = gemma4_result
                content_text = tweet.get("content", "")
                if analyze_media and "https://t.co/" in content_text:
                    try:
                        import concurrent.futures as _cf
                        with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                            _future = _pool.submit(resolve_tco_media, content_text)
                            try:
                                media = _future.result(timeout=20)
                            except _cf.TimeoutError:
                                media = {}
                    except Exception:
                        media = {}
                    if media.get("image_b64") or media.get("video_path"):
                        try:
                            if media.get("image_b64"):
                                v_result = classify_media(image_b64=media["image_b64"])
                            else:
                                v_result = classify_media(video_path=media["video_path"])
                            vision_map[i] = {
                                "result": v_result,
                                "media_url": media.get("media_url", ""),
                                "media_type": media.get("media_type", ""),
                            }
                        except Exception as ve:
                            vision_map[i] = {"error": str(ve)}
                progress.progress((i + 1) / len(tweets))
            if results:
                for i, r in enumerate(results):
                    if i in gemma4_map:
                        r["gemma4_label"] = gemma4_map[i].get("label")
                        r["gemma4_score"] = gemma4_map[i].get("score")
                st.header("\U0001f426 Tweet Analysis Results")
                df = results_to_df(results)
                col_a, col_b, col_c = st.columns(3)
                if "match" in df.columns:
                    agree = (df["match"] == "\u2705").sum()
                    col_a.metric("Agent Agreement", f"{agree / len(df):.1%}")
                if "gemma4_match" in df.columns:
                    g4_agree = (df["gemma4_match"] == "\u2705").sum()
                    col_b.metric("Gemma 4 Agreement", f"{g4_agree / len(df):.1%}")
                flagged = sum(1 for v in gemma4_map.values() if v.get("label") == "harassment")
                col_c.metric("\U0001f6a9 Flagged by Gemma 4", f"{flagged}/{len(tweets)}")
                # Surface error count: silent classification failures look
                # exactly like neutral results in the metrics row otherwise.
                err_count = sum(1 for v in gemma4_map.values() if v.get("label") == "error")
                if err_count:
                    st.warning(
                        f"\u26a0\ufe0f {err_count}/{len(tweets)} rows returned "
                        f"`error` from Gemma 4. Expand 'See detailed results' "
                        f"to inspect each failure's reason."
                    )
                st.dataframe(df)
                with st.expander("See detailed results per tweet"):
                    for i, tweet in enumerate(results):
                        st.subheader(tweet.get("title", "Tweet"))
                        st.write(f"**Content:** {tweet['content']}")
                        st.markdown(f"\U0001f9ea **Toxicity:** `{tweet['toxicity']}`")
                        st.markdown(f"\U0001f3af **Bias:** `{tweet['bias']}`")
                        st.markdown(f"\U0001f6ab **Misinformation:** `{tweet['misinformation']}`")
                        if i in gemma4_map:
                            st.markdown("---")
                            render_gemma4_result(gemma4_map[i])
                        if tweet.get("original_label") is not None:
                            label_text = "harassment" if tweet["original_label"] == 0 else "neutral"
                            st.markdown(f"\U0001f3f7\ufe0f **Original Label:** `{label_text}`")
                        if i in vision_map:
                            vdata = vision_map[i]
                            if "error" in vdata:
                                st.warning(f"\U0001f52c Vision error: {vdata['error']}")
                            else:
                                st.markdown(
                                    f"\U0001f52c **Gemini Vision** ({vdata['media_type']}: "
                                    f"[link]({vdata['media_url']}))"
                                )
                                render_vision_result(vdata["result"])
                        st.write("---")
# ── Mode 6: Benchmark Eval ───────────────────────────────────────────────────
elif input_mode == "🧪 Benchmark Eval":
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    st.markdown("""
    **Run a stratified evaluation across the multi-agent and direct-Gemma 4 pipelines**
    on a labeled dataset, then publish the result CSV to your Kaggle account
    as a public dataset.
    > 🚧 **Validation phase — capped at 50 rows.** Verify the full path runs
    > end-to-end locally **and** on Cloud Run (run → checkpoint → publish to
    > Kaggle) before scaling up. To raise the cap, bump `max_value` on the
    > **Sample size** input below from `50` → `25000`.
    The harness checkpoints every 10 rows — close the tab and resume later
    with the same Run ID. **Note:** local-disk checkpoints don't survive
    Cloud Run restarts; for production use, mount a persistent volume.
    """)
    tab_run, tab_results, tab_publish = st.tabs([
        "▶️ Run / Resume",
        "📊 Results & Metrics",
        "☁️ Publish to Kaggle",
    ])
    with tab_run:
        existing = eval_pipeline.list_runs()
        resume_run_id = None
        if existing:
            with st.expander(f"🔄 Resume an existing run ({len(existing)} found)"):
                for m in existing[:10]:
                    rid = m.get("run_id", "?")
                    n_done = m.get("n_completed", 0)
                    n_tot = m.get("n_total", 0)
                    last = m.get("last_updated", "?")
                    pct = (n_done / n_tot * 100) if n_tot else 0
                    cols = st.columns([4, 2, 2, 2])
                    cols[0].markdown(f"**`{rid}`** · {last}")
                    cols[1].progress(pct / 100, text=f"{n_done}/{n_tot}")
                    if cols[2].button("Resume", key=f"resume_{rid}"):
                        st.session_state["benchmark_run_id"] = rid
                        st.session_state["benchmark_resume"] = True
                        st.rerun()
                    if cols[3].button("Delete", key=f"del_{rid}"):
                        eval_pipeline.delete_run(rid)
                        st.rerun()
        st.subheader("New run configuration")
        labeled_keys = [
            k for k, v in KAGGLE_PUBLIC_DATASETS.items()
            if v.get("label_col") and v.get("label_map")
        ]
        col_l, col_r = st.columns([2, 1])
        with col_l:
            src_key = st.selectbox(
                "Source dataset (must have ground-truth labels)",
                labeled_keys,
                index=labeled_keys.index("Davidson Hate Speech (25K tweets)")
                       if "Davidson Hate Speech (25K tweets)" in labeled_keys else 0,
            )
        with col_r:
            n_rows = st.number_input(
                "Sample size", min_value=10, max_value=50,
                value=20, step=10,
                help="Capped at 50 during validation. Raise max_value in app.py "
                     "after Cloud Run is verified.",
            )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            stratify = st.checkbox("Stratified sampling", value=True)
        with col_b:
            seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        with col_c:
            default_rid = f"run_{_dt.utcnow().strftime('%Y%m%d_%H%M')}"
            run_id = st.text_input(
                "Run ID (used for checkpointing)",
                value=st.session_state.get("benchmark_run_id", default_rid),
                help="Reuse the same ID to resume after interruption.",
            )
        st.markdown("**Combinations to evaluate:**")
        col_cb1, col_cb2 = st.columns(2)
        with col_cb1:
            do_multi = st.checkbox(
                "🤖 Multi-agent (cloud)", value=True,
                help="ADK ParallelAgent: toxicity + bias + misinformation",
            )
        with col_cb2:
            do_direct_cloud = st.checkbox(
                "☁️ Direct Gemma 4 (cloud)", value=True,
                help="Single-shot classifier via Google AI Studio",
            )
        combos = []
        if do_multi:        combos.append("multiagent_cloud")
        if do_direct_cloud: combos.append("direct_cloud")
        sec_per_row = 0
        if "direct_cloud" in combos:     sec_per_row += 5
        if "multiagent_cloud" in combos: sec_per_row += 10
        eta_min = (n_rows * sec_per_row) / 60
        st.caption(
            f"⏱️ Rough estimate: **{eta_min:.0f} min** "
            f"({sec_per_row}s/row × {n_rows} rows). Real time depends on "
            f"network + rate limits."
        )
        if "benchmark_stop" not in st.session_state:
            st.session_state["benchmark_stop"] = False
        col_run, col_stop = st.columns([3, 1])
        run_clicked = col_run.button(
            "🚀 Start / Resume Benchmark",
            type="primary",
            disabled=not combos,
        )
        if col_stop.button("⏸ Pause"):
            st.session_state["benchmark_stop"] = True
            st.warning("Pause requested — current row will finish, then checkpoint.")
        if st.session_state.pop("benchmark_resume", False):
            run_clicked = True
        if run_clicked and combos:
            st.session_state["benchmark_stop"] = False
            ds_info = KAGGLE_PUBLIC_DATASETS[src_key]
            with st.spinner(f"📥 Downloading source `{ds_info['slug']}` from Kaggle..."):
                entries = download_and_list_kaggle_files(ds_info["slug"])
            if not entries:
                st.error("Could not list files in source dataset.")
                st.stop()
            csv_entries = [e for e in entries if e[5] == "csv"]
            if not csv_entries:
                st.error(f"No CSV files found in source `{ds_info['slug']}`.")
                st.stop()
            src_csv = max(csv_entries, key=lambda e: e[3])[4]
            st.caption(f"Using source file: `{os.path.basename(src_csv)}`")
            # Use encoding-tolerant reader here too — labeled datasets like
            # Davidson are usually UTF-8, but don't crash on a UTF-16 source.
            try:
                src_df, src_enc = _read_csv_robust(src_csv, on_bad_lines="skip", low_memory=False)
                if src_enc != "utf-8":
                    st.caption(f"📄 Source decoded as `{src_enc}`.")
            except Exception as e:
                st.error(f"Failed to read source CSV: {e}")
                st.stop()
            cp_completed, cp_sample, cp_config = eval_pipeline.load_checkpoint(run_id)
            if cp_sample is not None:
                st.info(
                    f"♻️ Resuming run `{run_id}` — "
                    f"{len(cp_completed)}/{len(cp_sample)} rows already done."
                )
                sample_df = cp_sample
                combos = cp_config.get("combinations", combos) or combos
                resume_from = cp_completed
            else:
                with st.spinner(f"🎯 Sampling {n_rows} rows (stratify={stratify})..."):
                    try:
                        sample_df = eval_pipeline.prepare_eval_sample(
                            src_df,
                            text_col=ds_info["text_col"],
                            label_col=ds_info["label_col"],
                            label_map=ds_info["label_map"],
                            n_rows=int(n_rows),
                            stratify=stratify,
                            seed=int(seed),
                        )
                    except Exception as e:
                        st.error(f"Sampling failed: {e}")
                        st.stop()
                resume_from = []
            balance = sample_df["ground_truth"].value_counts().to_dict()
            st.caption(f"📊 Class balance: {balance}")
            progress_bar = st.progress(0, text="Starting...")
            metrics_slot = st.empty()
            t_start = time.time()
            def _progress(done, total, last_row):
                pct = done / total
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                progress_bar.progress(
                    pct,
                    text=f"{done}/{total}  ·  {rate:.2f} rows/s  ·  ETA {remaining/60:.1f} min",
                )
            def _stop():
                return st.session_state.get("benchmark_stop", False)
            config = {
                "source_dataset": ds_info["slug"],
                "source_file":    os.path.basename(src_csv),
                "n_rows":         int(n_rows),
                "stratify":       bool(stratify),
                "seed":           int(seed),
                "combinations":   combos,
                "started_at":     _dt.utcnow().isoformat() + "Z",
            }
            try:
                final = eval_pipeline.run_eval(
                    sample_df=sample_df,
                    run_id=run_id,
                    combinations=combos,
                    classify_direct=classify_with_gemma4,
                    run_multiagent=run_analysis,
                    set_backend=lambda b: st.session_state.update(gemma_backend=b),
                    config=config,
                    progress_cb=_progress,
                    resume_from=resume_from,
                    stop_flag=_stop,
                )
            except Exception as e:
                st.error(f"Eval crashed: {e}")
                st.info(f"Partial results saved. Resume with Run ID `{run_id}`.")
                st.stop()
            done_n = len(final)
            total_n = len(sample_df)
            if done_n < total_n:
                st.warning(
                    f"⏸ Paused at {done_n}/{total_n}. "
                    f"Click Start again with the same Run ID to resume."
                )
            else:
                st.success(f"✅ Complete: {done_n}/{total_n} rows evaluated.")
                st.session_state["benchmark_run_id"] = run_id
                st.balloons()
    with tab_results:
        all_runs = eval_pipeline.list_runs()
        if not all_runs:
            st.info("No completed or in-flight runs yet. Start one in the Run tab.")
        else:
            run_choices = {
                f"{m['run_id']}  ({m.get('n_completed',0)}/{m.get('n_total',0)})": m["run_id"]
                for m in all_runs
            }
            picked_label = st.selectbox(
                "Select a run to inspect",
                list(run_choices.keys()),
                index=0,
            )
            picked_id = run_choices[picked_label]
            completed, sample, config = eval_pipeline.load_checkpoint(picked_id)
            if not completed:
                st.warning("Checkpoint is empty.")
            else:
                df = pd.DataFrame(completed)
                combos_used = config.get("combinations", [
                    c for c in eval_pipeline.KNOWN_COMBINATIONS
                    if f"{c}_label" in df.columns
                ])
                metrics_df = eval_pipeline.compute_metrics(df, combos_used)
                if not metrics_df.empty:
                    st.subheader("Per-combination metrics (harassment class)")
                    st.dataframe(
                        metrics_df.set_index("combination"),
                        use_container_width=True,
                    )
                if len(combos_used) >= 2:
                    st.subheader("Pairwise label agreement")
                    agree_rows = []
                    for i, a in enumerate(combos_used):
                        for b in combos_used[i+1:]:
                            ca, cb = f"{a}_label", f"{b}_label"
                            if ca in df.columns and cb in df.columns:
                                both = df[df[ca].isin(["harassment","neutral"])
                                          & df[cb].isin(["harassment","neutral"])]
                                if len(both):
                                    agree = (both[ca] == both[cb]).mean()
                                    agree_rows.append({
                                        "pair": f"{a} ↔ {b}",
                                        "n":    len(both),
                                        "agreement": round(float(agree), 4),
                                    })
                    if agree_rows:
                        st.dataframe(pd.DataFrame(agree_rows), use_container_width=True)
                st.subheader("Row-level results")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    show_only = st.selectbox(
                        "Filter",
                        ["All rows",
                         "Only disagreements between combinations",
                         "Only errors",
                         "Only ground-truth = harassment",
                         "Only ground-truth = neutral"],
                    )
                with col_f2:
                    n_show = st.number_input(
                        "Rows to display", min_value=10, max_value=2000,
                        value=100, step=10,
                    )
                filtered = df.copy()
                label_cols = [f"{c}_label" for c in combos_used
                              if f"{c}_label" in filtered.columns]
                if show_only == "Only disagreements between combinations" and len(label_cols) >= 2:
                    filtered = filtered[filtered[label_cols].nunique(axis=1) > 1]
                elif show_only == "Only errors":
                    mask = filtered[label_cols].isin(["error"]).any(axis=1) \
                           if label_cols else False
                    filtered = filtered[mask] if isinstance(mask, pd.Series) else filtered.iloc[0:0]
                elif show_only == "Only ground-truth = harassment":
                    filtered = filtered[filtered["ground_truth"] == "harassment"]
                elif show_only == "Only ground-truth = neutral":
                    filtered = filtered[filtered["ground_truth"] == "neutral"]
                st.caption(f"Showing {min(n_show, len(filtered))} of {len(filtered)} rows")
                st.dataframe(filtered.head(int(n_show)), use_container_width=True)
                col_d1, col_d2 = st.columns(2)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                col_d1.download_button(
                    "⬇️ Download results CSV",
                    data=csv_bytes,
                    file_name=f"eval_{picked_id}_results.csv",
                    mime="text/csv",
                )
                if not metrics_df.empty:
                    metrics_bytes = metrics_df.to_csv(index=False).encode("utf-8")
                    col_d2.download_button(
                        "⬇️ Download metrics CSV",
                        data=metrics_bytes,
                        file_name=f"eval_{picked_id}_metrics.csv",
                        mime="text/csv",
                    )
                st.session_state["benchmark_publish_run_id"] = picked_id
    with tab_publish:
        publish_id = st.session_state.get("benchmark_publish_run_id")
        if not publish_id:
            st.info(
                "Pick a run in the **Results & Metrics** tab first. "
                "It will be available here for publishing."
            )
        else:
            completed, sample, config = eval_pipeline.load_checkpoint(publish_id)
            if not completed:
                st.warning(f"Run `{publish_id}` has no results to publish.")
            else:
                df_pub = pd.DataFrame(completed)
                metrics_pub = eval_pipeline.compute_metrics(
                    df_pub,
                    config.get("combinations") or [
                        c for c in eval_pipeline.KNOWN_COMBINATIONS
                        if f"{c}_label" in df_pub.columns
                    ],
                )
                st.subheader(f"Publishing run: `{publish_id}`")
                st.caption(
                    f"{len(df_pub)} rows · "
                    f"source: `{config.get('source_dataset', 'unknown')}` · "
                    f"combinations: {', '.join(config.get('combinations', []))}"
                )
                me = os.environ.get("KAGGLE_USERNAME", "").strip()
                if not me:
                    st.error(
                        "❌ KAGGLE_USERNAME is not set in this environment. "
                        "Configure it in `.streamlit/secrets.toml`, `.env`, or "
                        "Cloud Run Secret Manager before publishing."
                    )
                else:
                    st.caption(f"Authenticated as Kaggle user: `{me}`")
                    default_slug = f"{me}/emakia-benchmark-{publish_id.lower()}"
                    slug = st.text_input(
                        "Target Kaggle dataset slug",
                        value=default_slug,
                        help="Format: <your-username>/<dataset-name>. "
                             "Owner must match KAGGLE_USERNAME.",
                    )
                    title = st.text_input(
                        "Dataset title (≤ 50 chars)",
                        value=f"Emakia Benchmark {publish_id}"[:50],
                    )
                    default_desc = (
                        f"# Emakia content moderation benchmark\n\n"
                        f"- **Run ID:** `{publish_id}`\n"
                        f"- **Source:** `{config.get('source_dataset','?')}`\n"
                        f"- **Sample size:** {config.get('n_rows','?')}\n"
                        f"- **Stratified:** {config.get('stratify','?')}\n"
                        f"- **Seed:** {config.get('seed','?')}\n"
                        f"- **Combinations:** {', '.join(config.get('combinations', []))}\n"
                        f"- **Started:** {config.get('started_at','?')}\n\n"
                        f"## Files\n"
                        f"- `results.csv` — per-row predictions for every combination\n"
                        f"- `metrics.csv` — accuracy / precision / recall / F1 per combination\n"
                        f"- `config.json` — exact run configuration for reproducibility\n"
                    )
                    description = st.text_area(
                        "Description (markdown)",
                        value=default_desc,
                        height=250,
                    )
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        new_version = st.checkbox(
                            "Update existing dataset (new version)",
                            value=False,
                            help="Tick this if the slug already exists on Kaggle.",
                        )
                    with col_p2:
                        license_name = st.selectbox(
                            "License",
                            ["CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "MIT"],
                            index=0,
                        )
                    version_notes = ""
                    if new_version:
                        version_notes = st.text_input(
                            "Version notes",
                            value=f"Update from run {publish_id}",
                        )
                    if st.button("☁️ Publish to Kaggle", type="primary"):
                        staging = _Path("/tmp/emakia_eval") / f"_stage_{publish_id}"
                        staging.mkdir(parents=True, exist_ok=True)
                        results_path = staging / "results.csv"
                        metrics_path = staging / "metrics.csv"
                        config_path = staging / "config.json"
                        df_pub.to_csv(results_path, index=False)
                        metrics_pub.to_csv(metrics_path, index=False)
                        config_path.write_text(json.dumps(config, indent=2))
                        with st.spinner("Uploading to Kaggle..."):
                            ok, msg = eval_pipeline.publish_to_kaggle(
                                files=[
                                    (results_path, "results.csv"),
                                    (metrics_path, "metrics.csv"),
                                    (config_path,  "config.json"),
                                ],
                                slug=slug,
                                title=title,
                                description=description,
                                license_name=license_name,
                                new_version=new_version,
                                version_notes=version_notes or "Updated results",
                            )
                        if ok:
                            st.success(f"✅ Published! [{msg}]({msg})")
                        else:
                            st.error(msg)
