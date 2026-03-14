import os
import sys
import uuid
import base64
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from google.adk.sessions import InMemorySessionService
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.genai import types

from google.oauth2 import service_account
from google.cloud import bigquery

# Resolve paths — structure is:
#   adk_hackathon/
#     adk_hackathon_streamlit_G.../   ← sibling folder (name found via glob)
#       ingest/   reddit_fetcher.py
#       tools/    gemini_vision_*.py
#     logic/
#       app.py    ← __file__
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../adk_hackathon/logic/
_ROOT = os.path.dirname(_HERE)                        # .../adk_hackathon/

# Search for ingest/ and tools/ inside any sibling folder, and also directly under _ROOT
for _name in ("ingest", "tools"):
    candidates = glob.glob(os.path.join(_ROOT, "*", _name)) + [os.path.join(_ROOT, _name)]
    for _p in candidates:
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

from reddit_fetcher import get_reddit_posts

try:
    from bigquery_fetcher import get_tweets_from_bigquery
except ImportError:
    try:
        from bigquery_loader import get_tweets_from_bigquery
    except ImportError:
        get_tweets_from_bigquery = None

# Use Secret Manager version on Cloud Run (K_SERVICE is set automatically),
# fall back to secrets.toml version for local development.
if os.getenv("K_SERVICE"):
    from gemini_vision_no_key import classify_media
else:
    from gemini_vision_with_key import classify_media

import re
import requests as _requests
import db_dtypes  # Ensures custom BigQuery types are handled

# ── t.co media resolver ───────────────────────────────────────────────────────
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

def _resolve_url(short_url: str, timeout: int = 10) -> str:
    """Follow redirects on a t.co link and return the final URL."""
    try:
        resp = _requests.head(short_url, allow_redirects=True, timeout=timeout,
                              headers={"User-Agent": "Mozilla/5.0"})
        return resp.url
    except Exception:
        return short_url


def _fetch_image_b64(url: str):
    try:
        resp = _requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode()
    except Exception:
        return None


def _download_video(url: str, dest: str):
    try:
        resp = _requests.get(url, timeout=60, stream=True,
                             headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
        return dest
    except Exception:
        return None


# Errors that mean media is gone — skip immediately, don't retry
_YTDLP_SKIP_ERRORS = (
    "no video could be found",
    "video #1 is unavailable",
    "unable to download webpage",
    "404",
    "not found",
    "this tweet",
    "suspended",
    "does not exist",
    "deleted",
)

def _ytdlp_download(url: str, timeout: int = 15) -> dict:
    """
    Use yt-dlp to download video. Skips immediately on known
    unavailable/deleted errors. Hard timeout of 15s per URL.
    """
    try:
        import yt_dlp
    except ImportError:
        return {}

    tmp_path = f"/tmp/ytdlp_{abs(hash(url))}.mp4"
    # First do a fast info-only check — no download, no retries
    # This catches "unavailable" errors in ~1s before wasting time downloading
    info_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "socket_timeout": 8, "retries": 0, "extractor_retries": 0,
        "extractor_args": {"twitter": {"api": ["syndication"]}},
    }
    try:
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            ydl.extract_info(url, download=False)
    except Exception as e:
        err = str(e).lower()
        if any(skip in err for skip in _YTDLP_SKIP_ERRORS):
            print(f"yt-dlp skip (unavailable): {url}")
            return {}
        # Unknown error on info check — still try the full download below

    ydl_opts = {
        "outtmpl": tmp_path,
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": timeout,
        "retries": 0,              # ← zero retries — fail fast
        "extractor_retries": 0,    # ← zero extractor retries — fail fast
        "extractor_args": {"twitter": {"api": ["syndication"]}},
        "ignoreerrors": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        import glob as _glob
        candidates = _glob.glob(f"{tmp_path}*")
        if candidates:
            return {"video_path": candidates[0], "media_url": url, "media_type": "video"}

        thumb = info.get("thumbnail")
        if thumb:
            b64 = _fetch_image_b64(thumb)
            if b64:
                return {"image_b64": b64, "media_url": thumb, "media_type": "image"}

    except Exception as e:
        err = str(e).lower()
        # Check if it's a known "media gone" error — log briefly and skip
        if any(skip in err for skip in _YTDLP_SKIP_ERRORS):
            print(f"yt-dlp skip (unavailable): {url}")
        else:
            print(f"yt-dlp failed for {url}: {e}")

    return {}


def resolve_tco_media(text: str) -> dict:
    """
    Find the first https://t.co/... URL in text, resolve it, then try:
      1. Direct image/video URL
      2. yt-dlp (Twitter/X native video, YouTube, Streamable, etc.)
      3. OG image tag fallback
    """
    urls = re.findall(r"https://t\.co/\S+", text)
    if not urls:
        return {}

    short_url = urls[0]
    final_url = _resolve_url(short_url)
    path_lower = final_url.split("?")[0].lower()

    # 1. Direct image
    if any(path_lower.endswith(ext) for ext in _IMAGE_EXTS):
        b64 = _fetch_image_b64(final_url)
        if b64:
            return {"image_b64": b64, "media_url": final_url, "media_type": "image"}

    # 2. Direct video file
    if any(path_lower.endswith(ext) for ext in _VIDEO_EXTS):
        tmp = f"/tmp/tweet_{abs(hash(final_url))}.mp4"
        saved = _download_video(final_url, tmp)
        if saved:
            return {"video_path": saved, "media_url": final_url, "media_type": "video"}

    # 3. yt-dlp — try both the short t.co URL and the resolved URL
    for attempt_url in [short_url, final_url]:
        result = _ytdlp_download(attempt_url)
        if result:
            return result

    # 4. OG image fallback (article thumbnails)
    try:
        from bs4 import BeautifulSoup
        resp = _requests.get(final_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.content, "html.parser")
        og_image = (
            soup.find("meta", property="og:image")
            or soup.find("meta", attrs={"name": "og:image"})
        )
        if og_image and og_image.get("content"):
            img_url = og_image["content"]
            b64 = _fetch_image_b64(img_url)
            if b64:
                return {"image_b64": b64, "media_url": img_url, "media_type": "image"}
    except Exception:
        pass

    return {"media_url": final_url, "media_type": "link"}

# --- Load environment variables ---
load_dotenv()

# --- BigQuery client (lazy init — works both locally and on Cloud Run) ---
_bq_client = None

def _get_bq_client():
    """
    On Cloud Run: uses Application Default Credentials automatically (no secrets needed).
    Locally: uses service account JSON from st.secrets['bq']['creds'].
    """
    global _bq_client
    if _bq_client is None:
        if os.getenv("K_SERVICE"):
            # Cloud Run — ADC handles auth automatically, no secrets.toml needed
            _bq_client = bigquery.Client(project="emakia")
        else:
            # Local — use service account from secrets.toml
            bq_creds_dict = dict(st.secrets["bq"]["creds"])
            if "\\n" in bq_creds_dict["private_key"]:
                bq_creds_dict["private_key"] = bq_creds_dict["private_key"].replace("\\n", "\n")
            creds = service_account.Credentials.from_service_account_info(bq_creds_dict)
            _bq_client = bigquery.Client(credentials=creds, project=creds.project_id)
    return _bq_client

# --- Constants ---
APP_NAME = "toxicity_misinformation_analysis"
USER_ID = f"user_{uuid.uuid4()}"
SESSION_ID = f"session_{uuid.uuid4()}"
GEMINI_MODEL = "gemini-2.0-flash"
debug_logs = []

# --- Configure Gemini API ---
# On Cloud Run: reads from env var set via gcloud
# Locally: reads from .streamlit/secrets.toml
def _get_google_api_key():
    key = os.getenv("GOOGLE_API_KEY")  # Cloud Run env var
    if key:
        return key
    try:
        return st.secrets["GOOGLE_API_KEY"]  # local secrets.toml
    except Exception:
        return None

GOOGLE_API_KEY = _get_google_api_key()
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY — set it as an env var on Cloud Run or in secrets.toml locally.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# --- Define Agents ---
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


# --- Helper: results to dataframe ---
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
    return df


# --- Core text analysis runner (multi-agent) ---
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
            runner = Runner(
                agent=parallel_agent,
                app_name=APP_NAME,
                session_service=session_service,
            )
            msg = types.Content(role="user", parts=[types.Part(text=item["content"])])

            result = {
                "title": item.get("title", "Untitled"),
                "content": item["content"],
                "toxicity": "No result",
                "bias": "No result",
                "misinformation": "No result",
                "original_label": item.get("label", None),
                "tweet_id": item.get("tweet_id", None),
            }

            async for event in runner.run_async(
                user_id=USER_ID, session_id=session.id, new_message=msg
            ):
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
            # Streamlit can share an event loop — create a new thread to run safely
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run_analysis_async(items))
                return future.result()
        else:
            return loop.run_until_complete(_run_analysis_async(items))
    except RuntimeError:
        return asyncio.run(_run_analysis_async(items))


# --- Helper: render Gemini Vision result ---
def render_vision_result(result: dict):
    """Display classify_media() output in a readable Streamlit layout."""
    flag = result.get("flag")
    confidence = result.get("confidence")
    report = result.get("report")

    # Top-level flag / confidence
    col1, col2 = st.columns(2)
    col1.metric("🚩 Flagged", "Yes" if flag else "No")
    if confidence is not None:
        col2.metric("📊 Confidence", f"{confidence:.0%}")

    # Detailed video report (only present for video input)
    if report:
        st.subheader("📋 Detailed Report")

        toxicity = report.get("toxicity", {})
        misinfo = report.get("misinformation", {})

        col_t, col_m = st.columns(2)
        with col_t:
            st.markdown("**🧪 Toxicity**")
            st.metric("Score", f"{toxicity.get('score', 0)} / 10")
            findings = toxicity.get("findings", [])
            if findings:
                st.write("Findings:", ", ".join(findings))
            timestamps = toxicity.get("timestamps", [])
            if timestamps:
                st.write("Timestamps:", ", ".join(timestamps))

        with col_m:
            st.markdown("**🚫 Misinformation**")
            st.metric("Score", f"{misinfo.get('score', 0)} / 10")
            claims = misinfo.get("claims", [])
            if claims:
                st.write("Claims:", ", ".join(claims))
            timestamps = misinfo.get("timestamps", [])
            if timestamps:
                st.write("Timestamps:", ", ".join(timestamps))

        verdict = report.get("overall_verdict", "—")
        verdict_color = {"SAFE": "🟢", "REVIEW": "🟡", "REMOVE": "🔴"}.get(verdict, "⚪")
        st.markdown(f"**Verdict:** {verdict_color} `{verdict}`")
        st.info(report.get("summary", ""))

    # Image-only result (no nested report)
    else:
        reason = result.get("reason")
        if reason:
            st.write(f"**Reason:** {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧠 Emakia — Multi-Agent Content Moderation")

input_mode = st.radio(
    "Choose Input Mode",
    ["Reddit Posts", "Paste Text", "Upload Image", "Upload Video", "📊 BigQuery Tweets"],
)

# ── Mode 1: Reddit Posts ─────────────────────────────────────────────────────
if input_mode == "Reddit Posts":
    subreddit = st.text_input("Subreddit", value="politics")
    limit = st.slider("Number of posts", 1, 10, 3)

    if st.button("Analyze Reddit Posts"):
        # ── Step 1: fetch posts ───────────────────────────────────────────────
        with st.spinner("Fetching posts from Reddit..."):
            try:
                posts = get_reddit_posts(subreddit, limit=limit)
            except Exception as e:
                st.error(f"❌ Reddit fetch failed: {e}")
                posts = []

        if not posts:
            st.warning("⚠️ No posts returned from Reddit. Check your credentials or subreddit name.")
        else:
            st.info(f"Fetched {len(posts)} posts. Running analysis...")

            # Show a preview of what was fetched before analysis
            with st.expander("📥 Raw posts fetched"):
                for p in posts:
                    st.markdown(f"**{p.get('title', '(no title)')}**")
                    st.caption(p.get("content", "")[:300] + ("…" if len(p.get("content","")) > 300 else ""))
                    if p.get("image_url"):
                        st.image(p["image_url"], width=200)
                    if p.get("video_url"):
                        st.caption(f"🎬 Video: {p['video_url']}")
                    st.write("---")

            # ── Step 2: text analysis via multi-agent ─────────────────────────
            with st.spinner("Running multi-agent text analysis..."):
                results = run_analysis(posts)

            # ── Step 3: Gemini Vision for any media found ─────────────────────
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
                            st.warning(f"Vision error for '{p.get('title', '')}': {e}")

            # ── Step 4: display results ───────────────────────────────────────
            if results:
                st.header("📰 Text Analysis Results")
                for post in results:
                    st.subheader(post["title"])
                    st.write(f"**Content:** {post['content'][:500]}{'…' if len(post['content']) > 500 else ''}")
                    st.markdown(f"🧪 **Toxicity:** `{post['toxicity']}`")
                    st.markdown(f"🎯 **Bias:** `{post['bias']}`")
                    st.markdown(f"🚫 **Misinformation:** `{post['misinformation']}`")
                    st.write("---")
            else:
                st.warning("⚠️ Text analysis returned no results. Check ADK agent logs above.")

            if vision_results:
                st.header("🔬 Gemini Vision Results")
                for vr in vision_results:
                    st.subheader(vr["title"])
                    render_vision_result(vr["result"])
                    st.write("---")

# ── Mode 2: Paste Text ────────────────────────────────────────────────────────
elif input_mode == "Paste Text":
    text_input = st.text_area("Paste your text here (Facebook post, tweet, etc.)")
    if st.button("Analyze Text"):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                results = run_analysis([{"content": text_input, "title": "Manual Input"}])
            for r in results:
                st.markdown(f"🧪 **Toxicity:** `{r['toxicity']}`")
                st.markdown(f"🎯 **Bias:** `{r['bias']}`")
                st.markdown(f"🚫 **Misinformation:** `{r['misinformation']}`")
        else:
            st.warning("Please enter some text first.")

# ── Mode 3: Upload Image ──────────────────────────────────────────────────────
elif input_mode == "Upload Image":
    st.caption("Gemini Vision scans images for harassment, hate speech, or threats.")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

        if st.button("Analyze Image with Gemini Vision"):
            with st.spinner("Calling Gemini Vision..."):
                try:
                    image_b64 = base64.b64encode(uploaded.read()).decode()
                    result = classify_media(image_b64=image_b64)
                    st.success("✅ Analysis complete")
                    render_vision_result(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── Mode 4: Upload Video ──────────────────────────────────────────────────────
elif input_mode == "Upload Video":
    st.caption("Gemini Vision analyzes videos for toxicity and misinformation.")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded:
        st.video(uploaded)

        if st.button("Analyze Video with Gemini Vision"):
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            with st.spinner("Uploading to Gemini & analyzing… (may take up to 30s)"):
                try:
                    result = classify_media(video_path=tmp_path)
                    st.success("✅ Analysis complete")
                    render_vision_result(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── Mode 5: BigQuery Tweets ───────────────────────────────────────────────────
elif input_mode == "📊 BigQuery Tweets":
    if get_tweets_from_bigquery is None:
        st.error(
            "❌ Could not import `get_tweets_from_bigquery`. "
            "Make sure `ingest/bigquery_fetcher.py` (or `bigquery_loader.py`) exists "
            "and is in the `ingest/` folder."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("Number of tweets to analyze", 5, 25, 15)
        with col2:
            filter_label = st.selectbox(
                "Filter by existing label", ["All", "harassment", "neutral"]
            )

        if st.button("Run Batch Analysis on Tweets"):
            with st.spinner("Fetching tweets from BigQuery..."):
                tweets = get_tweets_from_bigquery(
                    limit=limit,
                    filter_label=None if filter_label == "All" else filter_label,
                )

            if not tweets:
                st.warning("❌ No tweets returned from BigQuery.")
            else:
                st.info(f"Fetched {len(tweets)} tweets. Running analysis...")
                progress = st.progress(0)
                results = []
                vision_map = {}  # tweet index → vision result

                for i, tweet in enumerate(tweets):
                    # ── Text analysis ─────────────────────────────────────────
                    batch = run_analysis([tweet])
                    results.extend(batch)

                    # ── Gemini Vision if t.co link detected ───────────────────
                    content_text = tweet.get("content", "")
                    if "https://t.co/" in content_text:
                        try:
                            import concurrent.futures as _cf
                            # Hard 20s wall-clock timeout — yt-dlp can't hang forever
                            with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                                _future = _pool.submit(resolve_tco_media, content_text)
                                try:
                                    media = _future.result(timeout=20)
                                except _cf.TimeoutError:
                                    print(f"resolve_tco_media timed out for tweet {i}")
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
                        elif media.get("media_type") == "link":
                            # URL resolved but no downloadable media — skip Vision silently
                            pass

                    progress.progress((i + 1) / len(tweets))

                if results:
                    st.header("🐦 Tweet Analysis Results")
                    df = results_to_df(results)
                    st.dataframe(df)

                    if "match" in df.columns:
                        agree_count = (df["match"] == "✅").sum()
                        total = len(df)
                        st.subheader("🔍 Original Label vs Ensemble Comparison")
                        col1, col2 = st.columns(2)
                        col1.metric("Agreement Rate", f"{agree_count / total:.1%}")
                        col2.metric("Total Tweets Analyzed", total)

                    with st.expander("See detailed results"):
                        for i, tweet in enumerate(results):
                            st.subheader(tweet.get("title", "Tweet"))
                            st.write(f"**Content:** {tweet['content']}")
                            st.markdown(f"🧪 **Toxicity:** `{tweet['toxicity']}`")
                            st.markdown(f"🎯 **Bias:** `{tweet['bias']}`")
                            st.markdown(f"🚫 **Misinformation:** `{tweet['misinformation']}`")
                            if tweet.get("original_label") is not None:
                                label_text = "harassment" if tweet["original_label"] == 0 else "neutral"
                                st.markdown(f"🏷️ **Original Label:** `{label_text}`")

                            # ── Vision result for this tweet ──────────────────
                            if i in vision_map:
                                vdata = vision_map[i]
                                if "error" in vdata:
                                    st.warning(f"🔬 Vision error: {vdata['error']}")
                                else:
                                    st.markdown(
                                        f"🔬 **Gemini Vision** ({vdata['media_type']}: "
                                        f"[link]({vdata['media_url']}))"
                                    )
                                    render_vision_result(vdata["result"])

                            st.write("---")
                else:
                    st.warning("❌ No results from analysis.")
