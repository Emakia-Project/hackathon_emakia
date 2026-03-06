import os
import uuid
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

from inputs.reddit_scraper import get_reddit_posts
from inputs.maxnews_scraper import get_maxnews_articles
from inputs.bigquery_loader import get_tweets_from_bigquery

import db_dtypes  # Ensures custom BigQuery types are handled

# --- Load environment variables ---
load_dotenv()

# --- BigQuery credentials (loaded ONCE) ---
bq_creds_dict = dict(st.secrets["bq"]["creds"])
if "\\n" in bq_creds_dict["private_key"]:
    bq_creds_dict["private_key"] = bq_creds_dict["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(bq_creds_dict)
client = bigquery.Client(credentials=creds, project=creds.project_id)

# --- Constants ---
APP_NAME = "toxicity_misinformation_analysis"
USER_ID = f"user_{uuid.uuid4()}"
SESSION_ID = f"session_{uuid.uuid4()}"
GEMINI_MODEL = "gemini-2.0-flash"
debug_logs = []

# --- Configure Gemini API ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.error("Missing GOOGLE_API_KEY in environment.")
    st.stop()

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
    if 'original_label' in df.columns:
        df['original_label_text'] = df['original_label'].map({0: 'harassment', 1: 'neutral'})
        # Compare original label vs toxicity result
        df['match'] = df.apply(
            lambda r: '✅' if (
                (r['original_label'] == 0 and 'toxic' in str(r.get('toxicity', '')).lower() and 'non-toxic' not in str(r.get('toxicity', '')).lower()) or
                (r['original_label'] == 1 and 'non-toxic' in str(r.get('toxicity', '')).lower())
            ) else '❌', axis=1
        )
    return df

# --- Core analysis runner ---
def run_analysis(items):
    session_service = InMemorySessionService()
    session = session_service.create_session(
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
                session_service=session_service
            )
            content = types.Content(role="user", parts=[types.Part(text=item["content"])])
            events = runner.run(user_id=USER_ID, session_id=session.id, new_message=content)

            result = {
                "title": item.get("title", "Untitled"),
                "content": item["content"],
                "toxicity": "No result",
                "bias": "No result",
                "misinformation": "No result",
                "original_label": item.get("label", None),
                "tweet_id": item.get("tweet_id", None)
            }

            for event in events:
                if event and event.content.parts:
                    output = event.content.parts[0].text
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

# --- Streamlit UI ---
st.title("🧠 Emakia — Multi-Agent Content Moderation")

# --- Input Mode Selector ---
input_mode = st.radio("Choose Input Mode", [
    "Reddit Posts",
    "Paste Text",
    "Upload Screenshot",
    "📊 BigQuery Tweets"
])

# --- Mode 1: Reddit + Maxnews ---
if input_mode == "Reddit Posts":
    if st.button("Analyze Reddit & Maxnews Posts"):
        with st.spinner("Fetching and analyzing posts..."):
            posts = get_reddit_posts("politics", limit=3)
            posts += get_maxnews_articles()
            results = run_analysis(posts)

        if results:
            st.header("📰 Analysis Results")
            for post in results:
                st.subheader(post["title"])
                st.write(f"**Content:** {post['content']}")
                st.markdown(f"🧪 **Toxicity:** `{post['toxicity']}`")
                st.markdown(f"🎯 **Bias:** `{post['bias']}`")
                st.markdown(f"🚫 **Misinformation:** `{post['misinformation']}`")
                st.write("---")
        else:
            st.warning("❌ No articles available for analysis.")

# --- Mode 2: Paste Text ---
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

# --- Mode 3: Upload Screenshot ---
elif input_mode == "Upload Screenshot":
    uploaded_file = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file)
        st.info("🔜 Image analysis via Gemini Vision coming soon.")

# --- Mode 4: BigQuery Tweets ---
elif input_mode == "📊 BigQuery Tweets":
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Number of tweets to analyze", 10, 500, 100)
    with col2:
        filter_label = st.selectbox(
            "Filter by existing label", ["All", "harassment", "neutral"]
        )

    if st.button("Run Batch Analysis on Tweets"):
        with st.spinner("Fetching tweets from BigQuery..."):
            tweets = get_tweets_from_bigquery(
                limit=limit,
                filter_label=None if filter_label == "All" else filter_label
            )

        if not tweets:
            st.warning("❌ No tweets returned from BigQuery.")
        else:
            st.info(f"Fetched {len(tweets)} tweets. Running analysis...")
            progress = st.progress(0)
            results = []

            for i, tweet in enumerate(tweets):
                batch = run_analysis([tweet])
                results.extend(batch)
                progress.progress((i + 1) / len(tweets))

            if results:
                st.header("🐦 Tweet Analysis Results")

                # Comparison table
                df = results_to_df(results)
                st.dataframe(df)

                # Agreement rate
                if 'match' in df.columns:
                    agree_count = (df['match'] == '✅').sum()
                    total = len(df)
                    st.subheader("🔍 Original Label vs Ensemble Comparison")
                    col1, col2 = st.columns(2)
                    col1.metric("Agreement Rate", f"{agree_count/total:.1%}")
                    col2.metric("Total Tweets Analyzed", total)

                # Detailed expandable view
                with st.expander("See detailed results"):
                    for tweet in results:
                        st.subheader(tweet.get("title", "Tweet"))
                        st.write(f"**Content:** {tweet['content']}")
                        st.markdown(f"🧪 **Toxicity:** `{tweet['toxicity']}`")
                        st.markdown(f"🎯 **Bias:** `{tweet['bias']}`")
                        st.markdown(f"🚫 **Misinformation:** `{tweet['misinformation']}`")
                        if tweet.get('original_label') is not None:
                            label_text = "harassment" if tweet['original_label'] == 0 else "neutral"
                            st.markdown(f"🏷️ **Original Label:** `{label_text}`")
                        st.write("---")
            else:
                st.warning("❌ No results from analysis.")
