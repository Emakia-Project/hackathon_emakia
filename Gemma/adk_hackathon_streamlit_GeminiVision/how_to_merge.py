# HOW TO MERGE INTO YOUR EXISTING app.py
# =========================================
# Make these 3 changes to your existing hackathon app.py:


# ── CHANGE 1: Replace your imports at the top ──────────────────────────────────

# REMOVE these BigQuery imports:
# from google.cloud import bigquery
# from google.oauth2 import service_account

# ADD this instead:
from kaggle_gemma4_integration import (
    load_kaggle_tweets,
    classify_with_gemma4,
    classify_batch_with_gemma4,
    analyze_image_with_gemma4,
    render_kaggle_data_section,
    render_gemma4_classifier_section,
)


# ── CHANGE 2: Replace your BigQuery data load ──────────────────────────────────

# REMOVE code like this:
# client = bigquery.Client(credentials=creds, project=project_id)
# query = "SELECT * FROM emakia.politics2024.NoRetweets-political2024 LIMIT 100"
# df = client.query(query).to_dataframe()

# ADD this instead:
df = load_kaggle_tweets(limit=100)


# ── CHANGE 3: Replace your Gemini model calls ──────────────────────────────────

# REMOVE code like this:
# model = genai.GenerativeModel("gemini-2.0-flash")
# response = model.generate_content(prompt)
# result = response.text

# ADD this instead (returns structured dict, no parsing needed):
result = classify_with_gemma4(tweet_text)
label = result["label"]       # "harassment" or "neutral"
score = result["score"]       # 0.0 - 1.0
reason = result["reason"]     # one sentence explanation

# For images (same as your Gemini Vision calls):
image_result = analyze_image_with_gemma4(image_url, tweet_text)
