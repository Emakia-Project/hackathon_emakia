import os
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import db_dtypes  # Ensures custom BigQuery types are handled


def get_bigquery_client():
    """
    On Cloud Run: uses Application Default Credentials (no secrets needed).
    Locally: uses service account JSON from st.secrets['bq']['creds'].
    """
    if os.getenv("K_SERVICE"):
        # Cloud Run — ADC handles auth automatically
        return bigquery.Client(project="emakia")
    else:
        # Local — use service account from secrets.toml
        bq_creds_dict = dict(st.secrets["bq"]["creds"])
        if "\\n" in bq_creds_dict["private_key"]:
            bq_creds_dict["private_key"] = bq_creds_dict["private_key"].replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(bq_creds_dict)
        return bigquery.Client(credentials=creds, project=creds.project_id)


def get_tweets_from_bigquery(limit: int = 25, filter_label: str = None) -> list:
    """
    Fetch tweets from BigQuery.
    filter_label: 'harassment' or 'neutral' or None for all
    Returns list of dicts with keys: tweet_id, content, label, title
    """
    client = get_bigquery_client()

    where_clause = ""
    if filter_label == "harassment":
        where_clause = "WHERE label = 0"
    elif filter_label == "neutral":
        where_clause = "WHERE label = 1"

    query = f"""
        SELECT *
        FROM `emakia.politics2024.CoreMLpredictions_tweets_with_media`
        {where_clause}
        LIMIT {limit}
    """

    try:
        rows = client.query(query).result()
        tweets = []
        for row in rows:
            row_dict = dict(row)
            tweets.append({
                "tweet_id": row_dict.get("tweet_id", ""),
                "content":  row_dict.get("text", row_dict.get("content", "")),
                "label":    row_dict.get("label", None),
                "title":    f"Tweet {row_dict.get('tweet_id', '')}"
            })
        return tweets
    except Exception as e:
        print(f"BigQuery error: {e}")
        return []

