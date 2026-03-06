import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import db_dtypes  # Ensures custom BigQuery types are handled


def get_bigquery_client():
    bq_creds_dict = dict(st.secrets["bq"]["creds"])
    if "\\n" in bq_creds_dict["private_key"]:
        bq_creds_dict["private_key"] = bq_creds_dict["private_key"].replace("\\n", "\n")
    creds = service_account.Credentials.from_service_account_info(bq_creds_dict)
    return bigquery.Client(credentials=creds, project=creds.project_id)


def get_tweets_from_bigquery(limit: int = 100, filter_label: str = None) -> list:
    """
    Fetch tweets from BigQuery.
    filter_label: 'harassment' or 'neutral' or None for all
    Returns list of dicts with keys: tweet_id, content, label, title
    """
    client = get_bigquery_client()

    # Build WHERE clause if filtering
    where_clause = ""
    if filter_label == "harassment":
        where_clause = "WHERE label = 0"
    elif filter_label == "neutral":
        where_clause = "WHERE label = 1"

    query = f"""
        SELECT *
        FROM `emakia.tweets.tweets`
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
                "content": row_dict.get("text", row_dict.get("content", "")),
                "label": row_dict.get("label", None),
                "title": f"Tweet {row_dict.get('tweet_id', '')}"
            })
        return tweets
    except Exception as e:
        print(f"BigQuery error: {e}")
        return []
