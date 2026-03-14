from __future__ import annotations
import os
import praw
import requests
import base64
import streamlit as st
from bs4 import BeautifulSoup

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
VIDEO_DOMAINS = ("v.redd.it", "youtube.com", "youtu.be", "streamable.com")

# --- Reddit client (lazy init — avoids crash at import time on Cloud Run) ---
_reddit = None

def _get_reddit():
    """Initialize Reddit client on first use — reads from st.secrets or env vars."""
    global _reddit
    if _reddit is None:
        def _get(key):
            # Try st.secrets first (local), fall back to env var (Cloud Run)
            try:
                return st.secrets[key]
            except Exception:
                return os.getenv(key)
        _reddit = praw.Reddit(
            client_id=     _get("REDDIT_CLIENT_ID"),
            client_secret= _get("REDDIT_CLIENT_SECRET"),
            user_agent=    _get("REDDIT_USER_AGENT"),
        )
    return _reddit


def get_article_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        candidates = ["article", "main", "div#content", "div.article-body", "div#mainArticleDiv"]
        for selector in candidates:
            block = soup.select_one(selector)
            if block and block.get_text(strip=True):
                return block.get_text(strip=True, separator="\n")

        paragraphs = soup.find_all("p")
        if paragraphs:
            return "\n".join(p.get_text() for p in paragraphs)

        return "No article content found."
    except Exception:
        return f"[link]({url})"


def _fetch_image_as_b64(url: str) -> str | None:
    """Download an image URL and return a base64-encoded string, or None on failure."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        print(f"Image download failed ({url}): {e}")
        return None


def _download_video(url: str, dest_path: str) -> str | None:
    """Download a video to dest_path and return the path, or None on failure."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return dest_path
    except Exception as e:
        print(f"Video download failed ({url}): {e}")
        return None


def _extract_media(post) -> dict:
    """
    Inspect a PRAW post and return a dict with any of:
        image_b64   – base64 image string  (for classify_media)
        video_path  – local /tmp path      (for classify_media)
        image_url   – original image URL   (for display)
        video_url   – original video URL   (for display)
    Returns an empty dict if no media is found.
    """
    media = {}
    url = post.url or ""

    # ── Reddit-hosted image ───────────────────────────────────────────────────
    if post.domain == "i.redd.it" or url.lower().endswith(IMAGE_EXTENSIONS):
        b64 = _fetch_image_as_b64(url)
        if b64:
            media["image_b64"] = b64
            media["image_url"] = url

    # ── Reddit gallery (multiple images) — take the first one ────────────────
    elif hasattr(post, "is_gallery") and post.is_gallery:
        try:
            first_id = next(iter(post.gallery_data["items"]))["media_id"]
            img_url = post.media_metadata[first_id]["p"][-1]["u"]  # highest-res preview
            b64 = _fetch_image_as_b64(img_url)
            if b64:
                media["image_b64"] = b64
                media["image_url"] = img_url
        except Exception as e:
            print(f"Gallery extraction failed: {e}")

    # ── Reddit-hosted video (v.redd.it) ──────────────────────────────────────
    elif post.domain == "v.redd.it":
        try:
            video_url = post.media["reddit_video"]["fallback_url"]
            tmp_path = f"/tmp/{post.id}.mp4"
            saved = _download_video(video_url, tmp_path)
            if saved:
                media["video_path"] = saved
                media["video_url"] = video_url
        except Exception as e:
            print(f"Reddit video extraction failed: {e}")

    # ── External video link (YouTube, Streamable, etc.) — store URL only ─────
    elif any(post.domain.endswith(d) for d in VIDEO_DOMAINS):
        media["video_url"] = url  # can't download; pass URL for display

    # ── Preview image fallback (article posts often have a thumbnail) ─────────
    elif hasattr(post, "preview") and not media:
        try:
            preview_url = post.preview["images"][0]["source"]["url"]
            # Reddit preview URLs use HTML entities — fix them
            preview_url = preview_url.replace("&amp;", "&")
            b64 = _fetch_image_as_b64(preview_url)
            if b64:
                media["image_b64"] = b64
                media["image_url"] = preview_url
        except Exception as e:
            print(f"Preview image extraction failed: {e}")

    return media


def get_reddit_posts(subreddit: str = "politics", limit: int = 3) -> list[dict]:
    """
    Fetch Reddit posts and return a list of dicts with:
        title, content, link
        + any of: image_b64, video_path, image_url, video_url
    """
    posts = []
    try:
        reddit = _get_reddit()
        for post in reddit.subreddit(subreddit).new(limit=limit):
            # --- Text content ---
            content = post.selftext.strip()
            if not content and post.url:
                # Only scrape article text for non-media links
                if not (
                    post.url.lower().endswith(IMAGE_EXTENSIONS)
                    or post.domain in ("i.redd.it", "v.redd.it")
                ):
                    content = get_article_text_from_url(post.url)

            entry = {
                "title": post.title,
                "content": content or f"[link]({post.url})",
                "link": post.url,
            }

            # --- Media content ---
            media = _extract_media(post)
            entry.update(media)

            posts.append(entry)

    except Exception as e:
        print(f"Reddit error: {e}")

    return posts
