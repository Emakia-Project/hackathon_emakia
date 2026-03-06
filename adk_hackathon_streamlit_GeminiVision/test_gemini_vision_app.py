"""
test_gemini_vision_app.py
─────────────────────────
Minimal Streamlit UI to test gemini_vision_no_key.py on Cloud Run.
Upload an image or video → calls classify_media() → shows result.
"""

import base64
import sys
import os

import streamlit as st

# Make tools/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from gemini_vision_no_key import classify_media

st.set_page_config(page_title="Gemini Vision Test", page_icon="🔬")
st.title("🔬 Gemini Vision — Test Interface")
st.caption("Isolated test for `tools/gemini_vision_no_key.py`")

mode = st.radio("What do you want to test?", ["Image", "Video"])

# ── IMAGE ────────────────────────────────────────────────────────────────────
if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)
        if st.button("Run classify_media()"):
            with st.spinner("Calling Gemini Vision..."):
                try:
                    image_b64 = base64.b64encode(uploaded.read()).decode()
                    result = classify_media(image_b64=image_b64)
                    st.success("✅ Got a result!")
                    st.json(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── VIDEO ────────────────────────────────────────────────────────────────────
elif mode == "Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded:
        st.video(uploaded)
        if st.button("Run classify_media()"):
            # Save to a temp file — Gemini needs a file path for video
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            with st.spinner("Uploading to Gemini & analyzing... (may take 30s)"):
                try:
                    result = classify_media(video_path=tmp_path)
                    st.success("✅ Got a result!")
                    st.json(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
