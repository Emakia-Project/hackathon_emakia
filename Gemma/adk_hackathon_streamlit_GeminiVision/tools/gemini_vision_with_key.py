"""
gemini_vision_with_key.py
Loads GOOGLE_API_KEY from .streamlit/secrets.toml.
"""

import os
import json
import time
import base64
import toml

from google import genai
from google.genai import types


def _find_streamlit_secrets() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(current, ".streamlit", "secrets.toml")
        if os.path.exists(candidate):
            print(f"✅ secrets.toml found: {candidate}")
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("❌ .streamlit/secrets.toml not found in any parent folder.")
        current = parent


def _parse_gemini_json(text: str) -> dict:
    """Safely parse Gemini response — strips markdown fences if present."""
    if not text or not text.strip():
        raise ValueError(
            "❌ Gemini returned an empty response.\n"
            "   This usually means the video was blocked by safety filters,\n"
            "   the file format is unsupported, or processing timed out.\n"
            "   Try a shorter clip or a different video."
        )
    # Strip markdown code fences if Gemini wrapped the JSON
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first line (```json or ```) and last line (```)
        cleaned = "\n".join(lines[1:-1]).strip()
    print(f"📄 Raw Gemini response:\n{cleaned}\n")
    return json.loads(cleaned)


secrets = toml.load(_find_streamlit_secrets())
_client = genai.Client(api_key=secrets["GOOGLE_API_KEY"])

VIDEO_PROMPT = """Analyze this video and return ONLY valid JSON with no markdown fences:
{
  "toxicity": {
    "score": 0,
    "findings": ["list of toxic content found"],
    "timestamps": ["mm:ss of toxic moments"]
  },
  "misinformation": {
    "score": 0,
    "claims": ["list of false/misleading claims detected"],
    "timestamps": ["mm:ss of misinformation moments"]
  },
  "overall_verdict": "SAFE or REVIEW or REMOVE",
  "summary": "brief explanation"
}
Score range is 0-10. Return ONLY the JSON object. No markdown. No backticks. No explanation."""

IMAGE_PROMPT = (
    "Assess if this image depicts harassment, hate speech, or threatening content. "
    'Return ONLY valid JSON, no markdown: { "flag": true or false, "confidence": 0.0-1.0, "reason": "string" }'
)


def classify_media(image_b64: str = None, video_path: str = None) -> dict:
    """Classify an image or video for harmful content using Gemini Vision."""
    if video_path:
        print(f"Uploading video: {video_path}")
        uploaded = _client.files.upload(file=video_path)

        # Poll until processing complete
        print("Waiting for Gemini to process video", end="", flush=True)
        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = _client.files.get(name=uploaded.name)
            print(".", end="", flush=True)
        print(f" done! State: {uploaded.state.name}")

        # If processing failed, bail early with clear message
        if uploaded.state.name != "ACTIVE":
            raise ValueError(
                f"❌ Video processing failed. State: {uploaded.state.name}\n"
                "   Check that sea.mp4 is a supported format (mp4, mov, avi, mkv)."
            )

        response = _client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[uploaded, VIDEO_PROMPT]
        )
        report = _parse_gemini_json(response.text)
        flag = report["toxicity"]["score"] >= 6 or report["misinformation"]["score"] >= 6
        return {"flag": flag, "confidence": report["toxicity"]["score"] / 10, "report": report}

    elif image_b64:
        response = _client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"),
                IMAGE_PROMPT
            ]
        )
        return _parse_gemini_json(response.text)

    else:
        raise ValueError("Provide either image_b64 or video_path.")


if __name__ == "__main__":
    # Test image
    print("\n=== Testing image ===")
    with open("Amsterdam.jpeg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    result = classify_media(image_b64=b64)
    print(json.dumps(result, indent=2))

    # Test video
    print("\n=== Testing video ===")
    result = classify_media(video_path="sea.mp4")
    print(json.dumps(result, indent=2))
