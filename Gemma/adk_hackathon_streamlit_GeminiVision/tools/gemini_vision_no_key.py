"""
gemini_vision_no_key.py
Gemini Vision tool — API key fetched from GCP Secret Manager.
Use this version for Cloud Run deployment. No credentials in code.

Requires:
    pip install google-genai google-cloud-secret-manager google-adk
    Cloud Run service account must have roles/secretmanager.secretAccessor
"""

import json
import time
import base64

from google import genai
from google.genai import types
from google.cloud import secretmanager


def _get_secret(secret_id: str, project_id: str = "emakia") -> str:
    """Fetch the latest version of a secret from GCP Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")


def _parse_gemini_json(text: str) -> dict:
    """Safely parse Gemini response — strips markdown fences if present."""
    if not text or not text.strip():
        raise ValueError(
            "Gemini returned an empty response. "
            "The video may have been blocked by safety filters or the format is unsupported."
        )
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    return json.loads(cleaned)


# ── Configure Gemini once at module load using Secret Manager ────────────────
_client = genai.Client(api_key=_get_secret("GOOGLE_API_KEY"))

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
        uploaded = _client.files.upload(file=video_path)

        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = _client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise ValueError(f"Video processing failed. State: {uploaded.state.name}")

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
