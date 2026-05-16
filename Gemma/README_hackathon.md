# AI Scan — Multimodal Toxicity & Misinformation Agent

> **Gemini Live Agent Challenge Submission**  
> Category: **Live Agents 🗣️**  
> Built with Google ADK · Gemini Vision · Cloud Run · BigQuery · Vertex AI

[![Live Demo](https://img.shields.io/badge/Live%20App-Cloud%20Run-blue)](https://emakia-app-634576265618.us-central1.run.app)
[![Repo](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Emakia-Project/hackathon_emakia/tree/main/adk_hackathon_streamlit_GeminiVision)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🧠 What It Does

**AI Scan** is a multimodal content safety agent that analyzes social media posts — text, images, and videos — across three dimensions simultaneously:

| Dimension | Label | How |
|---|---|---|
| 🧪 Toxicity | toxic / non-toxic | ADK LlmAgent + fine-tuned Gemini |
| 🎯 Bias | biased / neutral | ADK LlmAgent + fine-tuned Gemini |
| 🚫 Misinformation | misinformation / accurate | ADK LlmAgent + fine-tuned Gemini |

A **Google ADK `ParallelAgent`** runs all three agents simultaneously — cutting latency ~3× vs sequential chaining — and aggregates their verdicts into a consensus result. **Gemini Vision** then analyzes any images or videos embedded in posts, adding a fourth modality that text-only tools completely miss.

---

## 🏗️ Architecture

```
INPUT SOURCES
├── Reddit Live Feed (PRAW)     — subreddit posts, images, videos, article text
├── BigQuery Tweets (78K rows)  — labeled dataset with t.co media links
├── Paste Text                  — manual input, any text content
└── Upload Media                — jpg/png images, mp4/mov videos

          ↓ INGEST

MEDIA RESOLUTION LAYER
├── t.co Resolver       — resolves redirect chain, detects image/video, yt-dlp fallback
├── Image Fetcher       — download + base64 encode, OG image fallback, Reddit gallery
└── Video Downloader    — Twitter native video, YouTube/Streamable, saves to /tmp/*.mp4

          ↓ ROUTE

ANALYSIS ENGINE — Google ADK + Gemini
├── TEXT TRACK — Parallel Agent (ADK)
│   ├── Toxicity Agent   → toxic / non-toxic
│   ├── Bias Agent       → biased / neutral
│   └── Misinfo Agent    → misinformation / accurate
│   └── Model: gemini-2.0-flash (fine-tuned on Vertex AI)
│
└── VISION TRACK — Gemini Vision
    └── classify_media() → toxicity score 0-10, misinformation score,
                           timestamps of flags, SAFE / REVIEW / REMOVE verdict
        Model: gemini-2.0-flash Vision

          ↓ RESULTS

OUTPUT — Streamlit UI
├── Text Results    — toxicity label + detail, bias classification, misinfo verdict
├── Vision Results  — flag + confidence %, score breakdown, verdict badge
└── BigQuery Table  — agreement rate %, original vs AI label, confusion analysis
```

---

## ☁️ Google Cloud Infrastructure

| Service | Role | Status |
|---|---|---|
| **Cloud Run** (us-central1) | Auto-scaling HTTPS containers | 🟢 LIVE |
| **BigQuery** (US) | `social_content` dataset — 78k+ labeled posts | 🟢 Active |
| **Secret Manager** | `GOOGLE_API_KEY` injected at runtime | 🔒 Secure |
| **Vertex AI Tuning** | Supervised fine-tune of `gemini-2.5-flash` on 78k posts | ✅ Complete |

No secrets in code. Application Default Credentials handle BigQuery auth on Cloud Run automatically.

---

## 🚀 Reproducible Setup

### Prerequisites

- Python 3.11+
- Google Cloud project with these APIs enabled:
  - Cloud Run
  - BigQuery
  - Vertex AI
  - Secret Manager
- `gcloud auth application-default login`

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/Emakia-Project/hackathon_emakia
cd hackathon_emakia/adk_hackathon_streamlit_GeminiVision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key for local dev
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_key_here

# 4. Run the app
streamlit run app.py
```

### Cloud Run Deployment

```bash
gcloud run deploy ai-scan \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest
```

### Testing the Agent

**Test 1 — Reddit live feed:**
1. Select the **Reddit** tab
2. Enter a subreddit name (e.g. `worldnews`)
3. Click **Fetch Data**
4. Watch the parallel agents classify each post in real time

**Test 2 — Paste text:**
1. Select **Paste Text** mode
2. Paste any social media post
3. Click **Analyze Text**
4. See toxicity, bias, and misinformation scores

**Test 3 — Gemini Vision on an image:**
1. Select **Upload Image** mode
2. Upload any jpg or png
3. Click **Analyze Image with Gemini Vision**
4. See SAFE / REVIEW / REMOVE verdict with confidence scores

**Test 4 — BigQuery batch:**
1. Select **📊 BigQuery Tweets** mode
2. Set limit to **10**
3. Click **Run Batch Analysis**
4. See agreement rate between original human labels and AI ensemble

---

## 📂 Project Structure

```
adk_hackathon_streamlit_GeminiVision/
├── app.py                      # Main Streamlit app — 5 analysis modes
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition for Cloud Run
├── .dockerignore
├── cloudbuild.yaml             # Cloud Build config
├── test_gemini_vision_app.py   # Integration tests
│
├── tools/
│   ├── gemini_vision_no_key.py  # Vision classifier — uses ADC (Cloud Run)
│   └── gemini_vision_with_key.py # Vision classifier — uses st.secrets (local)
│
├── ingest/
│   ├── reddit_fetcher.py        # PRAW Reddit data ingestion
│   ├── bigquery_fetcher.py      # BigQuery tweet loader
│   └── bigquery_loader.py      # Fallback loader
│
├── agent/                      # ADK agent definitions
├── compare/                    # Model comparison utilities
└── emakia-diagram/             # Architecture diagrams
```

---

## 🔑 Key Technical Decisions

**Why ADK `ParallelAgent`?**  
Running three independent LLM agents (toxicity, bias, misinfo) in parallel rather than sequentially cuts total latency by ~3× with no accuracy trade-off, since each dimension is independent.

**Why Gemini Vision + yt-dlp?**  
t.co links in tweets often hide the most harmful content — images and videos that text analysis completely misses. A 4-step fallback chain (redirect → direct file → yt-dlp → OG image scraping) handles the full variety of real-world media hosting.

**Why Application Default Credentials?**  
On Cloud Run, ADC gives seamless BigQuery access with zero key management. `Secret Manager` injects the Google API key at runtime — no secrets ever appear in code or container images.

**Why Vertex AI fine-tuning?**  
A general-purpose Gemini prompt performs worse on domain-specific harassment detection than a model fine-tuned on 78k labeled real-world posts. The fine-tuned `gemini-2.5-flash` understands coded language and context that generic prompts miss.

---

## 📊 Dataset

- **Source:** Labeled social media harassment dataset
- **Size:** 78,000+ posts
- **Storage:** Google BigQuery (`social_content` dataset)
- **Public dataset:** [Kaggle — Emakia Dataset](https://www.kaggle.com/datasets/corinnedavidemakia/emakia-dataset)

---

## 🔗 Links

| Resource | URL |
|---|---|
| Live App | https://emakia-app-634576265618.us-central1.run.app |
| GitHub Repo | https://github.com/Emakia-Project/hackathon_emakia/tree/main/adk_hackathon_streamlit_GeminiVision |
| GCP Proof (ADC + Secret Manager code) | https://github.com/Emakia-Project/hackathon_emakia/blob/main/adk_hackathon_streamlit_GeminiVision/tools/gemini_vision_no_key.py |
| Kaggle Dataset | https://www.kaggle.com/datasets/corinnedavidemakia/emakia-dataset |

---

## 👥 Team

**Corinne David** — Lead Developer  
Software engineer, Google Cloud / Vertex AI, ADK pipeline, BigQuery integration, Gemini Vision

**Krupa Karekar** — Co-Developer  
Dataset harmonization, model evaluation, graph analysis

**Magen Daniela Teasley** — Academic Collaborator  
Montclair State University NLP Lab — evaluation metrics, bias auditing

---

*Created for the [Gemini Live Agent Challenge](https://geminiliveagentchallenge.devpost.com/) — #GeminiLiveAgentChallenge*
