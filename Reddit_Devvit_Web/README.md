
# 🎥 YouTube Content Analyzer (Reddit Devvit Edition)

Built for the **Reddit Community Games Challenge 2025**, this interactive app analyzes YouTube videos for **toxicity, bias, and misinformation** — helping Reddit communities engage with content more critically and emotionally intelligently.

## 🚀 What It Does

- Accepts a YouTube URL or video ID
- Fetches transcript and metadata
- Scores toxicity (0–10) and flags severity
- Detects bias types and emotional impact
- Visualizes results with color-coded feedback
- Built with **React + Vite**, deployed via **Devvit Web** to run inside Reddit posts

## 🧠 Why It Matters

This app supports Emakia’s mission to build ethical AI tools for civic resilience. It empowers Reddit users to:
- Evaluate video content before sharing
- Spot manipulation and bias in real time
- Engage in emotionally aware discussions

## 🛠️ Tech Stack

- **Frontend**: React + Vite + TailwindCSS
- **Icons**: Lucide React
- **Devvit Web**: Reddit’s embedded app platform
- **API**: Custom toxicity and bias scoring backend (Node/Python)
- **Deployment**: Devvit CLI + Reddit Interactive Post

## 🧩 Devvit Integration

This app is embedded in a Reddit post using Devvit Web:

- Uses `devvit.config.ts` to define entry point and subreddit
- Deployed via `devvit deploy`
- Interactive post created in a test subreddit
- Fully functional inside Reddit — no external hosting required

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/emakia/youtube-analyzer-devvit

# Install dependencies
npm install

# Run locally
npm run dev

# Deploy to Reddit
devvit login
devvit deploy
