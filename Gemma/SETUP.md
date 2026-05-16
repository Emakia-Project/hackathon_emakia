# Gemma 4 Hackathon — Setup Instructions

## 1. Install new dependencies
```bash
pip install kaggle google-genai
```

## 2. Get your Kaggle API key
1. Go to kaggle.com → click your profile photo → Settings
2. Scroll to "API" → click "Create New Token"
3. It downloads `kaggle.json` with your username and key

## 3. Set environment variables
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_API_TOKEN="your_key_from_kaggle_json"
export GOOGLE_API_KEY="your_ai_studio_key"   # same one you already use
```

## 4. Update KAGGLE_DATASET_SLUG in kaggle_gemma4_integration.py
Find this line and update it to your actual Kaggle dataset URL:
```python
KAGGLE_DATASET_SLUG = "corinnedavidemakia/emakia-dataset"
```
Find your slug at kaggle.com/datasets → your dataset → copy the URL path

## 5. Run your app
```bash
streamlit run app.py
```

## File structure after merge
```
adk_hackathon_streamlit_GeminiVision/
├── app.py                          ← your existing file (edit per how_to_merge.py)
├── kaggle_gemma4_integration.py    ← new file (drop in)
├── how_to_merge.py                 ← reference only, not imported
└── requirements.txt                ← add: kaggle, google-genai
```
