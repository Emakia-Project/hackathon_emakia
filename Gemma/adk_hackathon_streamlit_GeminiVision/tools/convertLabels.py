import pandas as pd
import os

# CSV files are in the same folder as this script
DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

file_mapping = {
    'tweets-labels-emojisllm0.csv': 'tweets-labels-vertexAI-llm0.csv',
    'tweets-coreml-trainingllm3.csv': 'tweets-vertexAI-trainingllm3.csv',
    'tweets-coreml-trainingllm4.csv': 'tweets-vertexAI-trainingllm4.csv'
}

for old_name, new_name in file_mapping.items():
    df = pd.read_csv(DATA_PATH + old_name)
    
    # Print actual column names so we can see them
    print(f"Columns in {old_name}: {df.columns.tolist()}")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Find label column regardless of capitalization
    label_col = [c for c in df.columns if c.lower() == 'label'][0]
    
    # Convert - handles any capitalization like Neutral, HARASSMENT etc
    df[label_col] = df[label_col].str.strip().str.lower().map({
        'harassment': 0,
        'neutral': 1
    })
    
    df.to_csv(DATA_PATH + new_name, index=False)
    print(f"✅ Done: {old_name} → {new_name}")
    print(f"   Labels: {df[label_col].value_counts().to_dict()}\n")