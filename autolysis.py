# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "numpy"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# HARD-CODED API KEY (NOT RECOMMENDED)
openai.api_key = "YOUR_API_KEY_HERE"  # Replace with your actual key for testing only

MODEL_NAME = "gpt-4o-mini"

def load_and_clean_data(dataset_path):
    # Try UTF-8 first
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to latin1 if UTF-8 fails
        df = pd.read_csv(dataset_path, encoding="latin1")
    
    missing_values = df.isnull().sum()
    df = df.fillna(0)
    df_cleaned = df.select_dtypes(include=[np.number])
    return df, df_cleaned, missing_values

def analyze_dataset(dataset_path):
    df, df_cleaned, missing_values = load_and_clean_data(dataset_path)
    summary = df.describe(include='all')

    if len(df_cleaned.columns) > 1:
        correlation_matrix = df_cleaned.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig("correlation_matrix.png")
        plt.close()
    else:
        correlation_matrix = None

    with open("README.md", "w", encoding="utf-8") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary.to_string())
        f.write("\n\n## Missing Values\n")
        f.write(missing_values.to_string())

    return df, summary, missing_values

def generate_ai_summary(df):
    prompt = (
        "Analyze the following dataset and summarize key insights:\n\n"
        + df.describe(include='all').to_string()
    )
    messages = [
        {"role": "system", "content": "You are a data analysis assistant."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        ai_summary = response.choices[0].message.content.strip()
        return ai_summary
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]

    df, summary, missing_values = analyze_dataset(dataset_path)
    ai_summary = generate_ai_summary(df)
    if ai_summary:
        with open("ai_summary.txt", "w", encoding="utf-8") as f:
            f.write(ai_summary)

    print("Analysis complete. README.md and any charts have been created in the current directory.")

if __name__ == "__main__":
    main()
