# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "numpy",
#   "requests",
#   "scipy",
#   "scikit-learn"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json
import openai

# HARD-CODED API KEY (NOT RECOMMENDED)
openai.api_key = "YOUR_API_KEY_HERE"  # Replace with your actual key for testing only
MODEL_NAME = "gpt-4o-mini"

def load_and_clean_data(dataset_path):
    # Try reading with UTF-8 first; fallback to latin1 if needed
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="latin1")

    missing_values = df.isnull().sum()
    df = df.fillna(0)
    df_cleaned = df.select_dtypes(include=[np.number])
    return df, df_cleaned, missing_values

def detect_outliers(df):
    # Detect outliers using IQR method
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        return pd.Series(dtype=int)
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
    return outliers

def visualize_data(corr_matrix, outliers, df, output_dir):
    # Correlation heatmap
    if corr_matrix is not None and not corr_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        heatmap_file = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(heatmap_file)
        plt.close()
    else:
        heatmap_file = None

    # Outliers plot (if any)
    outliers_file = None
    if not outliers.empty and outliers.sum() > 0:
        plt.figure(figsize=(10, 6))
        outliers.plot(kind='bar', color='red')
        plt.title('Outliers Detection')
        plt.xlabel('Columns')
        plt.ylabel('Number of Outliers')
        outliers_file = os.path.join(output_dir, 'outliers.png')
        plt.savefig(outliers_file)
        plt.close()

    # Distribution plot of the first numeric column
    dist_plot_file = None
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        first_numeric_column = numeric_columns[0]
        plt.figure(figsize=(10, 6))
        sns.histplot(df[first_numeric_column], kde=True, color='blue', bins=30)
        plt.title('Distribution')
        dist_plot_file = os.path.join(output_dir, 'distribution_.png')
        plt.savefig(dist_plot_file)
        plt.close()

    return heatmap_file, outliers_file, dist_plot_file

def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir):
    readme_file = os.path.join(output_dir, 'README.md')
    try:
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# Automated Data Analysis Report\n\n")
            f.write("## Introduction\n")
            f.write("This is an automated analysis of the dataset, providing summary statistics, visualizations, and insights from the data.\n\n")

            # Summary Statistics
            f.write("## Summary Statistics\n")
            f.write("The summary statistics of the dataset are as follows:\n")
            f.write("\n| Statistic    | Value |\n")
            f.write("|--------------|-------|\n")
            for column in summary_stats.columns:
                f.write(f"| {column} - Mean | {summary_stats.loc['mean', column]:.2f} |\n")
                f.write(f"| {column} - Std Dev | {summary_stats.loc['std', column]:.2f} |\n")
                f.write(f"| {column} - Min | {summary_stats.loc['min', column]:.2f} |\n")
                f.write(f"| {column} - 25th Percentile | {summary_stats.loc['25%', column]:.2f} |\n")
                f.write(f"| {column} - 50th Percentile (Median) | {summary_stats.loc['50%', column]:.2f} |\n")
                f.write(f"| {column} - 75th Percentile | {summary_stats.loc['75%', column]:.2f} |\n")
                f.write(f"| {column} - Max | {summary_stats.loc['max', column]:.2f} |\n")
                f.write("|--------------|-------|\n\n")

            # Missing Values
            f.write("## Missing Values\n")
            f.write("The following columns contain missing values, with their respective counts:\n")
            f.write("\n| Column       | Missing Values Count |\n")
            f.write("|--------------|----------------------|\n")
            for column, count in missing_values.items():
                f.write(f"| {column} | {count} |\n")
            f.write("\n")

            # Outliers
            f.write("## Outliers Detection\n")
            f.write("The following columns contain outliers detected using the IQR method (values beyond the typical range):\n")
            f.write("\n| Column       | Outlier Count |\n")
            f.write("|--------------|---------------|\n")
            for column, count in outliers.items():
                f.write(f"| {column} | {count} |\n")
            f.write("\n")

            # Correlation Matrix
            f.write("## Correlation Matrix\n")
            f.write("Below is the correlation matrix of numerical features, indicating relationships between different variables:\n\n")
            if corr_matrix is not None and not corr_matrix.empty:
                f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

            # Outliers Visualization
            f.write("## Outliers Visualization\n")
            f.write("This chart visualizes the number of outliers detected in each column:\n\n")
            f.write("![Outliers](outliers.png)\n\n")

            # Distribution Plot
            f.write("## Distribution of Data\n")
            f.write("Below is the distribution plot of the first numerical column in the dataset:\n\n")
            f.write("![Distribution](distribution_.png)\n\n")

            # Conclusion
            f.write("## Conclusion\n")
            f.write("The analysis has provided insights into the dataset, including summary statistics, outlier detection, and correlations between key variables.\n")
            f.write("The generated visualizations and statistical insights can help in understanding the patterns and relationships in the data.\n\n")

            f.write("## Data Story\n")

        return readme_file
    except Exception as e:
        print(f"Error writing to README.md: {e}")
        return None

def question_llm(prompt, context):
    print("Generating story using LLM...")
    token = os.environ.get("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDM3MTFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.kLEdCZTEIM2OU741NM3WIDcgf02PLGG66eU04s7ibj0")
    if not token:
        print("AIPROXY_TOKEN not set. Unable to generate story.")
        return "AIPROXY_TOKEN not set."

    api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    full_prompt = f"""
    Based on the following data analysis, please generate a creative and engaging story. The story should include multiple paragraphs, a clear structure with an introduction, body, and conclusion, and should feel like a well-rounded narrative.

    Context:
    {context}

    Data Analysis Prompt:
    {prompt}

    The story should be elaborate and cover the following:
    - An introduction to set the context.
    - A detailed body that expands on the data points and explores their significance.
    - A conclusion that wraps up the analysis and presents any potential outcomes or lessons.
    - Use transitions to connect ideas and keep the narrative flowing smoothly.
    - Format the story with clear paragraphs and structure.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        story = response.json()['choices'][0]['message']['content'].strip()
        print("Story generated.")
        return story
    else:
        print(f"Error with request: {response.status_code} - {response.text}")
        return "Failed to generate story."

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Load and analyze dataset
    df, df_cleaned, missing_values = load_and_clean_data(dataset_path)
    summary_stats = df.describe(include='all')
    if len(df_cleaned.columns) > 1:
        corr_matrix = df_cleaned.corr()
    else:
        corr_matrix = pd.DataFrame()

    outliers = detect_outliers(df)

    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Visualize data
    heatmap_file, outliers_file, dist_plot_file = visualize_data(corr_matrix, outliers, df, output_dir)

    # Generate story
    story = question_llm(
        "Generate a nice and creative story from the analysis",
        context=f"Dataset Analysis:\nSummary Statistics:\n{summary_stats}\n\nMissing Values:\n{missing_values}\n\nCorrelation Matrix:\n{corr_matrix}\n\nOutliers:\n{outliers}"
    )

    # Create README and append story
    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir)
    if readme_file and story:
        try:
            with open(readme_file, 'a', encoding='utf-8') as f:
                f.write("## Story\n")
                f.write(story + "\n")
            print("Analysis complete! Results saved in current directory.")
            print(f"README file: {readme_file}")
            print(f"Visualizations: {heatmap_file}, {outliers_file}, {dist_plot_file}")
        except Exception as e:
            print(f"Error appending story to README.md: {e}")
    else:
        print("Error generating the README.md file or story.")

if __name__ == "__main__":
    main()
