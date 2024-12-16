# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "tenacity",
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
from tenacity import retry, stop_after_attempt, wait_fixed

# Set your OpenAI API key (AI Proxy token)
openai.api_key = os.environ.get("AIPROXY_TOKEN")

if len(sys.argv) < 2:
    print("Usage: uv run autolysis.py dataset.csv")
    sys.exit(1)

input_csv = sys.argv[1]

# Load the CSV into a DataFrame
df = pd.read_csv(input_csv)

# Basic Analysis
num_rows, num_cols = df.shape
column_info = []
for col in df.columns:
    col_data = df[col]
    column_info.append({
        "name": col,
        "dtype": str(col_data.dtype),
        "num_missing": col_data.isnull().sum(),
        "example_values": col_data.dropna().sample(min(5, len(col_data.dropna()))).tolist() if len(col_data.dropna()) > 0 else []
    })

basic_stats = df.describe(include='all', datetime_is_numeric=True).to_dict()

# Create a summary to send to LLM
summary_prompt = f"""
We have a dataset with {num_rows} rows and {num_cols} columns.
The columns are:
{[ (c["name"], c["dtype"], c["num_missing"], c["example_values"]) for c in column_info ]}

Basic stats (describe):
{basic_stats}

Please suggest a few interesting analyses or insights we might derive from this data. 
Be generic and consider that we don't know the domain of the dataset.
"""

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def ask_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a data expert."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

analysis_suggestions = ask_llm(summary_prompt)

# Print suggestions to console (just for debugging)
print("LLM Analysis Suggestions:\n", analysis_suggestions)

# For demonstration, let's assume we do a correlation analysis on numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    # Plot correlation heatmap
    plt.figure(figsize=(6,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

# After doing these analyses, we might prompt the LLM for a narrative
# We'll describe what we did and what we found, and ask LLM to produce a story.

final_prompt = f"""
We had a dataset with {num_rows} rows and {num_cols} columns. 
We computed basic stats and missing values. We also generated a correlation heatmap of numeric columns.

Here are some key points:
- Columns: {[ (c["name"], c["dtype"]) for c in column_info ]}
- Missing values: {{col: c["num_missing"] for c in column_info}}
- Basic numeric correlations have been plotted (see correlation_heatmap.png).

Based on these analyses, please write a well-structured Markdown story describing the dataset, the analysis performed, key insights, and possible next steps. Mention the correlation_heatmap.png image in the narrative.
"""

narrative = ask_llm(final_prompt)

# Save the narrative as README.md
with open("README.md", "w") as f:
    f.write(narrative)

print("Analysis complete. README.md and charts created.")

