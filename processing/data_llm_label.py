import pandas as pd
import json
import os
from openai import OpenAI
from tqdm import tqdm

# Initialize the client with the specified server address
client = OpenAI(
    base_url="http://ws1.mzen.dev:18181/v1",
    api_key="no-key-required"
)

# Read fewshot examples from fewshot.txt
with open("fewshot.txt", "r") as f:
    fewshot_content = f.read()

# Load data from parquet
input_file = "data/parenting_posts.parquet"
output_file = "output/processed.jsonl"

df = pd.read_parquet(input_file)

# Filter: is_self == True and selftext length >= 300
filtered_df = df[(df['is_self'] == True) & (df['selftext'].str.len() >= 300)]

# Cap to 500 samples
param_count = 20
if len(filtered_df) > 20:
    filtered_df = filtered_df.sample(n=20, random_state=42)

print(f"Total posts to process: {len(filtered_df)}")

# Process each row
for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    post_text = row['selftext']
    prompt = f"{fewshot_content}\n[POST]\n{post_text}\n[ANS]"

    try:
        response = client.chat.completions.create(
            model="NexaAI/Qwen3-VL-4B-Instruct-GGUF",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200 # Increased slightly for more complex answers
        )

        model_output = response.choices[0].message.content

        # Save result
        result = {
            "permalink": row['permalink'],
            "original_post": post_text,
            "model_response": model_output
        }

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        # Continue to next row even if one fails
        continue

print(f"Processing complete. Results saved to {output_file}")
