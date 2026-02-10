import polars as pl

input_file = "data/parenting_posts.jsonl"
output_file = "data/parenting_posts.parquet"

print("Step 1: Reading JSONL...")

schema = {
    "title": pl.Utf8,
    "selftext": pl.Utf8,
    "url": pl.Utf8,
    "is_self": pl.Boolean,
    "author": pl.Utf8,
    "permalink": pl.Utf8,
    "score": pl.Int64,
    "ups": pl.Int64,
    "upvote_ratio": pl.Float64,
    "num_comments": pl.Int64,
    "created_utc": pl.Int64,
}

try:
    df = pl.read_ndjson(input_file, schema=schema)
except Exception:
    print("Direct schema read failed, falling back to Utf8 read...")
    df = pl.read_ndjson(input_file)
    
target_fields = [
    "title", "selftext", "url", "is_self", "author", 
    "permalink", "score", "ups", "upvote_ratio", 
    "num_comments", "created_utc"
]

existing_cols = [col for col in target_fields if col in df.columns]
df = df.select(existing_cols)

print(f"Step 2: Writing {len(df)} rows to Parquet...")
df.write_parquet(output_file, compression="snappy")

print(f"Success! Saved to {output_file}")