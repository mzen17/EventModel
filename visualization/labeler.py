import streamlit as st
import pandas as pd
import json
import os

def load_data(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return pd.DataFrame(data)

def save_review(review_df, output_path):
    # Only save rows that have been reviewed (either marked correct or corrected)
    # For simplicity in this labeler, we'll save the whole batch being shown
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, row in review_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

def main():
    st.set_page_config(page_title="Model Output Labeler", layout="wide")
    st.title("Model Output Labeler")

    input_file = "output/processed.jsonl"
    output_file = "output/review.jsonl"

    if not os.path.exists(input_file):
        st.error(f"Input file {input_file} not found.")
        return

    # Load data
    df = load_data(input_file)
    
    if df.empty:
        st.warning("Processed dataset is empty.")
        return

    # Pagination state
    if 'start_idx' not in st.session_state:
        st.session_state.start_idx = 0

    total_rows = len(df)
    batch_size = 100
    end_idx = min(st.session_state.start_idx + batch_size, total_rows)

    subset = df.iloc[st.session_state.start_idx:end_idx].copy()
    
    # Initialize labeling columns if not present
    if 'is_wrong' not in subset.columns:
        subset['is_wrong'] = False
    if 'corrected_response' not in subset.columns:
        subset['corrected_response'] = ""

    st.sidebar.markdown(f"### Progress: {st.session_state.start_idx} to {end_idx} of {total_rows}")
    
    # Navigation
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("Previous 100") and st.session_state.start_idx >= batch_size:
        st.session_state.start_idx -= batch_size
        st.rerun()
    if col_next.button("Next 100") and end_idx < total_rows:
        st.session_state.start_idx += batch_size
        st.rerun()

    st.write(f"Displaying rows {st.session_state.start_idx} to {end_idx}")

    reviewed_data = []

    # Iterate through batch for spacious display
    for idx, row in subset.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Original Post**")
                st.text_area("Post", row['original_post'], height=300, disabled=True, key=f"orig_{idx}", label_visibility="collapsed")
            with col2:
                st.markdown("**Model Response**")
                st.text_area("Response", row['model_response'], height=300, disabled=True, key=f"model_{idx}", label_visibility="collapsed")
            
            c1, c2 = st.columns([1, 5])
            is_wrong = c1.checkbox("Is Wrong?", key=f"wrong_{idx}")
            corrected = ""
            if is_wrong:
                corrected = st.text_area("Rewritten Output", key=f"fix_{idx}", height=200)
            
            reviewed_data.append({
                **row.to_dict(),
                "is_wrong": is_wrong,
                "corrected_response": corrected
            })

    if st.button("Save Batch to Review File", use_container_width=True):
        save_review(pd.DataFrame(reviewed_data), output_file)
        st.success(f"Saved {len(reviewed_data)} rows to {output_file}")

if __name__ == "__main__":
    main()
