import streamlit as st
import pandas as pd
import pyarrow.parquet as pq

@st.cache_data
def load_data(file_path):
    # Use pyarrow to read the parquet file with a filter for is_self
    # This pre-filters the data at the pyarrow level for efficiency
    table = pq.read_table(
        file_path, 
        filters=[('is_self', '==', True)]
    )
    # Convert to pandas dataframe for streamlit
    df = table.to_pandas()
    
    # Filter for selftext length >= 300 characters
    df = df[df['selftext'].str.len() >= 300]
    
    return df

def main():
    st.set_page_config(page_title="Parenting Posts Data Visualizer", layout="wide")
    st.title("Parenting Posts (Self-Posts Only)")

    file_path = "data/parenting_posts.parquet"
    
    try:
        df = load_data(file_path)
        
        st.sidebar.info(f"Loaded {len(df)} self-posts.")
        
        # Select 100 random rows for preview
        sample_df = df.sample(n=min(len(df), 100))
        
        # Basic Stats
        st.write("### Data Preview (100 Random Rows)")
        # Updated to use width="stretch" per latest API
        st.dataframe(sample_df, width="stretch")
        
        st.write("### Dataset Overview (Full Filtered Dataset)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Self-Posts", len(df))
        col2.metric("Average Score", round(df['score'].mean(), 2))
        col3.metric("Average Comments", round(df['num_comments'].mean(), 2))

        # Visualizations
        st.write("### Score Distribution")
        st.bar_chart(df['score'].value_counts().head(20))

        st.write("### Top Authors")
        top_authors = df['author'].value_counts().head(10)
        st.bar_chart(top_authors)

    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please run the conversion script first.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

if __name__ == "__main__":
    main()
