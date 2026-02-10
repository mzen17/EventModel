import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px

# Use sentence-transformers for high-quality feature vectors
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
except ImportError:
    st.error("Please install sentence-transformers: `pip install sentence-transformers`")
    st.stop()

@st.cache_resource
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def load_processed_data(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        # Parse the model response which is a JSON string
                        resp = json.loads(entry['model_response'])
                        if 'characters' in resp:
                            for char_list in resp['characters']:
                                # Convert character list to a single string label
                                label = ", ".join([str(x) for x in char_list])
                                data.append(label)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
    return list(set(data)) # Unique labels

def main():
    st.set_page_config(page_title="Character Label Sampler", layout="wide")
    st.title("Character Label Clustering & Visualization")

    input_file = "output/processed.jsonl"
    
    if not os.path.exists(input_file):
        st.error(f"Input file {input_file} not found. Please run the inference script first.")
        return

    with st.spinner("Loading labels and computing embeddings..."):
        labels = load_processed_data(input_file)
        if not labels:
            st.warning("No labels found in processed data.")
            return
        
        model = get_model()
        embeddings = model.encode(labels)
        # Normalize for cosine similarity
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    st.sidebar.header("Configuration")
    threshold = st.sidebar.slider("Clustering Similarity Threshold", 0.0, 1.0, 0.95)
    # Threshold for AgglomerativeClustering with 'cosine' affinity 
    # translates to distance (1 - similarity).
    distance_threshold = 1.0 - threshold

    with st.spinner("Clustering..."):
        # Perform Agglomerative Clustering
        # affinity='cosine' and linkage='average' or 'complete'
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=distance_threshold
        )
        cluster_labels = clustering.fit_predict(norm_embeddings)
        
        # Group labels by cluster
        clusters = {}
        for label, cluster_id in zip(labels, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(label)

    # Visualization
    st.header("Cluster Visualization (2D Projection)")
    with st.spinner("Computing 2D projection..."):
        # t-SNE for dimensionality reduction
        # perplexity should be smaller than the number of samples
        perp = min(30, len(labels) - 1) if len(labels) > 1 else 1
        tsne = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto', perplexity=perp, random_state=42)
        dims_2d = tsne.fit_transform(norm_embeddings)
        
        viz_df = pd.DataFrame({
            'x': dims_2d[:, 0],
            'y': dims_2d[:, 1],
            'label': labels,
            'cluster': [str(c) for c in cluster_labels]
        })
        
        fig = px.scatter(viz_df, x='x', y='y', text='label', color='cluster', 
                         title="Character Labels Cosmic Similarity Projection",
                         labels={'cluster': 'Cluster ID'},
                         hover_data=['label'])
        fig.update_traces(textposition='top center')
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

    # Display Cluster List
    st.header(f"Merged Labels ({len(clusters)} Clusters)")
    
    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cluster_id, items in sorted_clusters:
        with st.expander(f"Cluster {cluster_id} ({len(items)} labels) - Representative: \"{items[0]}\""):
            st.write(", ".join(items))

if __name__ == "__main__":
    main()
