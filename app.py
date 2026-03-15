import streamlit as st
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Configuration
st.set_page_config(page_title="Cinematic RAG Engine", page_icon="🎬", layout="wide")

# --- SIDEBAR: PORTFOLIO INFO ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4233/4233830.png", width=100) # Generic tech icon
st.sidebar.markdown("### 👩‍💻 Developed by")
st.sidebar.markdown("**Khadija Imran**\n\n*BSc Computer Science @ LUMS '26*")
st.sidebar.markdown("Passionate about the intersection of Data Science, AI, and Human-Computer Interaction.")
st.sidebar.link_button("🔗 LinkedIn", "https://linkedin.com/in/khadija-imran-4a7262250")
st.sidebar.divider()
st.sidebar.markdown("**Tech Stack:**\n- Python & Pandas\n- Hugging Face (BART Zero-Shot)\n- Sentence-Transformers (RAG)\n- Streamlit")

# App Header
st.title("🎬 Cinematic DNA & Semantic Search Engine")
st.markdown("*A Natural Language Processing pipeline that analyzes movie scripts to extract thematic DNA and enables semantic 'vibe' searching.*")
st.divider()

# Create two tabs
tab1, tab2 = st.tabs(["🧬 Genre DNA Analyzer", "🔍 Semantic Scene Search (RAG)"])

# --- TAB 1: GENRE DNA ---
with tab1:
    st.markdown("### 📊 Movie Thematic Distribution")
    st.markdown("Filter the dataset by the AI-generated dominant vibe.")
    
    @st.cache_data
    def load_dna_data():
        return pd.read_csv("movie_dna.csv")

    try:
        df = load_dna_data()
        
        # UI: Top metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scripts Analyzed", len(df))
        col2.metric("Primary Genre", df['Dominant Vibe'].mode()[0])
        col3.metric("AI Model", "BART-Large-MNLI")
        
        st.divider()
        
        # UI: Filter and Dataframe
        genres = ["All Movies"] + list(df['Dominant Vibe'].unique())
        selected_vibe = st.selectbox("🎯 Filter by Dominant Vibe:", genres)
        
        if selected_vibe != "All Movies":
            filtered_df = df[df['Dominant Vibe'] == selected_vibe]
        else:
            filtered_df = df
            
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
    except FileNotFoundError:
        st.error("Please ensure 'movie_dna.csv' is uploaded to GitHub.")

# --- TAB 2: SEMANTIC SEARCH (RAG) ---
with tab2:
    st.markdown("### 🧠 Vector-Based Scene Retrieval")
    st.markdown("Search across all scripts using mathematical embeddings, not just keywords. Try searching for abstract concepts like: *'A tense argument between friends'* or *'Realizing they are trapped'*.")
    
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_data
    def load_embeddings():
        with open("embeddings.pkl", "rb") as f:
            return pickle.load(f)

    try:
        model = load_embedding_model()
        data = load_embeddings()
        
        # UI: Search Bar
        query = st.text_input("🔍 Enter your cinematic query:", placeholder="e.g., A quiet, philosophical moment looking at the stars")
        
        if query:
            with st.spinner("Searching the cinematic vector space..."):
                query_vec = model.encode([query])
                similarities = cosine_similarity(query_vec, data["embeddings"])[0]
                
                top_k = 3
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                st.markdown("### 🏆 Top Matches")
                
                # UI: Card-style layout for results
                for rank, idx in enumerate(top_indices, 1):
                    meta = data["metadata"][idx]
                    score = similarities[idx] * 100
                    text = data["chunks"][idx]
                    
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.subheader(f"#{rank}")
                            st.metric(label="Match Confidence", value=f"{score:.1f}%")
                            st.progress(int(score))
                        with col2:
                            st.markdown(f"**Movie:** 🎬 *{meta['movie']}*")
                            st.markdown(f"**Scene Excerpt:**")
                            st.info(f"*{text[:500]}...*")
                            
    except FileNotFoundError:
        st.error("Please ensure 'embeddings.pkl' is uploaded to GitHub.")