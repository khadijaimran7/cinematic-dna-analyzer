import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Cinematic DNA Analyzer", page_icon="🎬", layout="wide")

# App Header
st.title("🎬 Cinematic DNA & Genre Analyzer")
st.markdown("""
Welcome! This tool uses **Natural Language Processing (Zero-Shot Classification)** to read raw movie scripts 
and determine their underlying thematic DNA. 
Instead of relying on standard IMDB tags, it analyzes the actual dialogue and action lines to find the true "vibe" of the film.
""")

# Load the pre-computed data
@st.cache_data
def load_data():
    return pd.read_csv("movie_dna.csv")

try:
    df = load_data()
    
    # Sidebar Filtering
    st.sidebar.header("Filter Options")
    genres = ["All Movies"] + list(df['Dominant Vibe'].unique())
    selected_vibe = st.sidebar.selectbox("Filter by Dominant Vibe:", genres)
    
    # Filter logic
    if selected_vibe != "All Movies":
        filtered_df = df[df['Dominant Vibe'] == selected_vibe]
    else:
        filtered_df = df
        
    # Display the Data
    st.subheader(f"Results: {selected_vibe}")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.caption("Built with Python, HuggingFace (BART-Large-MNLI), and Streamlit.")

except FileNotFoundError:
    st.error("Data file not found. Please ensure 'movie_dna.csv' is in the repository.")