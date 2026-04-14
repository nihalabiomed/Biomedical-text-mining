import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Bio-NLP Search Engine", layout="wide")
st.title("🧬 Biomedical Semantic Search")
st.markdown("### Integrating Named Entity Recognition (NER) with Semantic Retrieval")

# Helper to locate files in the GitHub/Local environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- 2. DATA INTEGRATION (The "Glue") ---
@st.cache_data
def load_and_group_data():
    # Load the datasets from Person A and Person B
    df = pd.read_csv(get_path('ner_results_fixed.csv'))
    bio_df = pd.read_csv(get_path('bioasq_output.csv'))
    
    # Group entities by the question text to create a mapping
    # We use drop_duplicates so we don't get the same tag multiple times
    abstracts_grouped = df.groupby(["question"]).apply(
        lambda g: g[["entity", "type"]].drop_duplicates().to_dict("records")
    ).reset_index(name="entities")
    
    # Connect the entities to the full abstracts/titles
    final_data = pd.merge(
        abstracts_grouped, 
        bio_df[['questions', 'abstracts', 'titles']], 
        left_on='question', 
        right_on='questions', 
        how='inner'
    )
    return final_data

# --- 3. SEARCH ENGINE ENGINE (Self-Healing Fail-safe) ---
@st.cache_resource
def load_search_engine(text_list):
    # Using the high-accuracy MPNet model
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    pickle_path = get_path('embeddings.pkl')
    
    try:
        # Try to load existing 'Brain' file
        with open(pickle_path, 'rb') as f:
            embeddings = pickle.load(f)
        st.success("✅ Semantic Index Loaded.")
    except Exception:
        # Re-build if file is missing or corrupted
        with st.spinner("⌛ Generating Semantic 'Brain' (First-time setup)..."):
            embeddings = model.encode(text_list, convert_to_tensor=True, show_progress_bar=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(embeddings, f)
        st.success("✨ New Index built successfully!")
    return model, embeddings

# --- 4. APPLICATION INTERFACE ---
data = load_and_group_data()
model, corpus_embeddings = load_search_engine(data['questions'].tolist())

# Search Bar
query = st.text_input("🔍 Ask a clinical or genetic question:", placeholder="e.g. Which genes are associated with speech development?")

if query:
    # Convert query to vector and find top 5 matches
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]
    
    st.markdown("---")
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        row = data.iloc[idx]
        
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.subheader(row['questions'])
                st.caption(f"**Title:** {row['titles']}")
                st.write(row['abstracts'])
                
                # --- TAG SYSTEM ---
                # Verify that the tagged entity actually appears in THIS abstract
                abstract_lower = row['abstracts'].lower()
                tags_html = ""
                
                for ent in row['entities']:
                    entity_name = str(ent['entity']).lower()
                    
                    # Exact or
