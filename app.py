import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import os

# --- 1. SETUP ---
st.set_page_config(page_title="Bio-NLP Search Engine", layout="wide")
st.title("🧬 Biomedical Semantic Search")
st.markdown("### Integrating Bio-NLP Extraction with Semantic Retrieval")

# Finder for GitHub/Local paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- 2. LOAD & GROUP DATA ---
@st.cache_data
def load_and_group_data():
    # Load Person B & A's files
    df = pd.read_csv(get_path('ner_results_fixed.csv'))
    bio_df = pd.read_csv(get_path('bioasq_output.csv'))
    
    # Group entities by the question they belong to
    abstracts_grouped = df.groupby(["question"]).apply(
        lambda g: g[["entity", "type"]].drop_duplicates().to_dict("records")
    ).reset_index(name="entities")
    
    # Connect everything using 'question' as the bridge
    final_data = pd.merge(
        abstracts_grouped, 
        bio_df[['questions', 'abstracts', 'titles']], 
        left_on='question', 
        right_on='questions', 
        how='inner'
    )
    return final_data

# --- 3. LOAD ENGINE (Self-Healing Fail-safe) ---
@st.cache_resource
def load_search_engine(text_list):
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    pickle_path = get_path('embeddings.pkl')
    
    try:
        with open(pickle_path, 'rb') as f:
            embeddings = pickle.load(f)
        st.success("✅ Search Engine Ready.")
    except Exception:
        with st.spinner("⌛ Syncing Search Index (First-time setup)..."):
            embeddings = model.encode(text_list, convert_to_tensor=True, show_progress_bar=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(embeddings, f)
        st.success("✨ Index built successfully!")
    return model, embeddings

# --- 4. RUN INTERFACE ---
data = load_and_group_data()
model, corpus_embeddings = load_search_engine(data['questions'].tolist())

query = st.text_input("🔍 Search medical abstracts:", placeholder="e.g. Which genes cause speech development?")

if query:
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]
    
    st.markdown("---")
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        row = data.iloc[idx]
        
        # Display each paper in a clean container
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.subheader(row['questions'])
                st.caption(f"**Title:** {row['titles']}")
                st.write(row['abstracts'])
                
                # THE FIX: Only show tags that are actually IN this abstract text
                abstract_lower = row['abstracts'].lower()
                tags_html = ""
                
                # Check each entity from our mapping
                for ent in row['entities']:
                    if str(ent['entity']).lower() in abstract_lower:
                        # Color coding: Blue for Genes, Green for Diseases
                        bg = "#e6f1fb" if ent['type'] == 'Gene' else "#eaf3de"
                        txt = "#185fa5" if ent['type'] == 'Gene' else "#3b6d11"
                        tags_html += f'<span style="background-color:{bg}; color:{txt}; padding:4px 12px; border-radius:15px; margin:4px; font-size:12px; font-weight:bold; display:inline-block;">{ent["entity"]}</span>'
                
                if tags_html:
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.caption("No specific entities highlighted for this snippet.")
            
            with col2:
                st.metric("Similarity", f"{score:.1%}")
            
            st.markdown("<br>", unsafe_allow_html=True)
