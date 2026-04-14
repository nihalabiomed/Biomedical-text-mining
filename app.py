import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import os

# --- 1. SETUP ---
st.set_page_config(page_title="Bio-NLP Search Engine", layout="wide")
st.title("🧬 Biomedical Semantic Search")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- 2. LOAD & GROUP DATA (Combining Person B & C's Logic) ---
@st.cache_data
def load_and_group_data():
    # Load the results from Person B
    df = pd.read_csv(get_path('ner_results_fixed.csv'))
    bio_df = pd.read_csv(get_path('bioasq_output.csv'))
    
    # Person C's grouping logic: 34,000 rows -> 2,291 unique abstracts
    abstracts_grouped = df.groupby(["abstract_id", "question"]).apply(
        lambda g: g[["entity", "type", "umls_id"]].to_dict("records")
    ).reset_index(name="entities")
    
    # Merge with the full text from bioasq_output.csv
    # We use 'question' as the bridge to connect the two files
    final_data = pd.merge(
        abstracts_grouped, 
        bio_df[['questions', 'abstracts', 'titles']], 
        left_on='question', 
        right_on='questions', 
        how='inner'
    )
    return final_data

# --- 3. LOAD SEARCH ENGINE (With Automatic Fix for 'MARK' Error) ---
@st.cache_resource
def load_search_engine(text_list):
    # Using the high-performance model Person C showed in her first screenshot
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    pickle_path = get_path('embeddings.pkl')
    
    try:
        # Attempt to load the file from Person C
        with open(pickle_path, 'rb') as f:
            embeddings = pickle.load(f)
        st.success("✅ Search Engine loaded from file.")
    except Exception:
        # If 'MARK' error happens, we build it ourselves!
        with st.spinner("⚠️ The 'Brain' file (embeddings.pkl) was broken or missing. Re-building it now... (2-3 mins)"):
            embeddings = model.encode(text_list, convert_to_tensor=True, show_progress_bar=True)
            # Save a working version so the next time it's instant
            with open(pickle_path, 'wb') as f:
                pickle.dump(embeddings, f)
        st.success("✨ New 'Brain' generated and saved successfully!")
        
    return model, embeddings

# --- 4. THE INTERFACE ---
data = load_and_group_data()
model, corpus_embeddings = load_search_engine(data['question'].tolist())

query = st.text_input("🔍 Ask a clinical or genetic question:", placeholder="e.g. Which genes are associated with speech development?")

if query:
    # Math: Convert query to vector and find matches
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]
    
    st.markdown(f"### Top 5 Relevant Research Papers")
    
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        row = data.iloc[idx]
        
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.subheader(row['question'])
                st.caption(f"**Title:** {row['titles']}")
                st.write(row['abstracts'])
                
                # Show the colorful tags from Person B
                tags = ""
                for ent in row['entities']:
                    # Different colors for Genes vs Diseases
                    bg = "#e6f1fb" if ent['type'] == 'Gene' else "#eaf3de"
                    txt = "#185fa5" if ent['type'] == 'Gene' else "#3b6d11"
                    tags += f'<span style="background-color:{bg}; color:{txt}; padding:4px 10px; border-radius:15px; margin:3px; font-size:12px; font-weight:bold; display:inline-block;">{ent["entity"]}</span>'
                st.markdown(tags, unsafe_allow_html=True)
                
            with col2:
                st.metric("Similarity", f"{score:.1%}")