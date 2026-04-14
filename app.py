import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import os

# --- 1. SETUP ---
st.set_page_config(page_title="Bio-NLP Search", layout="wide")
st.title("🧬 Biomedical Semantic Search")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    ner_df = pd.read_csv(get_path('ner_results_fixed.csv'))
    bio_df = pd.read_csv(get_path('bioasq_output.csv'))
    # Fix potential naming mismatch
    bio_df['questions'] = bio_df['questions'].fillna("No Question")
    
    entity_map = ner_df.groupby('question').apply(
        lambda x: x[['entity', 'type']].to_dict('records')
    ).to_dict()
    return bio_df, entity_map

# --- 3. LOAD ENGINE (WITH EMERGENCY BACKUP) ---
@st.cache_resource
def load_engine(questions_list):
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    pickle_path = get_path('embeddings.pkl')
    
    # Try to load the file
    try:
        with open(pickle_path, 'rb') as f:
            embeddings = pickle.load(f)
        st.success("✅ Loaded pre-computed embeddings.")
    except Exception:
        # EMERGENCY BACKUP: If file is broken, we make it ourselves!
        with st.spinner("⚠️ Embeddings file was broken. Re-generating 'Brain'... Please wait 2-3 minutes."):
            embeddings = model.encode(questions_list, convert_to_tensor=True, show_progress_bar=True)
            # Save a working version so it's fast next time
            with open(pickle_path, 'wb') as f:
                pickle.dump(embeddings, f)
        st.success("✨ New 'Brain' generated and saved!")
        
    return model, embeddings

# --- 4. RUN APP ---
bio_df, entity_map = load_data()
model, corpus_embeddings = load_engine(bio_df['questions'].tolist())

query = st.text_input("🔍 Search medical questions:", placeholder="e.g. Hirschsprung disease")

if query:
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_embeddings, top_k=5)[0]
    
    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        q_text = bio_df.iloc[idx]['questions']
        abs_text = bio_df.iloc[idx]['abstracts']
        
        with st.expander(f"Match {score:.1%} | {q_text[:80]}..."):
            st.write(f"**Abstract:** {abs_text}")
            if q_text in entity_map:
                tags = ""
                for e in entity_map[q_text]:
                    color = "#e6f1fb" if e['type'] == 'Gene' else "#eaf3de"
                    tags += f'<span style="background-color:{color}; padding:2px 8px; border-radius:10px; margin:2px; font-size:12px;">{e["entity"]} ({e["type"]})</span> '
                st.markdown(tags, unsafe_allow_html=True)
                    
                    # Exact or
