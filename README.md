# Biomedical-text-mining
Project on biomedical text mining / NLP for genomics literature ie using AI/NLP to improve how research papers are searched and analyzed. 
## 🚀 Live Demo
Access the live Bio-AI Semantic Search Engine here: [Biomedical Text Mining App](https://biomedical-text-mining-5t9q5j6jiz4hhawstbqnns.streamlit.app/#which-gene-is-responsible-for-proper-speech-development)


📋 Project Contributions:

Phase 1: Data Acquisition & Pre-processing (Anagha)
Curated a specialized biomedical dataset consisting of 2,291 research abstracts and clinical questions.

Cleaned and formatted the raw BioASQ data into a structured CSV format suitable for machine learning pipelines.


Phase 2: Bio-NER Pipeline (Sebin)
Developed a Named Entity Recognition (NER) system to identify biological entities within the text.

Successfully categorized entities into Genes and Diseases, providing the foundation for our knowledge-tagging system.


Phase 3: Transformer Model Selection (Uma)
Evaluated multiple NLP models, selecting the multi-qa-mpnet-base-dot-v1 Sentence-Transformer for its high-performance semantic retrieval.
Prototyped the tokenization and vectorization logic to convert medical text into mathematical "embeddings."


Phase 4: Full-Stack Integration & Deployment (Nihala - Person D)
System Integration: Engineered the "Glue Code" to connect the NER tags (Person B) with the semantic search engine (Person C).

Robust Backend: Implemented a "Self-Healing" fail-safe mechanism that automatically detects and re-builds corrupted embedding indexes to ensure 100% app uptime.

UI/UX Design: Built the interactive web interface using Streamlit, featuring real-time similarity scoring and colorful entity tagging.

Cloud Deployment: Orchestrated the CI/CD pipeline to host the application live on Streamlit Cloud with automated dependency management.
