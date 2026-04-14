import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv("ner_results_fixed.csv")

# Group entities back per abstract
abstracts = df.groupby(["abstract_id", "question"]).apply(
    lambda g: g[["entity", "type", "umls_id"]].to_dict("records")
).reset_index(name="entities")

# Tokenize using BioBERT's WordPiece tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
tokens = tokenizer(abstracts["question"].tolist(), padding=True, truncation=True, return_tensors="pt")
