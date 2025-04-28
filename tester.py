import os
import pickle
import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── 1) Load cached DataFrame ──────────────────────────────────────────────────
CACHE = "report_data_last.pkl"
if not os.path.exists(CACHE):
    raise FileNotFoundError(f"{CACHE} not found – run your PDF parsing script first.")
with open(CACHE, "rb") as f:
    df = pickle.load(f)

# Ensure there is text to process
if "text" not in df.columns or df["text"].astype(str).str.strip().eq("").all():
    raise RuntimeError("No valid 'text' column in cache. Re-parse your PDFs.")

# ─── 2) Prepare documents ──────────────────────────────────────────────────────
# Firm-level documents
firm_groups = (
    df.groupby("firm")["text"]
      .apply(lambda ts: " ".join(t for t in ts.dropna().astype(str) if t.strip()))
      .reset_index()
)
firm_groups = firm_groups[firm_groups["text"].str.strip() != ""]
docs_f = firm_groups["text"].tolist()

# Year-level documents
year_groups = (
    df.groupby("year")["text"]
      .apply(lambda ts: " ".join(t for t in ts.dropna().astype(str) if t.strip()))
      .reset_index()
)
year_groups = year_groups[year_groups["text"].str.strip() != ""]
docs_y = year_groups["text"].tolist()

# ─── 3) Instantiate TF–IDF BERTopic model ───────────────────────────────────────
tfidf_model_f = BERTopic(
    nr_topics=10,
    calculate_probabilities=True,
    vectorizer_model=TfidfVectorizer(stop_words="english", ngram_range=(1,2))
)

# ─── 4) Fit & transform firm-level docs ────────────────────────────────────────
topics_f, probs_f = tfidf_model_f.fit_transform(docs_f)

# Optional: print a summary of the top 5 topics
info_f = tfidf_model_f.get_topic_info().nlargest(5, "Count")[["Topic", "Count", "Name"]]
print("Top 5 Firm-Level Topics (TF–IDF):")
print(info_f.to_string(), "\n")

# ─── 5) Save firm-level top words to CSV ───────────────────────────────────────
rows_f = []
for t in tfidf_model_f.get_topic_info().Topic:
    if t < 0:
        continue
    for word, weight in tfidf_model_f.get_topic(t):
        rows_f.append({
            "topic": t,
            "word": word,
            "weight": weight
        })

firm_out = "bertopic_firm_top_words_tfidf.csv"
pd.DataFrame(rows_f).to_csv(firm_out, index=False)
print(f"Saved firm-level TF–IDF top words to {firm_out}")

# ─── 6) Repeat for year-level docs ─────────────────────────────────────────────
tfidf_model_y = BERTopic(
    nr_topics=10,
    calculate_probabilities=True,
    vectorizer_model=TfidfVectorizer(stop_words="english", ngram_range=(1,2))
)

topics_y, probs_y = tfidf_model_y.fit_transform(docs_y)

# Optional: print a summary of the top 5 topics
info_y = tfidf_model_y.get_topic_info().nlargest(5, "Count")[["Topic", "Count", "Name"]]
print("Top 5 Year-Level Topics (TF–IDF):")
print(info_y.to_string(), "\n")

# ─── 7) Save year-level top words to CSV ───────────────────────────────────────
rows_y = []
for t in tfidf_model_y.get_topic_info().Topic:
    if t < 0:
        continue
    for word, weight in tfidf_model_y.get_topic(t):
        rows_y.append({
            "topic": t,
            "word": word,
            "weight": weight
        })

year_out = "bertopic_year_top_words_tfidf.csv"
pd.DataFrame(rows_y).to_csv(year_out, index=False)
print(f"Saved year-level TF–IDF top words to {year_out}")
