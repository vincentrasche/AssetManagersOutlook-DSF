import os
import pickle
import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# ─── 1) Load cached DataFrame ──────────────────────────────────────────────────
CACHE = "report_data_last.pkl"
if not os.path.exists(CACHE):
    raise FileNotFoundError(f"{CACHE} not found – run your PDF parsing script first.")
with open(CACHE, "rb") as f:
    df = pickle.load(f)

# Ensure we have text
if "text" not in df.columns or df["text"].astype(str).str.strip().eq("").all():
    raise RuntimeError("No valid 'text' column in cache. Re-parse your PDFs.")

# ─── 2) Prepare grouped documents ──────────────────────────────────────────────
#   One document per firm
firm_groups = (
    df.groupby("firm")["text"]
      .apply(lambda ts: " ".join(t for t in ts if t and t.strip()))
      .reset_index()
)
firm_groups = firm_groups[firm_groups["text"].str.strip() != ""]
docs_f = firm_groups["text"].tolist()
firms  = firm_groups["firm"].tolist()

#   One document per year
year_groups = (
    df.groupby("year")["text"]
      .apply(lambda ts: " ".join(t for t in ts if t and t.strip()))
      .reset_index()
)
year_groups = year_groups[year_groups["text"].str.strip() != ""]
docs_y = year_groups["text"].tolist()
years  = year_groups["year"].tolist()

print(f"Prepared {len(firms)} firm‐level docs and {len(years)} year‐level docs.\n")

# ─── 3) Define BERTopic variants ────────────────────────────────────────────────
models_f = {
    "base": BERTopic(
        nr_topics=10,
        calculate_probabilities=True,
        vectorizer_model=CountVectorizer(stop_words="english")
    ),
    "tfidf": BERTopic(
        nr_topics=10,
        calculate_probabilities=True,
        vectorizer_model=TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    ),
    "umap5": BERTopic(
        nr_topics=10,
        calculate_probabilities=True,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.1),
        vectorizer_model=CountVectorizer(stop_words="english")
    ),
    # HDBSCAN with probabilities enabled
    "hdbscan_prob": BERTopic(
        nr_topics="auto",
        calculate_probabilities=True,
        hdbscan_model=HDBSCAN(min_cluster_size=5,
                               min_samples=1,
                               prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english")
    ),
    # HDBSCAN without probabilities
    "hdbscan_no_prob": BERTopic(
        nr_topics="auto",
        calculate_probabilities=False,
        hdbscan_model=HDBSCAN(min_cluster_size=5, min_samples=1),
        vectorizer_model=CountVectorizer(stop_words="english")
    )
}

# ─── 4) Fit & display firm‐level results ─────────────────────────────────────────
for name, model in models_f.items():
    print(f"\n=== Firm‐level BERTopic ({name}) ===")
    topics, probs = model.fit_transform(docs_f)

    info = model.get_topic_info()
    print(info.to_string(), "\n")    # full topic table

    for t in info.Topic:
        if t < 0:
            continue
        top_words = [w for w, _ in model.get_topic(t)]
        print(f"Topic {t}: {top_words}")
    print("\n" + "-"*80)

# ─── 5) Fit & display year‐level results ─────────────────────────────────────────
for name in models_f:
    # re‐instantiate to avoid carrying over embeddings
    cfg = models_f[name]
    model_y = BERTopic(
        nr_topics=(cfg.nr_topics),
        calculate_probabilities=cfg.calculate_probabilities,
        umap_model=getattr(cfg, "umap_model", None),
        hdbscan_model=getattr(cfg, "hdbscan_model", None),
        vectorizer_model=cfg.vectorizer_model
    )

    print(f"\n=== Year‐level BERTopic ({name}) ===")
    topics, probs = model_y.fit_transform(docs_y)

    info = model_y.get_topic_info()
    print(info.to_string(), "\n")
    for t in info.Topic:
        if t < 0:
            continue
        top_words = [w for w, _ in model_y.get_topic(t)]
        print(f"Topic {t}: {top_words}")
    print("\n" + "="*80)

# ─── 6) (Optional) Save one variant’s CSVs ──────────────────────────────────────
# Example: save firm‐level TF–IDF top words
chosen = models_f["tfidf"]
tw = []
for t in chosen.get_topic_info().Topic:
    if t < 0: continue
    for w, wt in chosen.get_topic(t):
        tw.append({"group":"firm","topic":t,"word":w,"weight":wt})
pd.DataFrame(tw).to_csv("bertopic_firm_top_words_tfidf.csv", index=False)

print("\nDone — all topics printed and example CSV saved.")

