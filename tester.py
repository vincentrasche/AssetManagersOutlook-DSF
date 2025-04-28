import os
import re
import pickle
import pandas as pd

from pdfminer.high_level import extract_text
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

# ─── 1) Parse only the first N pages of each PDF ───────────────────────────────
def parse_first_pages(folder, n_pages=2):
    pat = re.compile(r"(\d+)_([A-Za-z]+)_?(\d{4})\.pdf")
    rows = []
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if not m:
            continue
        _, firm, year = m.groups()
        path = os.path.join(folder, fn)
        try:
            txt = extract_text(path, page_numbers=list(range(n_pages)))
        except Exception as e:
            print(f"Error extracting {fn}: {e}")
            continue
        if not txt or not txt.strip():
            continue
        rows.append({
            "filename": fn,
            "firm": firm,
            "year": int(year),
            "text": txt
        })
    return pd.DataFrame(rows)

# ─── 2) Load or build the short‐text cache ─────────────────────────────────────
FOLDER    = "Yearly data outlooks"
CACHE2    = "report_data_first2.pkl"
if os.path.exists(CACHE2):
    df = pickle.load(open(CACHE2, "rb"))
    print(f"✔ Loaded {len(df)} docs from '{CACHE2}'")
else:
    df = parse_first_pages(FOLDER, n_pages=2)
    pickle.dump(df, open(CACHE2, "wb"))
    print(f"✔ Parsed & cached {len(df)} docs (first 2 pages only) → '{CACHE2}'")

# ─── 3) Core function: fit HDBSCAN + save outputs ──────────────────────────────
def run_hdbscan(docs, labels, group_name):
    model = BERTopic(
        nr_topics="auto",                # let HDBSCAN decide cluster count
        calculate_probabilities=True,
        hdbscan_model=HDBSCAN(
            min_cluster_size=2,          # very small to allow many clusters
            min_samples=1,
            prediction_data=True
        ),
        vectorizer_model=CountVectorizer(stop_words="english")
    )
    print(f"\n▶ Fitting BERTopic+HDBSCAN on {group_name} ({len(docs)} docs)…")
    topics, probs = model.fit_transform(docs)

    # 3a) Save topic→word→weight lists
    info = model.get_topic_info()
    rows = []
    for t in info.Topic:
        if t < 0:
            continue
        for word, wt in model.get_topic(t):
            rows.append({
                group_name: group_name,
                "topic": t,
                "word": word,
                "weight": wt
            })
    topics_fn = f"{group_name.lower()}_topics.csv"
    pd.DataFrame(rows).to_csv(topics_fn, index=False)
    print(f"   • Wrote topic words → {topics_fn}")

    # 3b) Save doc‐level soft memberships
    n_topics = probs.shape[1]
    cols     = [f"topic_{i}" for i in range(n_topics)]
    dfp      = pd.DataFrame(probs, columns=cols)
    dfp.insert(0, group_name, labels)
    probs_fn = f"{group_name.lower()}_doc_probs.csv"
    dfp.to_csv(probs_fn, index=False)
    print(f"   • Wrote doc-topic probs → {probs_fn}")

    return model

# ─── 4) Run on individual Papers ───────────────────────────────────────────────
papers    = df["text"].tolist()
filenames = df["filename"].tolist()
run_hdbscan(papers, filenames, "Paper")

# ─── 5) Run on Firms (concatenate each firm’s short texts) ─────────────────────
firm_df = (
    df.groupby("firm")["text"]
      .apply(lambda ts: " ".join(ts))
      .reset_index(name="text")
)
firm_df = firm_df[firm_df["text"].str.strip() != ""]
run_hdbscan(firm_df["text"].tolist(), firm_df["firm"].tolist(), "Firm")

# ─── 6) Run on Years (concatenate each year’s short texts) ────────────────────
year_df = (
    df.groupby("year")["text"]
      .apply(lambda ts: " ".join(ts))
      .reset_index(name="text")
)
year_df = year_df[year_df["text"].str.strip() != ""]
run_hdbscan(year_df["text"].tolist(), year_df["year"].tolist(), "Year")

print("\n✅ All done — 6 CSVs created (3 groups × [topics, doc_probs]).")
