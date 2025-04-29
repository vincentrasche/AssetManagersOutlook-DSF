import os
import re
import pickle
import pandas as pd

from pdfminer.high_level import extract_text
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from hdbscan import HDBSCAN

# ─── 0) Define extra stop-words ─────────────────────────────────────────────
domain_stopwords = {
    # years
    "2020","2021","2022","2023","2024","2025",
    # generic finance / boilerplate
    "investment","investments","market","markets","investors","growth","global",
    "strategist","phd","capital","outlook","economic","economy","returns","year",
    "expected","new","equity","economist", "blackrock","assumptions", "estimates", "cross-roads", "vice president"
    "vice president", "chapter", "accessible", "earnings", "goldman", "sachs", "assets"
    # already common proper names
    "goldman","sachs","assets",
    # asset managers (split into tokens)
    "aqr","amundi","bmo","bny","mellon",
    "blackrock",
    "capital","group",
    "fidelity",
    "franklin","templeton",
    "gmo",
    "goldman","sachs",
    "invesco",
    "morgan",        # from J.P. Morgan
    "morningstar",
    "northern","trust",
    "state","street",
    "rowe","price",
    "ubs",
    "vanguard",
    "voya",
    "wells","fargo"
}

stop_words = list(ENGLISH_STOP_WORDS.union(domain_stopwords))

# ─── 1) Parse first N pages of each PDF ────────────────────────────────────
def parse_first_pages(folder, n_pages=2):
    pat = re.compile(r"(\d+)_([A-Za-z]+)_?(\d{4})\.pdf")
    rows = []
    for fn in os.listdir(folder):
        m = pat.match(fn)
        if not m: continue
        _, firm, year = m.groups()
        txt = extract_text(os.path.join(folder, fn),
                           page_numbers=list(range(n_pages)))
        if not txt.strip(): continue
        rows.append({"filename": fn, "firm": firm, "year": int(year), "text": txt})
    return pd.DataFrame(rows)

# ─── 2) Load or build cache ─────────────────────────────────────────────────
FOLDER, CACHE2 = "Yearly data outlooks", "report_data_first2.pkl"
if os.path.exists(CACHE2):
    df = pickle.load(open(CACHE2, "rb"))
else:
    df = parse_first_pages(FOLDER, n_pages=2)
    pickle.dump(df, open(CACHE2, "wb"))

# ─── 3) Core: fit + save + top-topic+score per group ────────────────────────
def run_hdbscan(docs, labels, group_name):
    vect = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=(1,3),
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b"
    )
    model = BERTopic(
        nr_topics="auto",
        calculate_probabilities=True,
        hdbscan_model=HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True),
        vectorizer_model=vect
    )

    print(f"\n▶ Fitting on {group_name} ({len(docs)} docs)…")
    topics, probs = model.fit_transform(docs)

    # 3a) topic → word → weight
    rows = []
    for t in model.get_topic_info().Topic:
        if t < 0: continue
        for w, wt in model.get_topic(t):
            rows.append({group_name: group_name, "topic": t, "word": w, "weight": wt})
    pd.DataFrame(rows).to_csv(f"{group_name.lower()}_topics.csv", index=False)

    # 3b) single-word labels for each topic
    topic_labels = [
        (model.get_topic(i)[0][0] if model.get_topic(i) else f"Topic {i}")
        for i in range(probs.shape[1])
    ]

    # 3c) full soft-probabilities
    dfp = pd.DataFrame(probs, columns=topic_labels)
    dfp.insert(0, group_name, labels)
    dfp.to_csv(f"{group_name.lower()}_doc_probs.csv", index=False)

    # 3d) dominant topic per doc
    dom = [topic_labels[t] if t >= 0 else "Outlier" for t in topics]
    pd.DataFrame({
        group_name: labels,
        "dominant_topic_num": topics,
        "dominant_topic":     dom
    }).to_csv(f"{group_name.lower()}_dominant_topic.csv", index=False)

    # 3e) top topic + score for all groups
    # inside run_hdbscan, replace 2e) with this

    # 2e) Export top 3 topics + scores for all groups
    top3_list = []
    for ent, prob_row in zip(labels, probs):
        # indices of top 3 probabilities
        idxs = prob_row.argsort()[::-1][:3]
        top3_list.append({
            group_name.lower():  ent,
            "topic1":            topic_labels[idxs[0]],
            "score1":            prob_row[idxs[0]],
            "topic2":            topic_labels[idxs[1]] if len(idxs)>1 else "",
            "score2":            prob_row[idxs[1]] if len(idxs)>1 else 0.0,
            "topic3":            topic_labels[idxs[2]] if len(idxs) > 2 else "",
            "score3":            prob_row[idxs[2]] if len(idxs) > 2 else 0.0,
        })
    pd.DataFrame(top3_list).to_csv(f"{group_name.lower()}_top3_topics.csv", index=False)

    return model

# ─── 4) Run for Papers, Firms, Years ────────────────────────────────────────
run_hdbscan(df["text"].tolist(), df["filename"].tolist(), "Paper")

firm_df = df.groupby("firm")["text"].apply(" ".join).reset_index(name="text")
run_hdbscan(firm_df["text"].tolist(), firm_df["firm"].tolist(), "Firm")

year_df = df.groupby("year")["text"].apply(" ".join).reset_index(name="text")
run_hdbscan(year_df["text"].tolist(), year_df["year"].tolist(), "Year")

print("\n✅ All done — <group>_top_topic.csv files include the top topic and its score.")


