import os
import re
import pickle

# 1) NLTK setup
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# 2) File paths & patterns
FOLDER  = 'Yearly data outlooks'       # <- your PDF folder
CACHE   = 'report__data__last.pkl'
pattern = re.compile(r'(\d+)_([A-Za-z]+)_?(\d{4})\.pdf')

# 3) Sentiment & summarizer
import pysentiment2 as ps
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

# FinBERT sentiment
finbert_tok   = BertTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipe  = pipeline(
    "sentiment-analysis",
    model=finbert_model,
    tokenizer=finbert_tok
)

# General summarizer
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Loughran-McDonald lexicon
lm = ps.LM()

# 4) Asset / region groups
ASSETS = {
    "Equity":       ["equities","stocks"],
    "FixedIncome":  ["fixed income","bonds"],
    "RealEstate":   ["real estate"],
    "Commodities":  ["commodities"],
    "Alternatives": ["alternatives"],
    "Infra":        ["infrastructure"],
    "PE":           ["private equity"],
    "HedgeFunds":   ["hedge funds"],
    "Cash":         ["cash"],
}
REGIONS = {
    "USA":    ["us","united states","america","usa"],
    "Europe": ["europe","eu","eurozone","germany","france"],
    "Asia":   ["asia","china","japan","india"],
    "EM":     ["emerging markets"],
}

# 5) Helpers
def finbert_score(lbl):
    return 1 if lbl.lower()=="positive" else -1 if lbl.lower()=="negative" else 0

def extract_sentences(text, terms):
    text_l = text.lower()
    return [s for s in sent_tokenize(text) if any(t in s for t in terms)]

def summarize_text(txt, mx=150, mn=40):
    words = txt.split()
    if len(words) > 500:
        txt = " ".join(words[:500])
    try:
        return summarizer(txt, max_length=mx, min_length=mn, do_sample=False)[0]["summary_text"]
    except:
        return " ".join(sent_tokenize(txt)[:2])

# 6) Load or parse all PDFs
import pandas as pd
from pdfminer.high_level import extract_text

if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        df = pickle.load(f)
    print(f"✔ Loaded {len(df)} docs from cache.")
else:
    rows = []
    for fn in os.listdir(FOLDER):
        m = pattern.match(fn)
        if not m:
            continue
        idx, firm, year = m.groups()
        path = os.path.join(FOLDER, fn)
        try:
            txt = extract_text(path)
        except Exception as e:
            print(f"❌ {fn}:", e)
            continue

        # L-M sentiment
        toks = lm.tokenize(txt)
        sc   = lm.get_score(toks)
        pos, neg = sc.get("Positive", 0), sc.get("Negative", 0)
        comp = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0

        # counts
        tl = txt.lower()
        asset_counts  = {g: sum(tl.count(t) for t in terms) for g, terms in ASSETS.items()}
        region_counts = {g: sum(tl.count(t) for t in terms) for g, terms in REGIONS.items()}

        # per-aspect FinBERT
        asp_sents = {}
        for g, terms in ASSETS.items():
            sents = extract_sentences(txt, terms)
            scores = []
            for s in sents:
                try:
                    r = finbert_pipe(s, truncation=True)[0]
                    scores.append(finbert_score(r["label"]) * r["score"])
                except:
                    pass
            asp_sents[g] = sum(scores) / len(scores) if scores else None

        # per-region FinBERT
        reg_sents = {}
        for g, terms in REGIONS.items():
            sents = extract_sentences(txt, terms)
            scores = []
            for s in sents:
                try:
                    r = finbert_pipe(s, truncation=True)[0]
                    scores.append(finbert_score(r["label"]) * r["score"])
                except:
                    pass
            reg_sents[g] = sum(scores) / len(scores) if scores else None

        rows.append({
            "firm": firm,
            "year": int(year),
            "filename": fn,
            "text": txt,
            "summary": summarize_text(txt),
            "positive": pos,
            "negative": neg,
            "compound": comp,
            "asset_counts": asset_counts,
            "region_counts": region_counts,
            "aspect_sentiments": asp_sents,
            "region_sentiments": reg_sents
        })

    df = pd.DataFrame(rows)
    with open(CACHE, "wb") as f:
        pickle.dump(df, f)
    print(f"✔ Parsed & cached {len(df)} docs.")

# 7) Expand & save CSVs
asp_df = df["aspect_sentiments"].apply(pd.Series)
pdf    = pd.concat([df.drop(columns=["aspect_sentiments"]), asp_df], axis=1)
pdf.to_csv("report_data_expanded.csv", index=False)

reg_df = df["region_sentiments"].apply(pd.Series)
rdf    = pd.concat([df.drop(columns=["region_sentiments"]), reg_df], axis=1)
rdf.to_csv("report_data_expanded_regions.csv", index=False)
print("✔ Wrote expanded CSVs.")

# 8) BERTopic — firm-level
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

firms  = []
docs_f = []
for firm, grp in df.groupby("firm"):
    firms.append(firm)
    docs_f.append(" ".join(grp["text"].tolist()))

model_f = BERTopic(
    nr_topics=10,
    calculate_probabilities=True,
    vectorizer_model=CountVectorizer(stop_words="english")
)

topics_f, probs_f = model_f.fit_transform(docs_f)

# Print firm-level topics
print("\n–– Firm-Level Topics ––")
topic_info_f = model_f.get_topic_info()
print(topic_info_f.to_string())
for t in topic_info_f.Topic:
    if t < 0:
        continue
    terms = [w for w, _ in model_f.get_topic(t)]
    print(f"Topic {t}: {terms}")

# Write firm-level CSVs
tw_f     = []
rows_f   = []
freqs_f  = model_f.get_topic_freq().set_index("Topic")["Count"]

for t in topic_info_f.Topic:
    if t < 0:
        continue
    for w, wt in model_f.get_topic(t):
        tw_f.append({"group": "firm", "topic": t, "word": w, "weight": wt})

for firm, pv in zip(firms, probs_f):
    d = {"firm": firm}
    for tid in freqs_f.index:
        if tid < 0:
            continue
        d[f"topic_{tid}"] = float(pv[tid])
    rows_f.append(d)

pd.DataFrame(tw_f).to_csv("bertopic_firm_top_words.csv", index=False)
pd.DataFrame(rows_f).to_csv("bertopic_firm_doc_probs.csv", index=False)

# 9) BERTopic — year-level
years  = sorted(df["year"].unique())
docs_y = [" ".join(df[df["year"] == y]["text"].tolist()) for y in years]

model_y = BERTopic(
    nr_topics=10,
    calculate_probabilities=True,
    vectorizer_model=CountVectorizer(stop_words="english")
)

topics_y, probs_y = model_y.fit_transform(docs_y)

# Print year-level topics
print("\n–– Year-Level Topics ––")
topic_info_y = model_y.get_topic_info()
print(topic_info_y.to_string())
for t in topic_info_y.Topic:
    if t < 0:
        continue
    terms = [w for w, _ in model_y.get_topic(t)]
    print(f"Year {years[t]} – Topic {t}: {terms}")

# Write year-level CSVs
tw_y    = []
rows_y  = []
freqs_y = model_y.get_topic_freq().set_index("Topic")["Count"]

for t in topic_info_y.Topic:
    if t < 0:
        continue
    for w, wt in model_y.get_topic(t):
        tw_y.append({"group": "year", "topic": t, "word": w, "weight": wt})

for y, pv in zip(years, probs_y):
    d = {"year": y}
    for tid in freqs_y.index:
        if tid < 0:
            continue
        d[f"topic_{tid}"] = float(pv[tid])
    rows_y.append(d)

pd.DataFrame(tw_y).to_csv("bertopic_year_top_words.csv", index=False)
pd.DataFrame(rows_y).to_csv("bertopic_year_doc_probs.csv", index=False)

print("✅ All done — 6 CSVs created.")
