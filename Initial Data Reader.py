import os
import re
import pickle
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from pdfminer.high_level import extract_text
import pysentiment2 as ps
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# ─── 1) NLTK setup ─────────────────────────────────────────────────────────────
nltk.download('punkt')

# ─── 2) File paths & patterns ─────────────────────────────────────────────────
FOLDER  = 'Yearly data outlooks'       # your PDF folder
CACHE   = 'report_data_last.pkl'
pattern = re.compile(r'(\d+)_([A-Za-z]+)_?(\d{4})\.pdf')

# ─── 3) Sentiment & summarizer setup ──────────────────────────────────────────
# Loughran-McDonald lexicon
lm = ps.LM()

# FinBERT sentiment pipeline
finbert_tok   = BertTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipe  = pipeline(
    "sentiment-analysis",
    model=finbert_model,
    tokenizer=finbert_tok
)

# General summarization pipeline
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# ─── 4) Asset / region groups ──────────────────────────────────────────────────
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

# ─── 5) Helper functions ───────────────────────────────────────────────────────
def finbert_score(label):
    return 1 if label.lower()=="positive" else -1 if label.lower()=="negative" else 0

def extract_sentences(text, terms):
    return [s for s in sent_tokenize(text) if any(t in s.lower() for t in terms)]

def summarize_text(txt, max_len=150, min_len=40):
    words = txt.split()
    snippet = txt if len(words) <= 500 else " ".join(words[:500])
    try:
        summary = summarizer(snippet, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
    except:
        summary = " ".join(sent_tokenize(snippet)[:2])
    return summary

# ─── 6) Load or regenerate cache ───────────────────────────────────────────────
def parse_all_pdfs():
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
            print(f"❌ Failed to extract {fn}: {e}")
            continue

        # Loughran-McDonald sentiment
        toks = lm.tokenize(txt)
        scores = lm.get_score(toks)
        pos, neg = scores.get("Positive",0), scores.get("Negative",0)
        comp = (pos-neg)/(pos+neg) if (pos+neg)>0 else 0

        # Term counts
        tl = txt.lower()
        asset_counts  = {g: sum(tl.count(t) for t in terms) for g,terms in ASSETS.items()}
        region_counts = {g: sum(tl.count(t) for t in terms) for g,terms in REGIONS.items()}

        # Per-asset FinBERT sentiment
        aspect_sents = {}
        for g, terms in ASSETS.items():
            sents = extract_sentences(txt, terms)
            scs = []
            for s in sents:
                try:
                    r = finbert_pipe(s, truncation=True)[0]
                    scs.append(finbert_score(r["label"]) * r["score"])
                except:
                    pass
            aspect_sents[g] = sum(scs)/len(scs) if scs else None

        # Per-region FinBERT sentiment
        region_sents = {}
        for g, terms in REGIONS.items():
            sents = extract_sentences(txt, terms)
            scs = []
            for s in sents:
                try:
                    r = finbert_pipe(s, truncation=True)[0]
                    scs.append(finbert_score(r["label"]) * r["score"])
                except:
                    pass
            region_sents[g] = sum(scs)/len(scs) if scs else None

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
            "aspect_sentiments": aspect_sents,
            "region_sentiments": region_sents
        })
    return pd.DataFrame(rows)

# Load cache if valid, else re-parse
if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        df = pickle.load(f)
    print(f"✔ Loaded {len(df)} rows from cache.")
    # Validate 'text' column
    if "text" not in df.columns or df["text"].astype(str).str.strip().eq("").all():
        print("⚠️ Cache invalid (no text). Re-parsing PDFs.")
        os.remove(CACHE)
        df = parse_all_pdfs()
        with open(CACHE, "wb") as f:
            pickle.dump(df, f)
        print(f"✔ Re-parsed & cached {len(df)} docs.")
else:
    df = parse_all_pdfs()
    with open(CACHE, "wb") as f:
        pickle.dump(df, f)
    print(f"✔ Parsed & cached {len(df)} docs.")

# ─── 7) Expand & save CSVs ────────────────────────────────────────────────────
asp_df = df["aspect_sentiments"].apply(pd.Series)
pdf    = pd.concat([df.drop(columns=["aspect_sentiments"]), asp_df], axis=1)
pdf.to_csv("report_data_expanded.csv", index=False)

reg_df = df["region_sentiments"].apply(pd.Series)
rdf    = pd.concat([df.drop(columns=["region_sentiments"]), reg_df], axis=1)
rdf.to_csv("report_data_expanded_regions.csv", index=False)
print("✔ Wrote expanded CSVs.")

# ─── 8) BERTopic –– Firm‐level ───────────────────────────────────────────────
firm_groups = (
    df.groupby("firm")["text"]
      .apply(lambda ts: " ".join(ts.dropna()))
      .reset_index()
)
firms  = firm_groups["firm"].tolist()
docs_f = firm_groups["text"].tolist()

print(f"\nBuilding BERTopic on {len(firms)} firms…")
model_f = BERTopic(nr_topics=10,
                   calculate_probabilities=True,
                   vectorizer_model=CountVectorizer(stop_words="english"))
topics_f, probs_f = model_f.fit_transform(docs_f)

print("\n–– Firm‐Level Topics ––")
info_f = model_f.get_topic_info()
print(info_f.to_string())
for t in info_f.Topic:
    if t < 0: continue
    terms = [w for w,_ in model_f.get_topic(t)]
    print(f"Topic {t}: {terms}")

# Save firm‐level CSVs
tw_f    = []
rows_f  = []
freqs_f = model_f.get_topic_freq().set_index("Topic")["Count"]
for t in info_f.Topic:
    if t<0: continue
    for w,wt in model_f.get_topic(t):
        tw_f.append({"group":"firm","topic":t,"word":w,"weight":wt})
for firm, pv in zip(firms, probs_f):
    row = {"firm": firm}
    for tid in freqs_f.index:
        if tid<0: continue
        row[f"topic_{tid}"] = float(pv[tid])
    rows_f.append(row)

pd.DataFrame(tw_f).to_csv("bertopic_firm_top_words.csv", index=False)
pd.DataFrame(rows_f).to_csv("bertopic_firm_doc_probs.csv", index=False)

# ─── 9) BERTopic –– Year‐level ────────────────────────────────────────────────
years  = sorted(df["year"].dropna().astype(int).unique())
docs_y = [" ".join(df[df["year"]==y]["text"].dropna().tolist()) for y in years]

print(f"\nBuilding BERTopic on {len(years)} years…")
model_y = BERTopic(nr_topics=10,
                   calculate_probabilities=True,
                   vectorizer_model=CountVectorizer(stop_words="english"))
topics_y, probs_y = model_y.fit_transform(docs_y)

print("\n–– Year‐Level Topics ––")
info_y = model_y.get_topic_info()
print(info_y.to_string())
for t in info_y.Topic:
    if t<0: continue
    terms = [w for w,_ in model_y.get_topic(t)]
    print(f"Year {years[t]} – Topic {t}: {terms}")

# Save year‐level CSVs
tw_y    = []
rows_y  = []
freqs_y = model_y.get_topic_freq().set_index("Topic")["Count"]
for t in info_y.Topic:
    if t<0: continue
    for w,wt in model_y.get_topic(t):
        tw_y.append({"group":"year","topic":t,"word":w,"weight":wt})
for y, pv in zip(years, probs_y):
    row = {"year": y}
    for tid in freqs_y.index:
        if tid<0: continue
        row[f"topic_{tid}"] = float(pv[tid])
    rows_y.append(row)

pd.DataFrame(tw_y).to_csv("bertopic_year_top_words.csv", index=False)
pd.DataFrame(rows_y).to_csv("bertopic_year_doc_probs.csv", index=False)

print("\n✅ All done — 6 CSVs created.")
