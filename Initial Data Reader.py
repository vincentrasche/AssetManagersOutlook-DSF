import os
import re
import json
import pickle
from pdfminer.high_level import extract_text
import pysentiment2 as ps
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from bertopic import BERTopic
from collections import Counter
import torch
import matplotlib.pyplot as plt
import numpy as np
import ast

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define folder and filename pattern
folder_path = 'Yearly data outlooks'
filename_pattern = re.compile(r'(\d+)_([A-Za-z]+)_?(\d{4})')

cache_file = "report_datas.pkl"

# Initialize FinBERT and summarization pipelines
finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize LM sentiment analyzer
lm = ps.LM()

# Define asset and region terms/groups
asset_terms = [
    "equities", "stocks", "fixed income", "bonds", "real estate",
    "commodities", "alternatives", "infrastructure", "private equity",
    "hedge funds", "cash"
]
region_terms = ["europe", "us", "united states", "america", "asia", "china", "japan", "emerging markets", "eu", "eurozone", "india"]

asset_class_groups = {
    "Equity": ["equities", "stocks"],
    "Fixed Income": ["fixed income", "bonds"],
    "Real Estate": ["real estate"],
    "Commodities": ["commodities"],
    "Alternatives": ["alternatives"],
    "Infrastructure": ["infrastructure"],
    "Private Equity": ["private equity"],
    "Hedge Funds": ["hedge funds"],
    "Cash": ["cash"]
}

region_groups = {
    "USA": ["us", "united states", "america", "usa", "united states of america"],
    "Europe": ["europe", "eu", "eurozone", "germany", "france"],
    "Asia": ["asia", "china", "japan", "india"],
    "Emerging Markets": ["emerging markets"]
}

# Helper: Convert FinBERT label to numeric score.
def finbert_score(label):
    if label.lower() == "positive":
        return 1
    elif label.lower() == "negative":
        return -1
    else:
        return 0

# Helper: Sliding-window FinBERT sentiment for long texts.
def finbert_sentiment_long(text, window_length=512, overlap=50):
    encoding = finbert_tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoding['input_ids'][0].tolist()
    attention_mask = encoding['attention_mask'][0].tolist()
    total_len = len(input_ids)
    chunk_length = window_length - 2  # Reserve space for [CLS] and [SEP]
    proba_list = []
    start = 0
    while start < total_len:
        end = start + chunk_length
        if end > total_len:
            end = total_len
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        # Add special tokens.
        chunk_ids = [finbert_tokenizer.cls_token_id] + chunk_ids + [finbert_tokenizer.sep_token_id]
        chunk_mask = [1] + chunk_mask + [1]
        pad_length = window_length - len(chunk_ids)
        if pad_length > 0:
            chunk_ids += [finbert_tokenizer.pad_token_id] * pad_length
            chunk_mask += [0] * pad_length
        input_dict = {
            'input_ids': torch.tensor([chunk_ids]),
            'attention_mask': torch.tensor([chunk_mask])
        }
        outputs = finbert_model(**input_dict)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        proba_list.append(probabilities)
        if end >= total_len:
            break
        start = end - overlap
    with torch.no_grad():
        stacked = torch.stack(proba_list).squeeze(1)
        mean_probs = stacked.mean(dim=0)
    labels = finbert_model.config.id2label
    aggregated = {labels[i].lower(): mean_probs[i].item() for i in range(len(mean_probs))}
    return aggregated

# Helper: Extract sentences containing any given terms.
def extract_sentences(text, terms):
    sentences = sent_tokenize(text)
    matching = [sent for sent in sentences if any(term.lower() in sent.lower() for term in terms)]
    return matching

# Helper: Summarize text to 1-4 sentences.
def summarize_text(text, max_length=150, min_length=40):
    words = text.split()
    if len(words) > 500:
        text = " ".join(words[:500])
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return " ".join(sent_tokenize(text)[:2])

# Process PDFs (or load from cache)
if os.path.exists(cache_file):
    print("Loading cached data from", cache_file)
    with open(cache_file, "rb") as f:
        df_results = pickle.load(f)
else:
    print("No cache found. Processing PDFs...")
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            match = filename_pattern.match(filename)
            if match:
                number, firm, year = match.groups()
                filepath = os.path.join(folder_path, filename)
                try:
                    text = extract_text(filepath)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
                # Build document result; store summary instead of full text.
                doc_result = {
                    'firm': firm,
                    'year': int(year),
                    'filename': filename,
                    'summary': summarize_text(text)
                }
                # Overall sentiment using LM lexicon.
                tokens = lm.tokenize(text)
                score = lm.get_score(tokens)
                positive_count = score.get('Positive', 0)
                negative_count = score.get('Negative', 0)
                total_tokens = score.get('Total', 1)
                compound_score = (positive_count - negative_count) / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
                doc_result.update({
                    'positive': positive_count,
                    'negative': negative_count,
                    'compound_sentiment': compound_score,
                    'total_tokens': total_tokens
                })
                text_lower = text.lower()
                # Grouped counts for asset classes and regions.
                grouped_asset_counts = {group: sum(text_lower.count(term) for term in terms)
                                        for group, terms in asset_class_groups.items()}
                grouped_region_counts = {group: sum(text_lower.count(term) for term in terms)
                                         for group, terms in region_groups.items()}
                doc_result.update({
                    'asset_counts': grouped_asset_counts,
                    'region_counts': grouped_region_counts
                })
                # Aspect-based sentiment analysis for asset groups.
                aspect_sentiments = {}
                for group, terms in asset_class_groups.items():
                    sentences_group = extract_sentences(text, terms)
                    if sentences_group:
                        scores = []
                        for sent in sentences_group:
                            try:
                                res = finbert_pipeline(sent, truncation=True)[0]
                                numeric = finbert_score(res['label']) * res['score']
                                scores.append(numeric)
                            except Exception as e:
                                print(f"Error processing FinBERT for {filename} on group {group} for sentence: {sent[:30]}: {e}")
                        avg_score = sum(scores) / len(scores) if scores else None
                    else:
                        avg_score = None
                    aspect_sentiments[group] = avg_score
                doc_result['aspect_sentiments'] = aspect_sentiments

                # NEW: Region-based sentiment analysis.
                region_sentiments = {}
                for region, terms in region_groups.items():
                    sentences_region = extract_sentences(text, terms)
                    if sentences_region:
                        scores = []
                        for sent in sentences_region:
                            try:
                                res = finbert_pipeline(sent, truncation=True)[0]
                                numeric = finbert_score(res['label']) * res['score']
                                scores.append(numeric)
                            except Exception as e:
                                print(f"Error processing FinBERT for {filename} on region {region} for sentence: {sent[:30]}: {e}")
                        avg_score = sum(scores) / len(scores) if scores else None
                    else:
                        avg_score = None
                    region_sentiments[region] = avg_score
                doc_result['region_sentiments'] = region_sentiments

                # Optional: Named Entity Extraction using NLTK.
                tokens_doc = word_tokenize(text)
                pos_tags = nltk.pos_tag(tokens_doc)
                ne_tree = nltk.ne_chunk(pos_tags, binary=False)
                entities = []
                for subtree in ne_tree:
                    if hasattr(subtree, 'label'):
                        entity = " ".join([leaf[0] for leaf in subtree.leaves()])
                        label = subtree.label()
                        if label in ['GPE', 'ORGANIZATION']:
                            entities.append(entity)
                doc_result['named_entities'] = dict(Counter(entities))
                results.append(doc_result)
    df_results = pd.DataFrame(results)
    # Save cache to avoid reprocessing PDFs.
    with open(cache_file, "wb") as f:
        pickle.dump(df_results, f)
    print("Processed PDFs and saved cache to", cache_file)

# Remove full raw text if it exists (we keep summary).
if 'text' in df_results.columns:
    df_results = df_results.drop(columns=['text'])

print("Detailed Document Results:")
print(df_results.head())

# Export CSV (report_data.csv) with summary and key info.
csv_output = "report_data.csv"
df_results.to_csv(csv_output, index=False)
print(f"CSV file '{csv_output}' has been written.")

# Aggregate overall sentiment per firm per year.
aggregation_fields = ['positive', 'negative', 'compound_sentiment', 'total_tokens']
df_avg = df_results.groupby(['firm', 'year'], as_index=False)[aggregation_fields].mean()
csv_avg_output = "avg_sentiment_per_firm_year.csv"
df_avg.to_csv(csv_avg_output, index=False)
print(f"CSV file '{csv_avg_output}' has been written.")

# Expand aspect_sentiments into separate columns.
df_results['aspect_sentiments'] = df_results['aspect_sentiments'].apply(lambda x: x if isinstance(x, dict) else {})
aspect_df = df_results['aspect_sentiments'].apply(pd.Series)
df_expanded = pd.concat([df_results.drop(columns=['aspect_sentiments']), aspect_df], axis=1)

# You may want to save the expanded aspect data as well, if needed:
df_expanded.to_csv("report_data_expanded.csv", index=False)
print("CSV file 'report_data_expanded.csv' has been written.")

# (Optional) Similarly, expand region_sentiments if you wish to analyze these further.
df_results['region_sentiments'] = df_results['region_sentiments'].apply(lambda x: x if isinstance(x, dict) else {})
region_df = df_results['region_sentiments'].apply(pd.Series)
df_expanded_regions = pd.concat([df_results.drop(columns=['region_sentiments']), region_df], axis=1)
df_expanded_regions.to_csv("report_data_expanded_regions.csv", index=False)
print("CSV file 'report_data_expanded_regions.csv' has been written.")

# --- Topic Modeling with BERTopic on summaries ---
if 'summary' in df_results.columns:
    docs = df_results['summary'].tolist()
else:
    print("Warning: 'summary' column not found. Using full text instead (if available).")
    if 'text' in df_results.columns:
        docs = df_results['text'].tolist()
    else:
        raise KeyError("Neither 'summary' nor 'text' columns are found in the data.")

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
topic_info = topic_model.get_topic_info()
print("Topic Information:")
print(topic_info)
topic_info.to_csv("topic_info.csv", index=False)
print("Topic information saved to 'topic_info.csv'.")
