import os
import re
import json
from pdfminer.high_level import extract_text
import pysentiment2 as ps
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from bertopic import BERTopic
from collections import Counter
import torch

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define the folder containing your PDF reports
folder_path = 'Yearly data outlooks'

# Regex pattern to extract number, firm name, and year from filenames like "123_FirmName_2020.pdf"
filename_pattern = re.compile(r'(\d+)_([A-Za-z]+)_?(\d{4})')

# Initialize the Loughran-McDonald sentiment analyzer from pysentiment2
lm = ps.LM()

# Define individual terms for asset classes and regions
asset_terms = [
    "equities", "stocks", "fixed income", "bonds", "real estate",
    "commodities", "alternatives", "infrastructure", "private equity",
    "hedge funds", "cash"
]
region_terms = ["europe", "us", "united states", "america", "asia", "china", "japan", "emerging markets", "eu",
                "eurozone", "india"]

# Define grouping dictionaries for asset classes and regions
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
    "Europe": ["europe", "eu", "eurozone", "germany","france"],
    "Asia": ["asia", "china", "japan", "india"],
    "Emerging Markets": ["emerging markets"]
}

# Initialize FinBERT using Hugging Face Transformers
finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)


# Helper function: convert FinBERT label to numeric score
def finbert_score(label):
    if label.lower() == "positive":
        return 1
    elif label.lower() == "negative":
        return -1
    else:
        return 0


# Helper function: sliding window sentiment analysis for long texts
def finbert_sentiment_long(text, window_length=512, overlap=50):
    """
    Splits the text into overlapping chunks (if necessary), adds special tokens,
    processes each chunk with FinBERT, and aggregates the probabilities.
    Returns the aggregated probabilities dictionary.
    """
    # Tokenize the text without special tokens
    encoding = finbert_tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoding['input_ids'][0].tolist()  # list of token ids
    attention_mask = encoding['attention_mask'][0].tolist()
    total_len = len(input_ids)

    # Set effective chunk length: we reserve 2 tokens for [CLS] and [SEP]
    chunk_length = window_length - 2
    proba_list = []
    start = 0
    while start < total_len:
        end = start + chunk_length
        if end > total_len:
            end = total_len
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        # Add special tokens: [CLS] and [SEP]
        chunk_ids = [finbert_tokenizer.cls_token_id] + chunk_ids + [finbert_tokenizer.sep_token_id]
        chunk_mask = [1] + chunk_mask + [1]
        # Pad if needed
        pad_length = window_length - len(chunk_ids)
        if pad_length > 0:
            chunk_ids += [finbert_tokenizer.pad_token_id] * pad_length
            chunk_mask += [0] * pad_length
        # Create input dict
        input_dict = {
            'input_ids': torch.tensor([chunk_ids]),
            'attention_mask': torch.tensor([chunk_mask])
        }
        # Process chunk
        outputs = finbert_model(**input_dict)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        proba_list.append(probabilities)
        if end >= total_len:
            break
        start = end - overlap  # move window with overlap

    # Aggregate probabilities: stack and take mean along chunk dimension
    with torch.no_grad():
        stacked = torch.stack(proba_list).squeeze(1)  # shape: (num_chunks, num_classes)
        mean_probs = stacked.mean(dim=0)
    # Convert tensor to dictionary with labels
    labels = finbert_model.config.id2label
    aggregated = {labels[i].lower(): mean_probs[i].item() for i in range(len(mean_probs))}
    return aggregated


# Function to extract sentences containing any of the given terms
def extract_sentences(text, terms):
    sentences = sent_tokenize(text)
    matching = [sent for sent in sentences if any(term.lower() in sent.lower() for term in terms)]
    return matching


# Prepare a list to hold results for each document
results = []

# Process each PDF file using pdfminer.six
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

            # Create base result dictionary
            doc_result = {
                'firm': firm,
                'year': int(year),
                'filename': filename,
                'text': text  # full raw text
            }

            # Sentiment analysis using pysentiment2's LM lexicon
            tokens = lm.tokenize(text)
            score = lm.get_score(tokens)
            positive_count = score.get('Positive', 0)
            negative_count = score.get('Negative', 0)
            total_tokens = score.get('Total', 1)
            compound_score = (positive_count - negative_count) / (positive_count + negative_count) if (
                                                                                                                  positive_count + negative_count) > 0 else 0

            doc_result.update({
                'positive': positive_count,
                'negative': negative_count,
                'compound_sentiment': compound_score,
                'total_tokens': total_tokens
            })

            text_lower = text.lower()

            # Count individual asset class and region mentions
            asset_counts = {term: text_lower.count(term) for term in asset_terms}
            region_counts = {term: text_lower.count(term) for term in region_terms}

            # Group counts for asset classes and regions
            grouped_asset_counts = {group: sum(text_lower.count(term) for term in terms)
                                    for group, terms in asset_class_groups.items()}
            grouped_region_counts = {group: sum(text_lower.count(term) for term in terms)
                                     for group, terms in region_groups.items()}

            doc_result.update({
                'asset_counts': asset_counts,
                'grouped_asset_counts': grouped_asset_counts,
                'region_counts': region_counts,
                'grouped_region_counts': grouped_region_counts
            })

            # Aspect-Based Sentiment Analysis using FinBERT on asset groups
            aspect_sentiments = {}
            for group, terms in asset_class_groups.items():
                sentences_group = extract_sentences(text, terms)
                if sentences_group:
                    # Combine sentences into one text block (or process individually)
                    combined_text = " ".join(sentences_group)
                    try:
                        # Use the sliding-window function to get aggregated probabilities
                        agg_probs = finbert_sentiment_long(combined_text, window_length=512, overlap=50)
                        # Map the probabilities to a numeric score: here we choose (Positive - Negative)
                        score_value = agg_probs.get("positive", 0) - agg_probs.get("negative", 0)
                    except Exception as e:
                        print(f"Error processing FinBERT for {filename} on group {group}: {e}")
                        score_value = None
                else:
                    score_value = None
                aspect_sentiments[group] = score_value
            doc_result['aspect_sentiments'] = aspect_sentiments

            # Extract ranking-related sentences
            ranking_indicators = ["rank", "ranking", "top", "best", "first", "second", "third", "fourth", "fifth"]
            sentences_all = sent_tokenize(text)
            ranking_sentences = [sent.strip() for sent in sentences_all if
                                 any(term in sent.lower() for term in asset_terms) and
                                 any(rk in sent.lower() for rk in ranking_indicators)]
            doc_result['ranking_sentences'] = ranking_sentences

            # Advanced Analysis: Named entity extraction using NLTK
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

# Create a DataFrame with the results
df_results = pd.DataFrame(results)
print("Detailed Document Results:")
print(df_results.head())

# Convert nested dictionary/list columns to JSON strings for CSV storage
cols_to_convert = ['asset_counts', 'grouped_asset_counts', 'region_counts',
                   'grouped_region_counts', 'ranking_sentences', 'named_entities', 'aspect_sentiments']
for col in cols_to_convert:
    df_results[col] = df_results[col].apply(json.dumps)

csv_output = "report_data.csv"
df_results.to_csv(csv_output, index=False)
print(f"CSV file '{csv_output}' has been written.")

# Aggregate average sentiment counts per firm per year
aggregation_fields = ['positive', 'negative', 'compound_sentiment', 'total_tokens']
df_avg = df_results.groupby(['firm', 'year'], as_index=False)[aggregation_fields].mean()
csv_avg_output = "avg_sentiment_per_firm_year.csv"
df_avg.to_csv(csv_avg_output, index=False)
print(f"CSV file '{csv_avg_output}' has been written.")

# --- Topic Modeling with BERTopic ---
docs = df_results['text'].tolist()
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
topic_info = topic_model.get_topic_info()
print("Topic Information:")
print(topic_info)
topic_info.to_csv("topic_info.csv", index=False)
print("Topic information saved to 'topic_info.csv'.")
