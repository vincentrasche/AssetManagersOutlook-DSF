import pip
pip.main(['install','pdfplumber'])
pip.main(['install','pysentiment2'])
pip.main(['install','pandas'])
pip.main(['install','nltk'])

import os
import re
import json
import pdfplumber
import pysentiment2 as ps
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')

# Define the folder containing your PDF reports
folder_path = 'Yearly data outlooks'

# Regex pattern to extract number, firm name, and year from filenames like "123_FirmName_2020.pdf"
filename_pattern = re.compile(r'(\d+)_([A-Za-z]+)_?(\d{4})')

# Initialize the Loughran-McDonald sentiment analyzer from pysentiment2
lm = ps.LM()

# Define asset classes, regions, and ranking-related keywords for extra extraction
asset_classes = [
    "equities", "stocks", "fixed income", "bonds", "real estate",
    "commodities", "alternatives", "infrastructure", "private equity",
    "hedge funds", "cash"
]
regions = ["europe", "us", "united states", "america", "asia", "china", "japan", "emerging markets"]
ranking_indicators = ["rank", "ranking", "top", "best", "first", "second", "third", "fourth", "fifth"]

results = []

# Process each PDF file in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.pdf'):
        match = filename_pattern.match(filename)
        if match:
            number, firm, year = match.groups()
            filepath = os.path.join(folder_path, filename)

            # Extract text from the PDF using pdfplumber
            try:
                with pdfplumber.open(filepath) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            # Tokenize and get sentiment counts using pysentiment2's LM lexicon
            tokens = lm.tokenize(text)
            score = lm.get_score(tokens)
            # Calculate counts for the different sentiment categories
            positive_count = score.get('Positive', 0)
            negative_count = score.get('Negative', 0)
            litigious_count = score.get('Litigious', 0)
            uncertainty_count = score.get('Uncertainty', 0)
            constraining_count = score.get('Constraining', 0)
            superfluous_count = score.get('Superfluous', 0)
            total_tokens = score.get('Total', 1)  # Avoid division by zero if needed

            # Tokenize text into sentences for additional analysis
            sentences = sent_tokenize(text)
            text_lower = text.lower()

            # Count asset class and region mentions
            asset_counts = {ac: text_lower.count(ac.lower()) for ac in asset_classes}
            region_counts = {region: text_lower.count(region.lower()) for region in regions}

            # Extract ranking-related sentences: sentences mentioning an asset class and a ranking term
            ranking_sentences = []
            for sent in sentences:
                sent_lower = sent.lower()
                if any(ac in sent_lower for ac in asset_classes) and any(rk in sent_lower for rk in ranking_indicators):
                    ranking_sentences.append(sent.strip())

            # Store results for this document
            results.append({
                'firm': firm,
                'year': int(year),
                'filename': filename,
                'positive': positive_count,
                'negative': negative_count,
                'litigious': litigious_count,
                'uncertainty': uncertainty_count,
                'constraining': constraining_count,
                'superfluous': superfluous_count,
                'total_tokens': total_tokens,
                'asset_counts': asset_counts,
                'region_counts': region_counts,
                'ranking_sentences': ranking_sentences
            })

# Create a DataFrame with the results for each document
df_results = pd.DataFrame(results)
print("Detailed Document Results:")
print(df_results)

# Convert nested dictionary/list columns to JSON strings for CSV storage
df_results['asset_counts'] = df_results['asset_counts'].apply(json.dumps)
df_results['region_counts'] = df_results['region_counts'].apply(json.dumps)
df_results['ranking_sentences'] = df_results['ranking_sentences'].apply(json.dumps)

# Save the detailed report data to a CSV file
csv_output = "report_data.csv"
df_results.to_csv(csv_output, index=False)
print(f"CSV file '{csv_output}' has been written.")

# Aggregate average sentiment counts per firm per year:
aggregation_fields = ['positive', 'negative', 'litigious', 'uncertainty', 'constraining', 'superfluous', 'total_tokens']
df_avg = df_results.groupby(['firm', 'year'], as_index=False)[aggregation_fields].mean()
csv_avg_output = "avg_sentiment_per_firm_year.csv"
df_avg.to_csv(csv_avg_output, index=False)
print(f"CSV file '{csv_avg_output}' has been written.")
