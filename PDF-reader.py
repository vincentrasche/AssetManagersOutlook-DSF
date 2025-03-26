import pip
pip.main(['install','pypdf2'])
pip.main(['install','pandas'])
pip.main(['install','feedparser'])
pip.main(['install','nltk'])
pip.main(['install','transformers'])
pip.main(['install','tenserflow'])

import os
import re
import json
import pandas as pd
import PyPDF2
import feedparser
import nltk
from PyPDF2 import PdfReader
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-classification", model="ProsusAI/finbert")


# Ensure the following NLTK downloads are available
nltk.download("vader_lexicon")  # For sentiment analysis
nltk.download("punkt_tab")  # For sentence tokenization
nltk.download("punkt")

# Path to the folder containing outlook PDFs
folder_path = r'C:\Users\hidde\OneDrive\Documenten\NLP Data\GitHub\AssetManagersOutlook-DSF\Yearly data outlooks'

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Ensure the folder exists
if not os.path.exists(folder_path):
    print(f"Folder path does not exist: {folder_path}")
else:
    print(f"Reading PDFs from: {folder_path}")

# List to store results for all files
results = []

# Loop through every PDF file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):  # Process only PDF files
        pdf_path = os.path.join(folder_path, filename)

        print(f"Processing file: {filename}")

        try:
            # Read the PDF file
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:  # Concatenate text from all pages
                text += page.extract_text()

            # Tokenize text into sentences for a quick summary
            sentences = sent_tokenize(text)
            summary = " ".join(sentences[:3])  # First three sentences

            # Compute sentiment score
            sentiment = sia.polarity_scores(text)

            # Save the results
            results.append({
                "filename": filename,
                "sentiment_score": sentiment,
                "summary": summary
            })

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Display results (you can save/export this data as needed)
for result in results:
    print("\n-------------------------------")
    print(f"Filename: {result['filename']}")
    print(f"Sentiment Score: {result['sentiment_score']}")
    print(f"Summary: {result['summary']}")
    print("--------------------------------")


