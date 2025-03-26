# Import necessary libraries
import os
import feedparser
import nltk
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# Ensure required NLTK downloads are available
nltk.download("punkt")

# Define paths and models
folder_path = r'C:\Users\hidde\OneDrive\Documenten\NLP Data\GitHub\AssetManagersOutlook-DSF\Yearly data outlooks'
sentiment_pipe = pipeline("text-classification", model="ProsusAI/finbert")
summary_pipe = pipeline("summarization", model="facebook/bart-large-cnn")

# Ensure the folder exists
if not os.path.exists(folder_path):
    print(f"Folder path does not exist: {folder_path}")
else:
    print(f"Reading PDFs from: {folder_path}")

# List to store results
results = []

# Process each PDF file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):  # Process only PDFs
        pdf_path = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")

        try:
            # Read the PDF file
            reader = PdfReader(pdf_path)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

            # Generate a summary using the summarization model
            summary = summary_pipe(text[:1000], max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

            # Compute sentiment using FinBERT
            sentiment = sentiment_pipe(text[:1000])[0]  # Analyze the first 1000 characters for performance

            # Store results
            results.append({
                "filename": filename,
                "sentiment_label": sentiment["label"],
                "sentiment_score": sentiment["score"],
                "summary": summary
            })

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Display results
for result in results:
    print("\n-------------------------------")
    print(f"Filename: {result['filename']}")
    print(f"Sentiment: {result['sentiment_label']}, Score: {result['sentiment_score']}")
    print(f"Summary: {result['summary']}")
    print("--------------------------------")
