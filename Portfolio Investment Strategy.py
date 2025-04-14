import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load data
df_sentiment = pd.read_csv('avg_sentiment_per_firm_year.csv')
df_report = pd.read_csv('report_data.csv')

# ---------------------------------------------
# 1. Average Sentiment Per Firm Over Time
# ---------------------------------------------
plt.figure(figsize=(12, 5))
for firm in df_sentiment['firm'].unique():
    firm_data = df_sentiment[df_sentiment['firm'] == firm]
    plt.plot(firm_data['year'], firm_data['compound_sentiment'], label=firm)
plt.title("Average Sentiment per Firm per Year")
plt.xlabel("Year")
plt.ylabel("Compound Sentiment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 2. Overall Avg Sentiment with Std Dev
# ---------------------------------------------
df_overall = df_sentiment.groupby("year").agg(
    avg_sentiment=('compound_sentiment', 'mean'),
    std_sentiment=('compound_sentiment', 'std')
).reset_index()

plt.figure(figsize=(12, 5))
plt.plot(df_overall['year'], df_overall['avg_sentiment'], label='Average Sentiment')
plt.fill_between(df_overall['year'],
                 df_overall['avg_sentiment'] - df_overall['std_sentiment'],
                 df_overall['avg_sentiment'] + df_overall['std_sentiment'],
                 alpha=0.2, label='±1 SD')
plt.title("Overall Average Compound Sentiment with Standard Deviation")
plt.xlabel("Year")
plt.ylabel("Sentiment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 3. Asset Sentiment Over Time (If Exists)
# ---------------------------------------------
if 'aspect_sentiments' in df_report.columns:
    df_report['aspect_sentiments'] = df_report['aspect_sentiments'].apply(ast.literal_eval)
    asset_sent = df_report[['year']].join(df_report['aspect_sentiments'].apply(pd.Series))
    df_asset_yearly = asset_sent.groupby('year').mean().reset_index()

    plt.figure(figsize=(12, 5))
    for col in df_asset_yearly.columns[1:]:
        plt.plot(df_asset_yearly['year'], df_asset_yearly[col], label=col)
    plt.title("Asset Sentiment Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Asset Sentiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'asset_sentiments' not found in the dataset — skipping asset sentiment plot.")

# ---------------------------------------------
# 4. Region Sentiment Over Time (If Exists)
# ---------------------------------------------
if 'region_sentiments' in df_report.columns:
    df_report['region_sentiments'] = df_report['region_sentiments'].apply(ast.literal_eval)
    region_sent = df_report[['year']].join(df_report['region_sentiments'].apply(pd.Series))
    df_region_yearly = region_sent.groupby('year').mean().reset_index()

    plt.figure(figsize=(12, 5))
    for col in df_region_yearly.columns[1:]:
        plt.plot(df_region_yearly['year'], df_region_yearly[col], label=col)
    plt.title("Region Sentiment Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Region Sentiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'region_sentiments' not found in the dataset — skipping region sentiment plot.")
