import pandas as pd
import matplotlib.pyplot as plt
import ast

# -------------------------------
# Overall Sentiment Visualization per Firm per Year
# -------------------------------

# Load the CSV containing average sentiment per firm per year.
df_sentiment = pd.read_csv('avg_sentiment_per_firm_year.csv')

# Group by both firm and year to compute the average compound sentiment.
df_firm_year = df_sentiment.groupby(['firm', 'year'], as_index=False)['compound_sentiment'].mean()

# Function to assign a market regime based on compound sentiment thresholds.
def assign_investment_factor(sentiment):
    if sentiment > 0.1:
        return 'Boom Forecast'
    elif sentiment < -0.1:
        return 'Defensive Forecast'
    else:
        return 'Neutral Forecast'

# Apply the function to assign a regime for each firm-year observation.
df_firm_year['investment_factor'] = df_firm_year['compound_sentiment'].apply(assign_investment_factor)

# Print the resulting DataFrame.
print(df_firm_year)

# Plot the average sentiment over time for each firm, with regime annotations.
plt.figure(figsize=(12, 6))
firms = df_firm_year['firm'].unique()
for firm in firms:
    df_firm = df_firm_year[df_firm_year['firm'] == firm]
    plt.plot(df_firm['year'], df_firm['compound_sentiment'], marker='o', linestyle='-', label=firm)
    for idx, row in df_firm.iterrows():
        plt.text(row['year'], row['compound_sentiment'], row['investment_factor'],
                 fontsize=8, ha='center', va='bottom')
plt.title('Average Compound Sentiment and Investment Regime per Year by Company')
plt.xlabel('Year')
plt.ylabel('Average Compound Sentiment')
plt.grid(True)
plt.legend(title='Firm')
plt.show()

# -------------------------------
# Asset Mentions Visualization
# -------------------------------

# Load the detailed report data CSV.
df = pd.read_csv("report_data.csv")
# Convert the 'asset_counts' column (stored as a string) back to a dictionary.
df["asset_counts"] = df["asset_counts"].apply(ast.literal_eval)

# Create an expanded DataFrame by expanding the asset_counts dictionary into separate columns.
df_expanded = pd.concat([df, df["asset_counts"].apply(pd.Series)], axis=1)

# Create new columns for Equity and Fixed Income mentions.
df_expanded['Equity_mentions'] = df_expanded['asset_counts'].apply(lambda x: x.get("Equity", 0) if isinstance(x, dict) else 0)
df_expanded['Fixed_Income_mentions'] = df_expanded['asset_counts'].apply(lambda x: x.get("Fixed Income", 0) if isinstance(x, dict) else 0)

# Group by year to compute the average mentions per report.
df_year_avg_mentions = df_expanded.groupby("year", as_index=False).agg({
    "Equity_mentions": "mean",
    "Fixed_Income_mentions": "mean"
})
df_year_avg_mentions["Difference"] = df_year_avg_mentions["Equity_mentions"] - df_year_avg_mentions["Fixed_Income_mentions"]

# Line chart: Average asset mentions per report by year.
plt.figure(figsize=(10,6))
plt.plot(df_year_avg_mentions["year"], df_year_avg_mentions["Equity_mentions"], marker="o", label="Equity (Avg Mentions)")
plt.plot(df_year_avg_mentions["year"], df_year_avg_mentions["Fixed_Income_mentions"], marker="s", label="Fixed Income (Avg Mentions)")
plt.xlabel("Year")
plt.ylabel("Average Mentions per Report")
plt.title("Average Asset Mentions by Year (Equity vs Fixed Income)")
plt.legend()
plt.grid(True)
plt.show()

# Stacked bar chart: Total asset mentions by year.
plt.figure(figsize=(10,6))
plt.bar(df_year_avg_mentions["year"], df_year_avg_mentions["Equity_mentions"], label="Equity")
plt.bar(df_year_avg_mentions["year"], df_year_avg_mentions["Fixed_Income_mentions"],
        bottom=df_year_avg_mentions["Equity_mentions"], label="Fixed Income")
plt.xlabel("Year")
plt.ylabel("Average Mentions per Report")
plt.title("Stacked Asset Mentions by Year (Equity vs Fixed Income)")
plt.legend()
plt.grid(axis='y')
plt.show()

# -------------------------------
# Dual-axis Plot: Overall Sentiment vs. Asset Discussion Difference
# -------------------------------

# Aggregate overall compound sentiment across all firms per year.
df_overall = df_sentiment.groupby("year", as_index=False)["compound_sentiment"].mean()

# Merge the overall sentiment with the asset discussion difference.
df_merged = pd.merge(df_overall, df_year_avg_mentions, on="year", how="inner")

fig, ax1 = plt.subplots(figsize=(10,6))
color1 = 'tab:blue'
ax1.set_xlabel("Year")
ax1.set_ylabel("Avg Compound Sentiment", color=color1)
ax1.plot(df_merged["year"], df_merged["compound_sentiment"], marker="o", color=color1, label="Avg Compound Sentiment")
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel("Difference in Mentions (Equity - Fixed Income)", color=color2)
ax2.plot(df_merged["year"], df_merged["Difference"], marker="s", color=color2, label="Difference in Mentions")
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.title("Average Sentiment vs. Asset Discussion Difference by Year")
plt.show()
