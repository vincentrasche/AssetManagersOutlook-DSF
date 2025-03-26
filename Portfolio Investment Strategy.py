import pandas as pd

# Load the CSV containing average sentiment per firm per year.
df_sentiment = pd.read_csv('avg_sentiment_per_firm_year.csv')

# Group by year and compute the average compound sentiment for each year.
df_yearly = df_sentiment.groupby('year', as_index=False)['compound_sentiment'].mean()

# Function to assign a market regime based on compound sentiment thresholds.
def assign_investment_factor(sentiment):
    if sentiment > 0.15:
        return 'Boom Factors'
    elif sentiment < -0.15:
        return 'Defensive Factors'
    else:
        return 'Neutral Factors'

# Apply the function to the average sentiment of each year.
df_yearly['investment_factor'] = df_yearly['compound_sentiment'].apply(assign_investment_factor)

# Print the resulting DataFrame
print(df_yearly)

# Optionally, you can plot the average sentiment and regime labels over time
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(df_yearly['year'], df_yearly['compound_sentiment'], marker='o', linestyle='-')
for idx, row in df_yearly.iterrows():
    plt.text(row['year'], row['compound_sentiment'], row['investment_factor'], fontsize=8, ha='center')
plt.title('Average Compound Sentiment and Investment Regime per Year')
plt.xlabel('Year')
plt.ylabel('Average Compound Sentiment')
plt.grid(True)
plt.show()
