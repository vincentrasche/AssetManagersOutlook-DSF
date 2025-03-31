import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV containing average sentiment per firm per year.
df_sentiment = pd.read_csv('avg_sentiment_per_firm_year.csv')

# Group by both firm and year to compute the average compound sentiment for each firm-year.
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

# Plot the average sentiment over time for each firm, and annotate with regime labels.
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
