import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ---------------------------------------------------------
# 1. Generate Simulated Tourism Data
# ---------------------------------------------------------
print("Generating simulated tourism dataset...")
dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='ME')
countries = ['USA', 'UK', 'Japan', 'Australia', 'Germany']

data = []
for date in dates:
    for country in countries:
        # Base trend: steady increase over time
        base_visitors = 10000 + (date.year - 2018) * 1000
        
        # Seasonality: higher in summer (mid-year)
        seasonality = np.sin((date.month - 3) * np.pi / 6) * 5000
        
        # Random noise
        noise = np.random.normal(0, 1500)
        
        visitors = int(base_visitors + seasonality + noise)
        
        # Covid impact: Massive drop in 2020 and 2021
        if date.year in [2020, 2021]:
            visitors = int(visitors * 0.15)
            
        # Spending based on country and random variance
        base_spend = {'USA': 150, 'UK': 130, 'Japan': 200, 'Australia': 180, 'Germany': 140}
        spend_per_visitor = np.random.normal(base_spend[country], 20)
        
        data.append({
            'Date': date,
            'Country_of_Origin': country,
            'Visitors': visitors,
            'Avg_Spend_USD': round(spend_per_visitor, 2)
        })

df = pd.DataFrame(data)

# Introduce missing values and outliers to demonstrate data cleaning
df.loc[12, 'Visitors'] = np.nan
df.loc[45, 'Avg_Spend_USD'] = np.nan
df.loc[150, 'Visitors'] = 850000  # Extreme outlier

# Save raw data to CSV
df.to_csv('raw_tourism_data.csv', index=False)
print("Raw data saved to 'raw_tourism_data.csv'")

# ---------------------------------------------------------
# 2. Data Cleaning & Transformation
# ---------------------------------------------------------
print("Starting Data Cleaning...")
# Copy for cleaning
df_clean = df.copy()

# Handle Missing Values
print("Missing values before cleaning:\n", df_clean.isnull().sum())
df_clean['Visitors'] = df_clean['Visitors'].fillna(df_clean['Visitors'].median())
df_clean['Avg_Spend_USD'] = df_clean['Avg_Spend_USD'].fillna(df_clean['Avg_Spend_USD'].mean())

# Handle Outliers (IQR Method for Visitors)
Q1 = df_clean['Visitors'].quantile(0.25)
Q3 = df_clean['Visitors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['Visitors'] < lower_bound) | (df_clean['Visitors'] > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")

# Cap the outliers to the upper bound
df_clean.loc[df_clean['Visitors'] > upper_bound, 'Visitors'] = upper_bound

# ---------------------------------------------------------
# 3. Exploratory Data Analysis & Visualizations
# ---------------------------------------------------------
print("Generating Visualizations...")
sns.set_theme(style="whitegrid")

# Plot 1: Time Series of Tourist Arrivals
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_clean, x='Date', y='Visitors', hue='Country_of_Origin')
plt.title('Monthly Tourist Arrivals by Country (2018-2023)')
plt.ylabel('Number of Visitors')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('visitors_time_series.png')
plt.close()

# Plot 2: Boxplot of Average Spend by Country
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='Country_of_Origin', y='Avg_Spend_USD', hue='Country_of_Origin', palette='Set2', legend=False)
plt.title('Distribution of Average Spending per Visitor by Country')
plt.ylabel('Average Spend (USD)')
plt.xlabel('Country of Origin')
plt.tight_layout()
plt.savefig('spend_boxplot.png')
plt.close()

# Plot 3: Total Revenue over time
df_clean['Total_Revenue'] = df_clean['Visitors'] * df_clean['Avg_Spend_USD']
plt.figure(figsize=(12, 6))
annual = df_clean.groupby(df_clean['Date'].dt.year)['Total_Revenue'].sum().reset_index()
sns.barplot(data=annual, x='Date', y='Total_Revenue', hue='Date', palette='viridis', legend=False)
plt.title('Total Annual Tourism Revenue (2018-2023)')
plt.ylabel('Total Revenue (USD)')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('annual_revenue.png')
plt.close()

# Save cleaned data
df_clean.to_csv('cleaned_tourism_data.csv', index=False)
print("Analysis complete! Visualizations saved as PNGs, and cleaned data as 'cleaned_tourism_data.csv'.")
