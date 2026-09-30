# 🌍 Tourism EDA Analysis

An exploratory data analysis (EDA) project on a simulated international tourism dataset spanning **2018–2023**.

## 📋 Project Overview

This project demonstrates Python-based data analysis and visualization techniques applied to the tourism sector, covering:
- Data generation & import
- Data cleaning (missing values, outlier handling)
- Statistical analysis
- 7 visualizations including time series, boxplots, histograms, heatmaps, and more
- Tourism trend implications

## 📁 File Structure

```
├── tourism_eda.py           # Main EDA Python script
├── raw_tourism_data.csv     # Raw generated dataset
├── cleaned_tourism_data.csv # Cleaned dataset
├── Tourism_EDA_Final.doc    # Comprehensive EDA report (Word document)
├── visitors_time_series.png # Monthly arrivals chart
├── spend_boxplot.png        # Spending distribution by country
├── annual_revenue.png       # Annual revenue bar chart
├── visitors_histogram.png   # Visitor count histogram
├── correlation_heatmap.png  # Correlation matrix
├── yoy_growth.png           # Year-over-year growth chart
├── monthly_seasonality.png  # Monthly seasonality chart
└── README.md
```

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualization |
| `python-docx` | DOC report generation |

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn python-docx

# Run the EDA script
python tourism_eda.py
```

## 📊 Key Findings

- **COVID-19 Impact**: Visitor counts dropped to ~15% of pre-pandemic levels in 2020–2021
- **Recovery**: Strong rebound from 2022, approaching pre-pandemic levels by 2023
- **High-Value Markets**: Japanese tourists spend the most (~$198 USD avg), followed by Australians (~$179 USD)
- **Seasonality**: Visitor peaks consistently in **July–August** across all source countries

## 📄 Dataset

Synthetically generated to mirror real-world tourism patterns. Includes deliberate missing values and outliers to demonstrate data cleaning techniques.

**Reference**: [World Tourism Organization (UNWTO)](https://www.unwto.org/)
