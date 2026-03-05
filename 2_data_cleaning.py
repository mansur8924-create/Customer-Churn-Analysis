"""
PROJECT: Customer Churn Analysis
FILE: 2_data_cleaning.py
AUTHOR: Mansur Mohammed
DESCRIPTION: This script takes the raw e-commerce data, removes null values, 
and prepares the 'Churn' column for machine learning.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Golden Data
file_path = r"C:\Users\mansu\OneDrive\Desktop\Data Analyst Boot Camp\Data Analyst Projects\Customer_Churn_Project\Telco_Churn_Cleaned.csv"
df = pd.read_csv(file_path)

print("Drawing and SAVING the charts...")

# 2. Set the Visual Style
sns.set_theme(style="whitegrid")

# 3. Create a Dashboard Layout (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- CHART 1: Total Churn ---
sns.countplot(data=df, x='Churn', ax=axes[0], palette='Set2')
axes[0].set_title('Total Customer Churn')

# --- CHART 2: Churn by Contract ---
sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[1], palette='Set1')
axes[1].set_title('Churn by Contract Type')

# --- CHART 3: Churn vs. Tenure ---
sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[2], palette='Pastel1')
axes[2].set_title('Churn vs. Tenure (Months)')

# --- STEP 4: The Professional Save ---
# WHAT: We save the chart as a PNG image file.
# WHY: This avoids the 'freezing' issue and gives you a file for your portfolio.
plt.tight_layout()
save_path = r"C:\Users\mansu\OneDrive\Desktop\Data Analyst Boot Camp\Data Analyst Projects\Customer_Churn_Project\Churn_Analysis_Charts.png"
plt.savefig(save_path)


print(f"Success! Charts saved at: {save_path}")
