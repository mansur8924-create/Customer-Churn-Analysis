"""
Customer Churn Analysis – Data Visualization

Purpose:
    Cleaned e-commerce churn data is visualized across three key dimensions:
    1. Overall churn counts
    2. Churn by contract type
    3. Churn versus customer tenure

Author: Mansur Mohammed
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the cleaned dataset

file_path = r"Telco_Churn_Cleaned.csv"
df = pd.read_csv(file_path)
print(f"✅ Loaded data from '{file_path}' successfully.")


# Step 2: Set visual style

sns.set_theme(style="whitegrid")  # Clean and readable charts


# Step 3: Create dashboard layout

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart 1: Total Churn
sns.countplot(data=df, x='Churn', ax=axes[0], palette='Set2')
axes[0].set_title('Total Customer Churn')

# Chart 2: Churn by Contract Type
sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[1], palette='Set1')
axes[1].set_title('Churn by Contract Type')

# Chart 3: Churn vs Tenure
sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[2], palette='Pastel1')
axes[2].set_title('Churn vs. Tenure (Months)')


# Step 4: Save the charts

plt.tight_layout()
save_path = r"Churn_Analysis_Charts.png"
plt.savefig(save_path)

print(f"✅ Success! Charts saved at '{save_path}'. Ready for analysis or presentation.")
