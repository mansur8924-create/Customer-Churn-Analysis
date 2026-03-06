"""
Customer Churn Prediction – Logistic Regression

Purpose:
    Load cleaned e-commerce data, train a logistic regression model to predict churn,
    evaluate performance, and identify the most influential features.

Author: Mansur Mohammed
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load the ML-ready dataset

file_path = r"Telco_Churn_ML_Ready.csv"
df = pd.read_csv(file_path)
print(f"✅ Loaded data from '{file_path}' successfully.")

# Step 2: Prepare Features (X) and Target (y)
# Features are all columns except 'Churn', target is 'Churn'
X = df.drop('Churn', axis=1)
y = df['Churn']

# Step 3: Split Data into Training and Test Sets
# Train on 80% of the data, test on 20% to check model performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data split: {X_train.shape[0]} training rows, {X_test.shape[0]} testing rows.")

# Step 4: Train the Logistic Regression Model
print("Training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Step 5: Evaluate Model Performance
predictions = model.predict(X_test)
print("\n--- Model Performance Report ---")
print(classification_report(y_test, predictions))

# Step 6: Inspect Feature Importance
# Logistic regression coefficients indicate which features have the strongest impact on churn
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
})

print("\n--- Top 5 Features Influencing Churn ---")
print(importance.sort_values(by='Importance', ascending=False).head(5))
