import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# 1. Load the ML-Ready Data
# -------------------------------
file_path = r"Telco_Churn_ML_Ready.csv"
df = pd.read_csv(file_path)
print(f"✅ Loaded data from '{file_path}' successfully.")

# -------------------------------
# 2. Prepare Features and Target
# -------------------------------
# X = features (all columns except 'Churn'), y = target (Churn)
X = df.drop('Churn', axis=1)
y = df['Churn']

# -------------------------------
# 3. Split Data into Training & Test Sets
# -------------------------------
# We train on 80% of the data and test on 20% to see how well the model predicts churn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data split: {X_train.shape[0]} training rows, {X_test.shape[0]} testing rows.")

# -------------------------------
# 4. Train the Logistic Regression Model
# -------------------------------
print("Training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")

# -------------------------------
# 5. Evaluate Model Performance
# -------------------------------
predictions = model.predict(X_test)
print("\n--- Model Performance Report ---")
print(classification_report(y_test, predictions))

# -------------------------------
# 6. Inspect Feature Importance
# -------------------------------
# Coefficients tell us which features most influence churn predictions
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.coef_[0]
})

print("\n--- Top 5 Predictors of Churn ---")
print(importance.sort_values(by='Importance', ascending=False).head(5))

