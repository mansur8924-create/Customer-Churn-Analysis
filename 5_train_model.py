import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the ML-Ready Data
file_path = r"file_path = "Telco_Churn_ML_Ready.csv"
df = pd.read_csv(file_path)

# 2. Split the Data (The 'Exam' Strategy)
# WHAT: X is the features (Contract, Monthly Charges), y is the answer (Churn).
X = df.drop('Churn', axis=1)
y = df['Churn']

# WHY: We split the data 80/20. We let the robot study 80% of the customers, 
# then we 'test' it on the remaining 20% to see if it can guess correctly.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Logistic Regression model...")

# 3. Create and Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Evaluate the Results
predictions = model.predict(X_test)

print("\n--- Model Performance Report ---")
print(classification_report(y_test, predictions))

# 5. Find the 'Smoking Guns'
# WHAT: We look at the 'coefficients' to see which features most strongly predict churn.
importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
print("\n--- Top Predictors of Churn ---")

print(importance.sort_values(by='Importance', ascending=False).head(5))
