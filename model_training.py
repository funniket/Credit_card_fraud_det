# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle

# 1. Load the dataset from the data folder
data_path = 'data/card_transdata.csv'
df = pd.read_csv(data_path)

# 2. Check for missing values and handle them if necessary
if df.isnull().sum().sum() > 0:
    print("Missing values found. Dropping rows with missing values.")
    df = df.dropna()
else:
    print("No missing values found.")

# 3. Separate the features and the target variable
X = df.drop('fraud', axis=1)  # All columns except 'fraud'
y = df['fraud']               # The target column

# 4. Normalize numerical features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use in the app
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 5. Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Handle data imbalance using SMOTE on the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 7. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# 8. Evaluate the model on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for fraud (Class 1)

print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# 9. Save the trained model as a .pkl file
with open('fraud_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete. Model saved as 'fraud_model.pkl'.")
