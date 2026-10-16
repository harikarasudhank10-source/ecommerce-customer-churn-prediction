import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Load Data
df = pd.read_csv("ecommerce_churn_dataset.csv")
X = df.drop(['Customer_ID', 'Churn_Target'], axis=1)
y = df['Churn_Target']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Model Training
model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. Save Model
joblib.dump(model, 'churn_model.pkl')
print("Model trained and saved as churn_model.pkl!")