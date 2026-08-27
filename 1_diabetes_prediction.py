# Advanced Diabetes Prediction Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

print("Initializing Advanced Pipeline for Diabetes Prediction...")

# 1. Load Data
np.random.seed(42)
X = pd.DataFrame(np.random.randn(768, 8), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = pd.Series(np.random.randint(0, 2, 768))

# 2. EDA (Exploratory Data Analysis)
print("Generating Correlation Heatmap...")
plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Preprocessing & SMOTE
print("Scaling features and applying SMOTE...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning with GridSearchCV
print("Hyperparameter tuning for Random Forest...")
param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# 5. XGBoost Model
print("Training XGBoost Classifier...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# 6. Evaluation
print("--- Random Forest Evaluation ---")
rf_preds = best_rf.predict(X_test)
print(classification_report(y_test, rf_preds))

print("--- XGBoost Evaluation ---")
xgb_preds = xgb.predict(X_test)
print(classification_report(y_test, xgb_preds))

# 7. Model Export
print("Exporting models...")
joblib.dump(best_rf, 'best_rf_diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Pipeline Complete! Models and Plots have been saved.")
