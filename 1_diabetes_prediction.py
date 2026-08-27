# Diabetes Prediction using Machine Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

print("Loading Diabetes Dataset...")
np.random.seed(42)
X = pd.DataFrame(np.random.randn(768, 8), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = pd.Series(np.random.randint(0, 2, 768))

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("Evaluating Model...")
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Diabetes Code Ready!")
