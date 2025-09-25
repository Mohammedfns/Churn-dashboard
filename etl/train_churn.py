import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Charger employés
EMP = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"), parse_dates=["hire_date","leave_date","dob"])

# Feature engineering
EMP["left"] = EMP["leave_date"].notna().astype(int)   # 1 si l’employé a quitté
EMP["tenure_days"] = (pd.to_datetime("today") - EMP["hire_date"]).dt.days
EMP["age"] = (pd.to_datetime("today") - EMP["dob"]).dt.days // 365

# Variables catégorielles → numériques
df = EMP[["gender","department","site","contract","salary","tenure_days","age","left"]].copy()
df = pd.get_dummies(df, columns=["gender","department","site","contract"], drop_first=True)

# Features & target
X = df.drop("left", axis=1)
y = df["left"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
print("✔ Rapport de classification :")
print(classification_report(y_test, y_pred))

# Sauvegarde
MODEL_PATH = os.path.join(DATA_DIR, "churn_model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"✔ Modèle sauvegardé dans {MODEL_PATH}")
