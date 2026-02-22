import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Load dataset
df = pd.read_csv("student_stress_dataset.csv")

# Encode target variable
le = LabelEncoder()
df["Stress_Level"] = le.fit_transform(df["Stress_Level"])

X = df[["Study_Hours", "Sleep_Hours"]]
y = df["Stress_Level"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    cv_score = cross_val_score(pipeline, X, y, cv=5).mean()

    results.append([name, acc, prec, rec, f1, cv_score])

    # Save model
    joblib.dump(pipeline, f"models/{name.replace(' ', '_')}.pkl")

# Display results
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Cross-Val Score"
])

print("\nModel Comparison Results:\n")
print(results_df.sort_values(by="Accuracy", ascending=False))
