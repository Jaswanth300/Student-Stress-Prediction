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
# ==============================
# Feature Engineering
# ==============================

df["Study_Sleep_Ratio"] = df["Study_Hours"] / df["Sleep_Hours"]
df["Sleep_Deficit"] = (df["Sleep_Hours"] < 5).astype(int)
df["Academic_Load_Index"] = df["Study_Hours"] * (10 - df["Sleep_Hours"])

X = df[[
    "Study_Hours",
    "Sleep_Hours",
    "Study_Sleep_Ratio",
    "Sleep_Deficit",
    "Academic_Load_Index"
]]
y = df["Stress_Level"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models dictionary
from sklearn.model_selection import GridSearchCV

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {
            "classifier__C": [0.1, 1, 10]
        }
    ),
    "Decision Tree": (
        DecisionTreeClassifier(),
        {
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [None, 10]
        }
    ),
    "SVM": (
        SVC(probability=True),
        {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"]
        }
    )
}
results = []

for name, (model, param_grid) in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results.append([name, acc, prec, rec, f1, grid.best_score_])

    joblib.dump(best_model, f"models/{name.replace(' ', '_')}.pkl")

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Best CV Score"
])

print("\nTuned Model Comparison Results:\n")
print(results_df.sort_values(by="Accuracy", ascending=False))

    # Save model
joblib.dump(pipeline, f"models/{name.replace(' ', '_')}.pkl")

# Display results
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Cross-Val Score"
])

print("\nModel Comparison Results:\n")
print(results_df.sort_values(by="Accuracy", ascending=False))
