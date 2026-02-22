import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load dataset
df = pd.read_csv("student_stress_dataset.csv")

# Encode target
le = LabelEncoder()
df["Stress_Level"] = le.fit_transform(df["Stress_Level"])

X = df[["Study_Hours", "Sleep_Hours"]]
y = df["Stress_Level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load best model (SVM recommended)
model = joblib.load("models/SVM.pkl")

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# =============================
# 1️⃣ Confusion Matrix
# =============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("reports/confusion_matrix.png")
plt.close()

# =============================
# 2️⃣ ROC Curve (Multi-class)
# =============================
y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test_bin.shape[1]

plt.figure()

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.savefig("reports/roc_curve.png")
plt.close()

# =============================
# 3️⃣ Correlation Heatmap
# =============================
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("reports/correlation_heatmap.png")
plt.close()

print("Visualizations saved in reports/ folder.")