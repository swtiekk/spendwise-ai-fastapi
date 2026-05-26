"""
SpendWise AI — Model Training Script
=====================================
Run this once from your project root:
    python ml/train.py

Output:
    ml/cluster_model.pkl        ← K-Means behavior clustering
    ml/risk_model.pkl           ← Decision Tree risk classifier
    ml/sustainability_model.pkl ← Decision Tree sustainability classifier
"""

import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE   # Fix 3: SMOTE for imbalance

# ── Paths ──────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
CSV_PATH           = os.path.join(BASE_DIR, "spendwise_training_data.csv")
CLUSTER_MODEL_PATH = os.path.join(BASE_DIR, "cluster_model.pkl")
RISK_MODEL_PATH    = os.path.join(BASE_DIR, "risk_model.pkl")
SUSTAIN_MODEL_PATH = os.path.join(BASE_DIR, "sustainability_model.pkl")

# ── Load dataset ───────────────────────────────────────────
print("\n" + "="*55)
print("  SpendWise AI — ML Model Training")
print("="*55)

print("\n[1/3] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"      Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"      Cluster labels : {df['cluster_label'].value_counts().to_dict()}")
print(f"      Risk labels    : {df['risk_label'].value_counts().to_dict()}")
print(f"      Sustain labels : {df['sustainability_label'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════
# MODEL 1 — Spending Behavior Clustering (K-Means)
# ══════════════════════════════════════════════════════════
print("\n" + "-"*55)
print("  MODEL 1: Spending Behavior Clustering (K-Means)")
print("-"*55)

CLUSTER_FEATURES = [
    "income_amount",
    "total_expenses",
    "spending_ratio",
    "top_category_amount",
    "num_transactions",
]

X_cluster = df[CLUSTER_FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_scaled)

cluster_df = df.copy()
cluster_df["kmeans_id"] = kmeans.labels_

label_map = {}
for cluster_id in range(4):
    subset       = cluster_df[cluster_df["kmeans_id"] == cluster_id]
    majority     = subset["cluster_label"].value_counts().idxmax()
    label_map[cluster_id] = majority

print(f"  Cluster mapping : {label_map}")

predicted_labels = [label_map[c] for c in kmeans.labels_]
correct          = sum(p == t for p, t in zip(predicted_labels, df["cluster_label"]))
cluster_accuracy = round(correct / len(df) * 100, 2)
print(f"  Label match accuracy : {cluster_accuracy}%")
print(f"  Inertia (lower=better): {round(kmeans.inertia_, 2)}")

cluster_bundle = {
    "model":     kmeans,
    "scaler":    scaler,
    "label_map": label_map,
    "features":  CLUSTER_FEATURES,
    "accuracy":  cluster_accuracy,
}
with open(CLUSTER_MODEL_PATH, "wb") as f:
    pickle.dump(cluster_bundle, f)
print(f"  Saved → {CLUSTER_MODEL_PATH}")


# ══════════════════════════════════════════════════════════
# MODEL 2 — Risk Classification (Decision Tree)
# ══════════════════════════════════════════════════════════
print("\n" + "-"*55)
print("  MODEL 2: Risk Classification (Decision Tree)")
print("-"*55)

RISK_FEATURES = [
    "balance",
    "daily_burn_rate",
    "days_remaining",
    "spending_ratio",
    "savings_goal_progress",
]

X_risk = df[RISK_FEATURES].values
y_risk = df["risk_label"].values

le_risk = LabelEncoder()
y_risk_encoded = le_risk.fit_transform(y_risk)
print(f"  Classes : {list(le_risk.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_risk, y_risk_encoded, test_size=0.2, random_state=42, stratify=y_risk_encoded
)

# Fix 3: Apply SMOTE to handle class imbalance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE → Train samples : {len(X_train)}")

# Fix 2: Reduced complexity to prevent overfitting
risk_model = DecisionTreeClassifier(
    max_depth=4,           # was 8
    min_samples_split=10,  # was 5
    min_samples_leaf=5,    # was 2
    random_state=42,
    class_weight="balanced",
)
risk_model.fit(X_train, y_train)

# Evaluate
y_pred        = risk_model.predict(X_test)
risk_accuracy = accuracy_score(y_test, y_pred)
cv_scores     = cross_val_score(risk_model, X_risk, y_risk_encoded, cv=5)

print(f"  Test accuracy   : {round(risk_accuracy * 100, 2)}%")
print(f"  CV accuracy     : {round(cv_scores.mean() * 100, 2)}% ± {round(cv_scores.std() * 100, 2)}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_risk.classes_))

importances = risk_model.feature_importances_
print("  Feature importances:")
for feat, imp in sorted(zip(RISK_FEATURES, importances), key=lambda x: -x[1]):
    print(f"    {feat:<30} {round(imp * 100, 2)}%")

risk_bundle = {
    "model":    risk_model,
    "encoder":  le_risk,
    "features": RISK_FEATURES,
    "accuracy": round(risk_accuracy * 100, 2),
    "cv_mean":  round(cv_scores.mean() * 100, 2),
}
with open(RISK_MODEL_PATH, "wb") as f:
    pickle.dump(risk_bundle, f)
print(f"\n  Saved → {RISK_MODEL_PATH}")


# ══════════════════════════════════════════════════════════
# MODEL 3 — Sustainability Prediction (Decision Tree)
# ══════════════════════════════════════════════════════════
print("\n" + "-"*55)
print("  MODEL 3: Sustainability Prediction (Decision Tree)")
print("-"*55)

# Fix 1: Removed leaky feature "projected_days_left"
SUSTAIN_FEATURES = [
    "balance",
    "daily_burn_rate",
    "days_remaining",
    "spending_ratio",
]

X_sustain = df[SUSTAIN_FEATURES].values
y_sustain = df["sustainability_label"].values

le_sustain = LabelEncoder()
y_sustain_encoded = le_sustain.fit_transform(y_sustain)
print(f"  Classes : {list(le_sustain.classes_)}")

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sustain, y_sustain_encoded, test_size=0.2, random_state=42, stratify=y_sustain_encoded
)
print(f"  Train samples : {len(X_train_s)}, Test samples : {len(X_test_s)}")

sustain_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
)
sustain_model.fit(X_train_s, y_train_s)

# Evaluate
y_pred_s        = sustain_model.predict(X_test_s)
sustain_accuracy = accuracy_score(y_test_s, y_pred_s)
cv_scores_s      = cross_val_score(sustain_model, X_sustain, y_sustain_encoded, cv=5)

print(f"  Test accuracy   : {round(sustain_accuracy * 100, 2)}%")
print(f"  CV accuracy     : {round(cv_scores_s.mean() * 100, 2)}% ± {round(cv_scores_s.std() * 100, 2)}%")
print(f"\n  Classification Report:")
print(classification_report(y_test_s, y_pred_s, target_names=le_sustain.classes_))

importances_s = sustain_model.feature_importances_
print("  Feature importances:")
for feat, imp in sorted(zip(SUSTAIN_FEATURES, importances_s), key=lambda x: -x[1]):
    print(f"    {feat:<30} {round(imp * 100, 2)}%")

sustain_bundle = {
    "model":    sustain_model,
    "encoder":  le_sustain,
    "features": SUSTAIN_FEATURES,
    "accuracy": round(sustain_accuracy * 100, 2),
    "cv_mean":  round(cv_scores_s.mean() * 100, 2),
}
with open(SUSTAIN_MODEL_PATH, "wb") as f:
    pickle.dump(sustain_bundle, f)
print(f"\n  Saved → {SUSTAIN_MODEL_PATH}")


# ── Summary ────────────────────────────────────────────────
print("\n" + "="*55)
print("  Training Complete!")
print("="*55)
print(f"  Model 1 — Clustering    : {cluster_accuracy}% label match")
print(f"  Model 2 — Risk          : {round(risk_accuracy * 100, 2)}% test accuracy")
print(f"  Model 3 — Sustainability: {round(sustain_accuracy * 100, 2)}% test accuracy")
print("\n  Files saved:")
print(f"    ml/cluster_model.pkl")
print(f"    ml/risk_model.pkl")
print(f"    ml/sustainability_model.pkl")
print("\n  Next: run your FastAPI server!")
print("="*55 + "\n")