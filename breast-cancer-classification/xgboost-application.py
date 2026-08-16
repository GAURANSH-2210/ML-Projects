"""
Breast cancer classification
Two changes from my earlier version that affect what the numbers mean:
1. sklearn labels 1=benign, 0=malignant. Every recall/precision figure I
   computed before was therefore measuring performance on the benign class.
   Relabelled so 1 = malignant.
2. Added XGBoost and LightGBM.
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                             classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import cast

X, y_sklearn = load_breast_cancer(
    return_X_y=True,
    as_frame=True
)

X = cast(pd.DataFrame, X)
y_sklearn = cast(pd.Series, y_sklearn)

y = 1 - y_sklearn

print("shape:", X.shape, "malignant:", y.sum(), "benign:", (1 - y).sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

models = {
    "LogReg":       make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVC(rbf)":     make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "RandomForest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "ExtraTrees":   ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "GradBoost":    GradientBoostingClassifier(random_state=42),
    "XGBoost":      XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42, n_jobs=-1),
    "LightGBM":     LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                                   random_state=42, n_jobs=-1, verbose=-1),
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []

for name, model in models.items():
    start = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    rows.append({
        "model": name, "CV_AUC": scores.mean(), "CV_std": scores.std(),
        "test_AUC": roc_auc_score(y_test, proba),
        "acc": accuracy_score(y_test, pred),
        "missed_malignant": fn,      
        "false_alarm": fp,
        "sec": time.time() - start,
    })

print("\n" + pd.DataFrame(rows).sort_values("CV_AUC", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


print("\n" + "=" * 70)
print("THRESHOLD: a missed malignancy is not equal to a false alarm")
print("=" * 70)

best = models["XGBoost"]
proba = best.predict_proba(X_test)[:, 1]

print(f"{'thr':>5} {'missed_malignant(FN)':>21} {'false_alarm(FP)':>16} "
      f"{'recall':>8} {'precision':>10} {'acc':>7}")
for thr in [0.10, 0.20, 0.30, 0.50, 0.70]:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0
    print(f"{thr:>5.2f} {fn:>21} {fp:>16} {recall:>8.4f} {precision:>10.4f} "
          f"{accuracy_score(y_test, pred):>7.4f}")

# permutation importance
print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE - XGBoost (drop in test ROC-AUC when shuffled)")
print("=" * 70)

result = permutation_importance(best, X_test, y_test, n_repeats=30,
                                random_state=42, scoring="roc_auc", n_jobs=-1)
imp = pd.DataFrame({"feature": X.columns,
                    "mean": result["importances_mean"],
                    "std": result["importances_std"]})
print(imp.sort_values("mean", ascending=False).head(10)
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 70)
print("XGBoost native importance")
print("=" * 70)
for kind in ["weight", "gain", "cover"]:
    scores = best.get_booster().get_score(importance_type=kind)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
    print(f"{kind:7s}: {', '.join(k for k, _ in top)}")