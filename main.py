import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance

# wdbc.data has no header row
feature_names = ['radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points','symmetry','fractal_dimension']
columns = ['id', 'diagnosis'] + [f'{f}_{stat}' for stat in ['mean','se','worst'] for f in feature_names]
df = pd.read_csv('wdbc.data', header=None, names=columns)
df = df.drop(columns='id')
y_raw = df['diagnosis']                       # 'M' / 'B'
y = (y_raw == 'M').astype(int)                # 1 = malignant, 0 = benign
X = df.drop(columns='diagnosis')
print(X.shape, y.shape)
df.head()

print(X.info())
print("\nMissing values:", X.isna().sum().sum())
print("\nClass balance:\n", y_raw.value_counts(), "\n", y_raw.value_counts(normalize=True).round(3))
X.describe().T[['mean','std','min','max']].round(2)

fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.countplot(x=y_raw, ax=ax[0]); ax[0].set_title("Diagnosis distribution")
sns.boxplot(x=y_raw, y=X['radius_mean'], ax=ax[1]); ax[1].set_title("Mean radius by class")
plt.tight_layout(); plt.show()
plt.figure(figsize=(14,10))
sns.heatmap(X.corr(), cmap='coolwarm', center=0, square=True, cbar_kws={'shrink':.6})
plt.title("Feature correlation matrix"); plt.show()
sep = X.groupby(y).mean().T
sep['diff'] = (sep[1]-sep[0]).abs() / X.std()
sep.sort_values('diff', ascending=False).head(10).round(3)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42)
}

for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred, prob = m.predict(X_te), m.predict_proba(X_te)[:, 1]
    cv = cross_val_score(m, X, y, cv=5, scoring='roc_auc')

    print(name)
    print("CV ROC-AUC:", round(cv.mean(), 3), "+/-", round(cv.std(), 3))
    print("Test ROC-AUC:", round(roc_auc_score(y_te, prob), 3))
    print(classification_report(y_te, pred, target_names=['Benign', 'Malignant']))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    print()

# Permutation importance
rf = models["RandomForest"]
r = permutation_importance(rf, X_te, y_te, n_repeats=30, random_state=42, scoring='roc_auc', n_jobs=-1)
imp = pd.DataFrame({'feature': X.columns, 'mean': r['importances_mean'], 'std': r['importances_std']})
imp = imp.sort_values('mean', ascending=False)
top = imp.head(15).iloc[::-1]
plt.figure(figsize=(8,6))
plt.barh(top['feature'], top['mean'], xerr=top['std'], color='steelblue')
plt.xlabel("Drop in test ROC-AUC when shuffled")
plt.title("Permutation importance — Random Forest"); plt.tight_layout(); plt.show()
imp.head(15).round(4)