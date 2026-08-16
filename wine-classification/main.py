import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
           'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
           'color_intensity', 'hue', 'od280_od315', 'proline']

df = pd.read_csv('wine.data', header=None, names=columns)

X = df.drop(columns='class')
y = df['class'] - 1          # XGBoost needs classes starting at 0

print(X.shape)
print(y.value_counts().sort_index())
df.head()

print(X.describe().T[['mean', 'std', 'min', 'max']].round(2))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x=df['class'], ax=ax[0]); ax[0].set_title("Cultivar distribution")
sns.boxplot(x=df['class'], y=X['flavanoids'], ax=ax[1]); ax[1].set_title("Flavanoids by cultivar")
plt.tight_layout(); plt.savefig('cultivar_distribution.png', dpi=120); plt.close()
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': .6})
plt.title("Feature correlation"); plt.savefig('correlation_heatmap.png', dpi=120); plt.close()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                             objective='multi:softprob', num_class=3,
                             random_state=42, eval_metric='mlogloss')
}

for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    cv = cross_val_score(m, X, y, cv=5, scoring='accuracy')

    print(name)
    print("CV accuracy:", round(cv.mean(), 3), "+/-", round(cv.std(), 3))
    print(classification_report(y_te, pred, target_names=['Cultivar 1', 'Cultivar 2', 'Cultivar 3']))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    print()

xgb = models["XGBoost"]
r = permutation_importance(xgb, X_te, y_te, n_repeats=30, random_state=42, n_jobs=-1, scoring='neg_log_loss')
imp = pd.DataFrame({'feature': X.columns, 'mean': r['importances_mean'], 'std': r['importances_std']})
imp = imp.sort_values('mean', ascending=False)
plt.figure(figsize=(8, 5))
plt.barh(imp['feature'][::-1], imp['mean'][::-1], xerr=imp['std'][::-1], color='indianred')
plt.xlabel("Drop in accuracy when shuffled")
plt.title("Permutation importance - XGBoost"); plt.tight_layout(); plt.savefig('permutation_importance.png', dpi=120); plt.close()
print(imp.round(4))