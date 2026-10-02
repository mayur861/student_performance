"""
Exploratory analysis + feature ablation.  Run: python eda.py
Creates artifacts/plots/eda_*.png and artifacts/ablation.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score

from features import CATEGORICAL, FEATURES, NUMERIC, TARGET, build_pipeline, validate_data

BASE = Path(__file__).resolve().parent
PLOTS = BASE / "artifacts" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

df = validate_data(pd.read_csv(BASE / "data" / "stud.csv"))

# ---- EDA plots ----
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, col in zip(ax, [TARGET, "reading_score", "writing_score"]):
    sns.histplot(df[col], bins=25, kde=True, ax=a)
    a.set_title(col)
fig.tight_layout(); fig.savefig(PLOTS / "eda_distributions.png", dpi=130); plt.close(fig)

plt.figure(figsize=(4.5, 3.8))
sns.heatmap(df[NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Score correlations"); plt.tight_layout()
plt.savefig(PLOTS / "eda_correlation.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 5, figsize=(20, 3.8), sharey=True)
for a, col in zip(ax, CATEGORICAL):
    sns.barplot(data=df, x=col, y=TARGET, ax=a, errorbar="sd")
    a.tick_params(axis="x", rotation=30); a.set_title(f"math by {col}")
fig.tight_layout(); fig.savefig(PLOTS / "eda_group_means.png", dpi=130); plt.close(fig)

# ---- feature ablation: drop one feature at a time (Ridge, repeated CV) ----
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

def cv_rmse(cols):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c in NUMERIC]
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pipe = Pipeline([("pre", pre), ("model", Ridge(1.0))])
    s = cross_val_score(pipe, df[cols], df[TARGET], cv=cv, scoring="neg_root_mean_squared_error")
    return -s.mean(), s.std()

full, full_std = cv_rmse(FEATURES)
rows = [{"setting": "all features", "cv_rmse": round(full, 3), "delta_vs_full": 0.0}]
for f in FEATURES:
    m, _ = cv_rmse([c for c in FEATURES if c != f])
    rows.append({"setting": f"drop {f}", "cv_rmse": round(m, 3), "delta_vs_full": round(m - full, 3)})
m, _ = cv_rmse(CATEGORICAL)
rows.append({"setting": "only demographics (no reading/writing)", "cv_rmse": round(m, 3), "delta_vs_full": round(m - full, 3)})
ablation = pd.DataFrame(rows)
ablation.to_csv(BASE / "artifacts" / "ablation.csv", index=False)
print(ablation.to_string(index=False))
print(f"\nfold-to-fold std of CV RMSE ~ {full_std:.2f}: deltas below this are noise")
