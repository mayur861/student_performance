"""
Student Performance Prediction - Training Pipeline
Target: math_score
Run: python train.py

What this script does
1. Loads + validates the data
2. Holds out 20% as a final test set (touched only once)
3. Tunes every candidate model with repeated 5-fold CV on the training part
4. Picks the simplest model within 1 standard error of the best CV RMSE
5. Evaluates it on the held-out test set, then refits on all data
6. Computes a 90% prediction interval from out-of-fold residuals
7. Saves pipeline, metrics, metadata and diagnostic plots
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold,
                                     cross_val_predict, train_test_split)

from features import FEATURES, TARGET, build_pipeline, validate_data

SEED = 42
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "stud.csv"
ARTIFACTS = BASE / "artifacts"
PLOTS = ARTIFACTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Ordered from simplest to most complex (used by the 1-SE rule).
# name -> (estimator, use_feature_engineering, param_grid)
CANDIDATES = {
    "Baseline (mean)": (DummyRegressor(strategy="mean"), False, {}),
    "Linear Regression": (LinearRegression(), False, {}),
    "Ridge": (Ridge(), False, {"model__alpha": [0.1, 1, 3, 10, 30, 100]}),
    "Ridge + FE": (Ridge(), True, {"model__alpha": [0.1, 1, 3, 10, 30, 100]}),
    "Random Forest + FE": (
        RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1), True,
        {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [3, 5, 10]},
    ),
    "Gradient Boosting + FE": (
        GradientBoostingRegressor(random_state=SEED), True,
        {"model__n_estimators": [150, 300], "model__learning_rate": [0.03, 0.06],
         "model__max_depth": [2, 3], "model__subsample": [0.8]},
    ),
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = validate_data(pd.read_csv(DATA_PATH))
    X, y = df[FEATURES], df[TARGET]
    print(f"Data OK: {len(df)} rows, {len(FEATURES)} features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)

    # ---- 1. tune + cross-validate every candidate on the training part ----
    rows, fitted = [], {}
    for name, (estimator, use_fe, grid) in CANDIDATES.items():
        pipe = build_pipeline(estimator, use_fe)
        search = GridSearchCV(pipe, grid or {"model": [estimator]},
                              scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        i = search.best_index_
        cv_rmse = -search.cv_results_["mean_test_score"][i]
        cv_std = search.cv_results_["std_test_score"][i]
        pred = search.predict(X_test)
        rows.append({
            "model": name,
            "cv_rmse": round(cv_rmse, 4),
            "cv_rmse_std": round(cv_std, 4),
            "test_rmse": round(rmse(y_test, pred), 4),
            "test_mae": round(mean_absolute_error(y_test, pred), 4),
            "test_r2": round(r2_score(y_test, pred), 4),
            "best_params": json.dumps({k.replace("model__", ""): v for k, v in search.best_params_.items()}
                                      if grid else {}),
        })
        fitted[name] = search
        print(f"{name:24s} CV RMSE={cv_rmse:.3f}±{cv_std:.3f} | test RMSE={rows[-1]['test_rmse']:.3f} "
              f"| test R2={rows[-1]['test_r2']:.3f}")

    results = pd.DataFrame(rows)

    # ---- 2. model selection: simplest model within 1 SE of the best CV RMSE ----
    n_folds = cv.get_n_splits()
    best_row = results.loc[results["cv_rmse"].idxmin()]
    threshold = best_row["cv_rmse"] + best_row["cv_rmse_std"] / np.sqrt(n_folds)
    eligible = results[(results["cv_rmse"] <= threshold) & (results["model"] != "Baseline (mean)")]
    best_name = eligible.iloc[0]["model"]            # CANDIDATES order = simplest first
    print(f"\nLowest CV RMSE : {best_row['model']} ({best_row['cv_rmse']:.3f})")
    print(f"Selected (1-SE): {best_name}")

    estimator, use_fe, grid = CANDIDATES[best_name]
    best_params = fitted[best_name].best_params_ if grid else {}
    eval_pipe = fitted[best_name].best_estimator_            # fitted on the training part only
    test_pred = eval_pipe.predict(X_test)
    test_metrics = {
        "rmse": round(rmse(y_test, test_pred), 4),
        "mae": round(mean_absolute_error(y_test, test_pred), 4),
        "r2": round(r2_score(y_test, test_pred), 4),
    }
    print("Held-out test metrics:", test_metrics)

    # ---- 3. refit on all data + out-of-fold residuals for the prediction interval ----
    final_pipe = build_pipeline(estimator, use_fe).set_params(**best_params).fit(X, y)
    oof = cross_val_predict(build_pipeline(estimator, use_fe).set_params(**best_params),
                            X, y, cv=KFold(10, shuffle=True, random_state=SEED))
    residuals = y.to_numpy() - oof
    lo, hi = np.quantile(residuals, [0.05, 0.95])
    interval = {"lower": round(float(lo), 3), "upper": round(float(hi), 3), "coverage": 0.90}
    print(f"90% prediction interval: prediction {lo:+.1f} / {hi:+.1f}")

    # ---- 4. diagnostic plots ----
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y, oof, s=10, alpha=0.5)
    ax.plot([0, 100], [0, 100], "r--", lw=1)
    ax.set(xlabel="Actual math score", ylabel="Predicted (out-of-fold)", title=f"Actual vs predicted - {best_name}")
    fig.tight_layout(); fig.savefig(PLOTS / "actual_vs_predicted.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(residuals, bins=30, color="#4c72b0"); ax[0].axvline(0, color="r", ls="--")
    ax[0].set(title="Residual distribution", xlabel="Actual - predicted")
    ax[1].scatter(oof, residuals, s=10, alpha=0.5); ax[1].axhline(0, color="r", ls="--")
    ax[1].set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Residual")
    fig.tight_layout(); fig.savefig(PLOTS / "residuals.png", dpi=130); plt.close(fig)

    perm = permutation_importance(eval_pipe, X_test, y_test, n_repeats=30, random_state=SEED,
                                  scoring="neg_root_mean_squared_error")
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(imp.index, imp.values, xerr=pd.Series(perm.importances_std, index=FEATURES)[imp.index], color="#55a868")
    ax.set(title="Permutation importance (increase in test RMSE)", xlabel="RMSE increase")
    fig.tight_layout(); fig.savefig(PLOTS / "feature_importance.png", dpi=130); plt.close(fig)

    # ---- 5. save everything ----
    joblib.dump(final_pipe, ARTIFACTS / "student_performance_pipeline.joblib")
    results.to_csv(ARTIFACTS / "model_results.csv", index=False)
    import sklearn
    metadata = {
        "target": TARGET,
        "best_model": best_name,
        "best_params": {k.replace("model__", ""): v for k, v in best_params.items()},
        "selection_rule": "simplest model within 1 standard error of the lowest CV RMSE",
        "features": FEATURES,
        "feature_engineering": bool(use_fe),
        "dataset_rows": int(len(df)),
        "test_size": 0.20,
        "cv": "RepeatedKFold(5 folds x 2 repeats) on the training split",
        "test_metrics": test_metrics,
        "prediction_interval": interval,
        "permutation_importance": {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).items()},
        "sklearn_version": sklearn.__version__,
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    print("\nModel comparison:")
    print(results.drop(columns="best_params").to_string(index=False))
    print(f"\nSaved to {ARTIFACTS}")


if __name__ == "__main__":
    main()
