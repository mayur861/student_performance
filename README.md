# StudentIQ — Student Performance Prediction

End-to-end ML project that predicts a student's **mathematics score** from demographic, family, test-preparation, reading and writing information, and serves it through a Flask web app and a JSON API.

**Result (held-out test set, 200 students):** RMSE **5.39**, MAE **4.21**, R² **0.88** — versus RMSE 15.7 for a "predict the mean" baseline. Each prediction comes with a **90% prediction range** (about ±9 points).

## What's in the project

| File | Purpose |
|---|---|
| `features.py` | Data validation, feature engineering, pipeline builder (shared by everything) |
| `train.py` | Repeated-CV tuning of 6 candidates, model selection, evaluation, plots, saving |
| `eda.py` | EDA plots + feature ablation table |
| `app.py` | Flask app (`/`, `/api/predict`, `/health`) with server-side validation |
| `tests/` | 12 pytest tests (pipeline, validation, API, web form) |
| `artifacts/` | Saved pipeline, `model_results.csv`, `metadata.json`, `ablation.csv`, `plots/` |

## ML pipeline

```text
stud.csv → validate_data → train/test split (80/20, test set touched once)
        → [optional] add_features (rw_avg, rw_diff, edu_ord)
        → ColumnTransformer
             numeric:     median imputer → StandardScaler
             categorical: most-frequent imputer → OneHotEncoder(handle_unknown="ignore")
        → model → GridSearchCV (RepeatedKFold 5×2) → 1-SE model selection
        → refit on all data → joblib pipeline + 90% prediction interval
```

**Model selection rule:** among all models whose CV RMSE is within one standard error of the best, the *simplest* one is chosen (one-standard-error rule). This avoids picking a model because of a lucky split.

| Model | CV RMSE (train, 10 folds) | Test RMSE | Test R² |
|---|---|---|---|
| Baseline (mean) | 15.02 ± 0.72 | 15.73 | -0.02 |
| **Linear Regression (selected)** | 5.43 ± 0.26 | 5.39 | 0.880 |
| Ridge (tuned) | 5.43 ± 0.28 | 5.39 | 0.881 |
| Ridge + engineered features | 5.43 ± 0.26 | 5.39 | 0.880 |
| Random Forest + FE (tuned) | 6.00 ± 0.28 | 6.18 | 0.843 |
| Gradient Boosting + FE (tuned) | 5.67 ± 0.24 | 5.45 | 0.878 |

## Key findings

1. **The relationship is almost linear.** Linear, Ridge and Ridge+FE are statistically identical (differences ≪ fold std of ~0.26). Tree models are *worse* here because the dataset is small (1,000 rows) and the signal is linear.
2. **Feature engineering did not help linear models.** `rw_avg` and `rw_diff` are linear combinations of reading/writing (correlation 0.955), so they carry no new information. Interactions, polynomial terms and a low-reading flag were also tested: all within noise. This is a real, reported result, not a missed step.
3. **Gender is the strongest non-score feature.** Removing it raises RMSE from 5.42 to 8.18. At the same reading/writing score, males score ~9 points higher in math. See the fairness note below.
4. **Parental education adds almost nothing once scores are known** (ΔRMSE 0.015). Race, lunch and test prep each add a small but real ~0.2.
5. **Reading looks unimportant in permutation importance only because writing carries the same information** (they are 0.955 correlated). Dropping reading costs just 0.10 RMSE; dropping writing costs 0.66.
6. **Demographics alone are weak** (R² ≈ 0.22, RMSE 13.3). The model is a *score-to-score* predictor, not an early-warning system, because reading/writing scores are normally known only after the exams.
7. **Regression to the mean:** low scorers (≤ 40) are over-predicted by ~3.6 points and high scorers (> 80) under-predicted by ~2.8 (see `plots/residuals.png`).
8. A single 80/20 split is noisy: the same Ridge model gave test RMSE between 4.76 and 5.87 across 20 random seeds, which is why model comparison uses repeated CV.

## Run the project

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python train.py                  # tune, evaluate, save artifacts (~1 min)
python eda.py                    # optional: EDA plots + ablation
python -m pytest -q              # run tests
python app.py                    # http://127.0.0.1:5000
```

> The saved `.joblib` was trained with scikit-learn 1.8.0. If you use another version, run `python train.py` once to regenerate it.

Production: `gunicorn app:app` (debug mode is off by default; set `FLASK_DEBUG=1` locally if you want it).

### API

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
  "gender":"female","race_ethnicity":"group C","parental_level_of_education":"some college",
  "lunch":"standard","test_preparation_course":"none","reading_score":70,"writing_score":72}'
```

Returns `prediction`, `label`, `interval_low`, `interval_high`, `at_risk`. Invalid input returns HTTP 400 with an error message.

## Limitations and fairness note

- The model uses `gender` and `race_ethnicity` because they improve accuracy on this dataset. In a real school setting, using protected attributes to score individual students is ethically and often legally problematic. The ablation shows the accuracy cost of removing them (gender: +2.8 RMSE) so the trade-off is explicit.
- The dataset is small and appears to be synthetic/educational; results will not transfer to other schools without retraining.
- The prediction interval is global (same width for everyone), derived from out-of-fold residuals.
- `at_risk` (predicted < 50) is a simple threshold on the regression output, not a trained classifier.

## Interview explanations

**Why Pipeline + ColumnTransformer?** Preprocessing and model are fitted together, so training and prediction can never go out of sync; numeric and categorical columns need different transformations.

**Why `handle_unknown="ignore"`?** Unseen categories at prediction time produce an all-zero encoding instead of a crash (covered by a test).

**Why repeated CV and the 1-SE rule?** With 1,000 rows one split is noisy (RMSE ranged 4.8–5.9 across seeds). CV gives mean ± std, and the 1-SE rule prefers the simpler model when differences are not significant.

**Why didn't feature engineering help?** The engineered features were linear combinations of existing ones, and the target is almost linear in the inputs. They remain in the pipeline (`use_fe=True`) for tree models, where they gave a small gain.

## Future improvements

- SHAP / per-student explanation in the UI
- Prediction history in SQLite
- Classification model for pass/fail risk
- Docker + deployment (Render/Railway/Azure)
- Model monitoring and scheduled retraining
