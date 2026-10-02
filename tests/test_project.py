import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as app_module  # noqa: E402
from features import FEATURES, add_features, validate_data, validate_input  # noqa: E402

VALID = {
    "gender": "female", "race_ethnicity": "group C",
    "parental_level_of_education": "some college", "lunch": "standard",
    "test_preparation_course": "none", "reading_score": 70, "writing_score": 72,
}


@pytest.fixture
def client():
    return app_module.app.test_client()


def test_prediction_in_range_and_interval_ordered():
    r = app_module.predict(VALID)
    assert 0 <= r["interval_low"] <= r["prediction"] <= r["interval_high"] <= 100


def test_monotonic_in_scores():
    low = app_module.predict({**VALID, "reading_score": 30, "writing_score": 30})["prediction"]
    high = app_module.predict({**VALID, "reading_score": 90, "writing_score": 90})["prediction"]
    assert high > low


@pytest.mark.parametrize("bad", [
    {"reading_score": 150}, {"writing_score": -1}, {"reading_score": "abc"},
    {"gender": "other"}, {"lunch": None},
])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        validate_input({**VALID, **bad})


def test_api_ok_and_error(client):
    assert client.post("/api/predict", json=VALID).status_code == 200
    assert client.post("/api/predict", json={**VALID, "reading_score": 999}).status_code == 400
    assert client.post("/api/predict", data="not json").status_code == 400


def test_web_form_and_health(client):
    assert client.get("/health").json["status"] == "ok"
    resp = client.post("/", data=VALID)
    assert resp.status_code == 200 and b"prediction range" in resp.data


def test_unknown_category_does_not_crash_pipeline():
    row = pd.DataFrame([{**VALID, "race_ethnicity": "group Z"}])[FEATURES]
    assert app_module.model.predict(row).shape == (1,)


def test_feature_engineering_columns():
    out = add_features(pd.DataFrame([VALID]))
    assert out.loc[0, "rw_avg"] == 71 and out.loc[0, "rw_diff"] == -2 and out.loc[0, "edu_ord"] == 2


def test_validate_data_rejects_bad_scores():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "stud.csv")
    df.loc[0, "math_score"] = 101
    with pytest.raises(ValueError):
        validate_data(df)
