"""Behavior checks with synthetic inputs; these do not benchmark real datasets."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mlproject.core import predict, prepare, train
from mlproject.features import DomainFeatures


CONFIG = json.loads(Path("project.json").read_text())


def sample(config, n=100):
    rng = np.random.default_rng(12)
    domain = config["domain"]
    x = rng.normal(size=n)
    df = pd.DataFrame({"signal": x, "category": np.where(x > 0, "a", "b"), "flag": x > 0})
    if config.get("features"):
        df = pd.DataFrame({col: rng.uniform(10, 90, n) for col in config["features"]})
        x = df.iloc[:, 0].to_numpy() / 20
    if domain == "house-prices":
        df["Area"] = ["1,200" if i % 2 else "85" for i in range(n)]
        df["Address"] = np.where(x > 0, "north", "south")
        df["Price(USD)"] = x * 100
        df["Price_Per_Meter(USD)"] = x * 2
    if domain == "fuel-emissions":
        df["CO2_per_MPG"] = x * 100
    if domain == "student-performance":
        df["Hours Studied"] = rng.integers(1, 10, n)
        df["Sample Question Papers Practiced"] = rng.integers(1, 10, n)
        df["Performance Index Cat"] = x > 0
    if domain == "diabetes":
        df["Glucose"] = rng.uniform(70, 180, n)
        df.loc[0, "Glucose"] = 0
        df["Age"] = rng.integers(20, 70, n)
        df["Pregnancies"] = rng.integers(0, 5, n)
    if domain == "credit-risk":
        df["person_income"] = rng.uniform(100, 900, n)
        df.loc[0, "person_income"] = 0
        df["loan_amnt"] = rng.uniform(10, 300, n)
    for i, target in enumerate(config["targets"]):
        df[target] = (x > 0).astype(int) if config["task"] == "classification" else x * (i + 2) + rng.normal(0, .1, n)
    if domain == "student-performance":
        df["Performance Index"] = np.tile([15., 40., 65., 90.], n // 4)
    if config.get("text"):
        df[config["text"]] = [f"{'win free offer' if i % 2 else 'hello meeting friend'} unique{i}" for i in range(n)]
        df[config["targets"][0]] = ["spam" if i % 2 else "ham" for i in range(n)]
    return df


class PipelineTests(unittest.TestCase):
    def test_training_and_artifact_roundtrip(self):
        df = sample(CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            report = train(df, CONFIG, tmp, folds=2, importance=True)
            prediction = predict(Path(tmp) / "model.joblib", df.iloc[:4].drop(columns=CONFIG["targets"]))
            self.assertEqual(len(prediction), 4)
            self.assertEqual(report["train_rows"] + report["test_rows"], len(df))
            self.assertTrue(np.isfinite(report["cv_selection_score"]))
            artifact = joblib.load(Path(tmp) / "model.joblib")
            self.assertTrue(set(CONFIG["targets"]).isdisjoint(artifact["features"]))
            self.assertTrue(set(CONFIG.get("exclude", [])).isdisjoint(artifact["features"]))
            json.loads((Path(tmp) / "report.json").read_text())
            missing = df.iloc[:4].drop(columns=artifact["features"][0])
            with self.assertRaisesRegex(ValueError, "Missing predictor"):
                predict(Path(tmp) / "model.joblib", missing)

    def test_duplicate_predictors_stay_together(self):
        df = sample(CONFIG)
        df = pd.concat([df, df.iloc[:20]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            report = train(df, CONFIG, tmp, folds=2)
            self.assertTrue(report["grouped_split"])

    def test_missing_target_is_explained(self):
        with self.assertRaisesRegex(ValueError, "Missing target"):
            prepare(sample(CONFIG).drop(columns=CONFIG["targets"]), CONFIG)

    def test_missing_predictors_and_unseen_categories_at_inference(self):
        if CONFIG.get("text"):
            self.skipTest("Tabular preprocessing behavior")
        df = sample(CONFIG)
        numeric = [c for c in df.select_dtypes(include=np.number) if c not in CONFIG["targets"] + CONFIG.get("exclude", [])]
        if numeric:
            df.loc[0, numeric[0]] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            train(df, CONFIG, tmp, folds=2)
            incoming = df.iloc[:3].copy()
            for col in incoming.select_dtypes(include=object):
                incoming[col] = "previously unseen category"
            result = predict(Path(tmp) / "model.joblib", incoming)
            self.assertEqual(len(result), 3)

    def test_excluded_leakage_columns_do_not_change_predictions(self):
        df = sample(CONFIG)
        X, y, _, _ = prepare(df, CONFIG)
        changed = df.copy()
        for col in CONFIG.get("exclude", []):
            changed[col] = "answer leak"
        again, _, _, _ = prepare(changed, CONFIG)
        pd.testing.assert_frame_equal(X, again)

    def test_features_do_not_mutate_inputs(self):
        X, _, _, _ = prepare(sample(CONFIG), CONFIG)
        before = X.copy(deep=True)
        transformed = DomainFeatures(CONFIG["domain"]).fit_transform(X)
        pd.testing.assert_frame_equal(X, before)
        self.assertFalse(np.isinf(transformed.select_dtypes(include=np.number)).any().any())

    def test_student_classification_has_no_target_features(self):
        if CONFIG["domain"] != "student-performance":
            self.skipTest("Student-specific alternate task")
        df = sample(CONFIG)
        X, y, _, _ = prepare(df, CONFIG, task="classification")
        self.assertNotIn("Performance Index Cat", X)
        self.assertNotIn("Performance Index", X)
        self.assertEqual(set(y), {"low", "fair", "good", "high"})
        with tempfile.TemporaryDirectory() as tmp:
            report = train(df, CONFIG, tmp, task="classification", folds=2)
            self.assertEqual(report["task"], "classification")


if __name__ == "__main__":
    unittest.main()
