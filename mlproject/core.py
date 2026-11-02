"""Holdout-first training, fold-local preprocessing, and portable artifacts."""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
    KFold, StratifiedKFold, train_test_split)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import DomainFeatures


def read_data(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Use a CSV or XLSX input file.")
    return pd.read_csv(path)


def prepare(df, config, task=None, target=None):
    """Separate labels before any domain transform; reject ambiguous duplicates."""
    task = task or config["task"]
    if task != config["task"] and config["domain"] != "student-performance":
        raise ValueError("This project supports only its configured task.")
    targets = [target] if target else config["targets"]
    missing = set(targets) - set(df)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    df = df.dropna(subset=targets).copy()
    if df.empty:
        raise ValueError("No rows with observed targets.")
    y = df[targets[0]].copy() if len(targets) == 1 else df[targets].copy()
    if config["domain"] == "student-performance" and task == "classification":
        if target is not None:
            raise ValueError("Student classification derives bands from Performance Index; omit --target.")
        if not pd.to_numeric(y, errors="coerce").between(0, 100).all():
            raise ValueError("Performance Index must be numeric and between 0 and 100.")
        y = pd.cut(y, bins=[-np.inf, 25, 50, 75, 100], labels=["low", "fair", "good", "high"]).astype(str)
    elif task == "regression":
        if not np.isfinite(np.asarray(y, dtype=float)).all():
            raise ValueError("Regression targets must be finite numbers.")
    forbidden = set(targets + config.get("exclude", []) + config["targets"])
    if config.get("features"):
        missing = set(config["features"]) - set(df)
        if missing:
            raise ValueError(f"Missing predictor columns: {sorted(missing)}")
        X = df[config["features"]].copy()
    elif config.get("text"):
        if config["text"] not in df:
            raise ValueError(f"Missing text column: {config['text']}")
        X = df[[config["text"]]].copy()
    else:
        X = df.drop(columns=list(forbidden), errors="ignore").copy()
    X = X.drop(columns=list(forbidden), errors="ignore")
    if not len(X.columns):
        raise ValueError("No predictor columns remain.")
    # Identical predictors cannot occur in both train and test. Conflicting labels
    # are retained as one group, so their disagreement is measured honestly.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    return X.reset_index(drop=True), y.reset_index(drop=True), groups, task


def preprocess(domain):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
    return Pipeline([("features", DomainFeatures(domain)),
                     ("columns", ColumnTransformer([
                         ("numeric", numeric, make_column_selector(dtype_include=np.number)),
                         ("categorical", categorical, make_column_selector(dtype_exclude=np.number))]))])


def candidates(config, task, seed, tune):
    if config.get("text"):
        feature = ColumnTransformer([("text", TfidfVectorizer(sublinear_tf=True,
            strip_accents="unicode", max_features=50000, ngram_range=(1, 2)), config["text"])])
        models = [DummyClassifier(strategy="prior"),
                  LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                  ComplementNB()]
    else:
        feature = preprocess(config["domain"])
        if task == "classification":
            models = [DummyClassifier(strategy="prior"),
                      LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                      RandomForestClassifier(n_estimators=120, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=1),
                      ExtraTreesClassifier(n_estimators=120, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=1)]
        else:
            models = [DummyRegressor(strategy="mean"), Ridge(alpha=1, solver="lsqr"),
                      RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=1),
                      ExtraTreesRegressor(n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=1)]
    grids = []
    for model in models:
        params = {"model": [model]}
        if tune:
            if isinstance(model, (LogisticRegression, Ridge)):
                params["model__C" if isinstance(model, LogisticRegression) else "model__alpha"] = [0.1, 1, 10]
            elif isinstance(model, (RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor)):
                params.update({"model__max_depth": [None, 12], "model__min_samples_leaf": [2, 5]})
            elif isinstance(model, ComplementNB):
                params["model__alpha"] = [0.1, 1, 10]
        grids.append(params)
    return Pipeline([("preprocess", feature), ("model", models[0])]), grids


def train(df, config, output, *, task=None, target=None, seed=42, folds=5,
          test_size=0.2, tune=False, jobs=1, importance=False, group_column=None):
    if folds < 2 or not 0 < test_size < 1:
        raise ValueError("Use folds >= 2 and 0 < test-size < 1.")
    if group_column:
        if group_column not in df or df[group_column].isna().any():
            raise ValueError("Group column must exist and contain no missing values.")
        config = dict(config, exclude=config.get("exclude", []) + [group_column])
    X, y, duplicate_groups, task = prepare(df, config, task, target)
    if config.get("text"):
        X[config["text"]] = X[config["text"]].fillna("").astype(str)
    # Recompute text groups after normalizing missing text.
    duplicate_groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    groups = duplicate_groups
    grouped = len(np.unique(groups)) != len(X)
    if group_column:
        groups = df.dropna(subset=[target] if target else config["targets"])[group_column].to_numpy()
        grouped = True
    if grouped:
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_size,
            random_state=seed, stratify=y if task == "classification" else None)
    Xtrain, Xtest = X.iloc[train_idx], X.iloc[test_idx]
    ytrain, ytest = y.iloc[train_idx], y.iloc[test_idx]
    if task == "regression" and len(ytest) < 2:
        raise ValueError("Regression holdout needs at least two observations for R2.")
    if set(duplicate_groups[train_idx]) & set(duplicate_groups[test_idx]):
        raise ValueError("Identical predictors span the chosen groups; reconcile duplicates before training.")
    if task == "classification" and (set(ytrain) != set(y) or set(ytest) != set(y)):
        raise ValueError("Every class must occur in both splits. Add data or adjust test-size/seed.")
    if grouped:
        if len(np.unique(groups[train_idx])) < folds:
            raise ValueError("Not enough independent training groups for the requested folds.")
        cv = list(GroupKFold(n_splits=folds).split(Xtrain, ytrain, groups[train_idx]))
    elif task == "classification":
        if ytrain.value_counts().min() < folds:
            raise ValueError("Each training class needs at least --folds observations.")
        cv = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xtrain, ytrain))
    else:
        if len(Xtrain) < 2 * folds:
            raise ValueError("Need at least two training rows per fold.")
        cv = list(KFold(folds, shuffle=True, random_state=seed).split(Xtrain))
    if task == "classification":
        if any(set(ytrain.iloc[a]) != set(ytrain) or set(ytrain.iloc[b]) != set(ytrain) for a, b in cv):
            raise ValueError("Every CV fold must contain all classes. Reduce folds or revise groups.")
    scoring = "f1_macro" if task == "classification" else "neg_mean_absolute_error"
    pipeline, grid = candidates(config, task, seed, tune)
    search = GridSearchCV(pipeline, grid, scoring=scoring, cv=cv, n_jobs=jobs,
                          refit=True, error_score="raise", return_train_score=False)
    search.fit(Xtrain, ytrain)
    model = search.best_estimator_
    pred = model.predict(Xtest)  # Holdout is used only after CV selection.
    if task == "classification":
        classes = model.classes_
        proba = model.predict_proba(Xtest)
        metrics = {"accuracy": accuracy_score(ytest, pred),
                   "balanced_accuracy": balanced_accuracy_score(ytest, pred),
                   "f1_macro": f1_score(ytest, pred, average="macro", zero_division=0),
                   "classification_report": classification_report(ytest, pred, zero_division=0, output_dict=True),
                   "classes": classes.tolist(), "confusion_matrix": confusion_matrix(ytest, pred, labels=classes).tolist()}
        if len(classes) == 2:
            binary_y = np.asarray(ytest) == classes[1]
            metrics.update({"roc_auc": roc_auc_score(binary_y, proba[:, 1]),
                "average_precision": average_precision_score(binary_y, proba[:, 1]),
                "positive_class": str(classes[1])})
    else:
        metrics = {"mae": mean_absolute_error(ytest, pred),
                   "rmse": float(np.sqrt(mean_squared_error(ytest, pred))), "r2": r2_score(ytest, pred)}
        if isinstance(ytest, pd.DataFrame):
            metrics["per_target"] = {col: {"mae": mean_absolute_error(ytest[col], pred[:, i]),
                "rmse": float(np.sqrt(mean_squared_error(ytest[col], pred[:, i]))),
                "r2": r2_score(ytest[col], pred[:, i])} for i, col in enumerate(ytest.columns)}
    leaderboard = [{"model": type(p["model"]).__name__,
        "parameters": {k: v for k, v in p.items() if k != "model"},
        "cv_score": float(search.cv_results_["mean_test_score"][i]),
        "cv_std": float(search.cv_results_["std_test_score"][i])} for i, p in enumerate(search.cv_results_["params"])]
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    report = {"project": config["domain"], "task": task, "seed": seed, "folds": folds,
              "scoring": scoring, "cv_selection_score": search.best_score_,
              "selected_model": type(model["model"]).__name__, "leaderboard": leaderboard,
              "holdout": metrics, "train_rows": len(train_idx), "test_rows": len(test_idx),
              "input_rows": len(df), "missing_target_rows_removed": len(df) - len(X),
              "grouped_split": grouped, "group_column": group_column,
              "features": list(X.columns), "data_fingerprint": digest,
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__},
              "note": "CV scores select the model and are optimistic; holdout metrics are the final evaluation. Negative MAE scores are higher-is-better. Synthetic tests are not real-data benchmarks."}
    if importance:
        perm = permutation_importance(model, Xtest, ytest, scoring=scoring, n_repeats=5, random_state=seed, n_jobs=jobs)
        report["holdout_permutation_importance"] = {c: {"mean": float(m), "std": float(s)}
            for c, m, s in zip(X.columns, perm.importances_mean, perm.importances_std)}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    artifact = {"pipeline": model, "features": list(X.columns), "text": config.get("text"),
                "task": task, "targets": [target] if target else config["targets"],
                "versions": report["versions"]}
    joblib.dump(artifact, out / "model.joblib")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def predict(artifact_path, df):
    # Load only artifacts you trust: joblib uses Python pickle.
    artifact = joblib.load(artifact_path)
    missing = set(artifact["features"]) - set(df)
    if missing:
        raise ValueError(f"Missing predictor columns: {sorted(missing)}")
    X = df[artifact["features"]].copy()
    if artifact["text"]:
        X[artifact["text"]] = X[artifact["text"]].fillna("").astype(str)
    model = artifact["pipeline"]
    pred = model.predict(X)
    if pred.ndim == 1:
        result = pd.DataFrame({"prediction": pred}, index=df.index)
    else:
        result = pd.DataFrame(pred, columns=artifact["targets"], index=df.index)
    if artifact["task"] == "classification":
        probabilities = model.predict_proba(X)
        for i, label in enumerate(model.classes_):
            result[f"probability_{label}"] = probabilities[:, i]
    return result
