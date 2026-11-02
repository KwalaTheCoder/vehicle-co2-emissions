# Vehicle CO2 emissions regression

Standalone refactor of **KwalaTheCoder's independently written Fuel_Consumption.ipynb**. The original work is preserved in `legacy/`; the modern implementation is AI-assisted. See `PROVENANCE.json` for provenance and source hashes.

## What changed

Correctly names the target: the original predicts CO2EMISSIONS, not fuel consumption. Excludes CO2_per_MPG because it contains the answer. Keeps nominal fuel/vehicle categories, adds a city-highway consumption gap, and removes test-driven PCA selection. Excluding MODEL avoids memorizing vehicle names; validate by model/year groups for deployment.

## Data contract

FuelConsumption.csv with CO2EMISSIONS and vehicle specifications/consumption values. Predicting emissions using consumption is valid only when consumption is available at inference.

Data is not included in the source repository. Obtain the original dataset legally and verify its provenance, schema and license. No real-data accuracy claim is made. Missing labels are removed and counted in the report; missing predictors are imputed. Do not add identifiers, post-outcome fields or target-derived columns as predictors. Columns beyond the documented schema require manual leakage review.

## Run

Use Python 3.10–3.12 from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m mlproject train --data data/dataset.csv --output artifacts --folds 5
python -m mlproject train --data data/dataset.csv --output artifacts-tuned --folds 5 --tune --importance
python -m mlproject predict --model artifacts/model.joblib --data data/new_rows.csv --output predictions.csv
python -m unittest discover -s tests -v
```

On Windows activate with `.venv\Scripts\activate`. XLSX files can replace CSV. `--target` overrides a single target; `--seed`, `--test-size`, `--jobs`, and `--group-column` control the experiment. Run `python -m mlproject train --help` for all options. Default jobs=1 avoids nested parallelism and limits memory use. `--tune` expands the bounded CV search; use it only when resources permit.

For notebooks: `pip install -r requirements-notebook.txt`, launch `jupyter lab`, and open `notebooks/experiment.ipynb`. The modern notebook invokes the same tested pipeline. Original notebooks are historical references and retain their old paths and defects.

## Evaluation and outputs

The raw dataset is split first. All learned preprocessing stays inside the pipeline and each CV fold. Models include a dummy baseline, a linear baseline and task-appropriate alternatives. CV selects the model by macro F1 (classification) or negative MAE (regression). Only the chosen model is evaluated on the final holdout. CV selection scores are optimistic estimates, not final performance. Do not repeatedly inspect the holdout while tuning.

Identical predictor rows are kept together in the split and GroupKFold. Supply `--group-column` for repeated entities; the group field is excluded from predictors. Ambiguous cross-group duplicates and folds lacking a class produce an actionable error. Random splits do not establish temporal or geographic generalization; use a separate prospective dataset before deployment.

`artifacts/model.joblib` stores the complete preprocessing/model pipeline and input schema. `artifacts/report.json` stores the CV leaderboard, final metrics, per-target or per-class results, seed, row counts, data fingerprint and dependency versions. Optional permutation importance describes the holdout and must not be used to tune against it. Predictions include class probabilities for classifiers. Binary ROC AUC/AP use probabilities, and the positive class is recorded explicitly. Multi-output aggregate MAE/RMSE weight outputs equally; per-target metrics retain their units.

Dependencies are bounded; record a fully resolved environment for real experiments with `pip freeze > requirements-lock.txt`. Synthetic integration tests check runtime behavior and do not prove model quality. New preprocessing invalidates comparison to leaked legacy scores.

## Responsible use and preservation

Educational work; independent validation is required before medical, credit or other consequential use. Check consent, privacy, representativeness, bias and dataset licensing before distributing data. Only load trusted joblib artifacts. No new license is imposed on the original work; choose a distribution license after verifying dataset and source permissions.

`source-commits.json` preserves commit metadata only. It is **not** a complete Git backup. Do not delete Machine-Learning until all ten remote repositories are verified and a full Git bundle has been backed up.
