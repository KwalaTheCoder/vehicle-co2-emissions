# Real-data evaluation — 2026-11-02

Input: `Fuel_Consumption/FuelConsumption.csv`. Dataset SHA-256: `79a51cbe05a07c47cac6e0ee3efd16a6bddae8b3f76fcd8b3b7284eed16def7d`.

Model selection used only training-fold CV. One final holdout was reserved before preprocessing. The full CV leaderboard, dependency versions, seed, feature schema and split sizes are in [experiments/report.json](experiments/report.json).

- Model: **ExtraTreesRegressor**
- Training rows: **855**; holdout rows: **212**
- Selection metric: `neg_mean_absolute_error`; selected CV score: **-1.639249**
- Dummy baseline CV score on the same folds: **-50.510092**
- Five CV folds; seed 42; tuning: **True**
- Identical raw predictor rows grouped: **True**

| Metric | Holdout value |
|---|---:|
| mae | 1.162499 |
| rmse | 6.128641 |
| r2 | 0.991488 |


## Dataset-specific interpretation

The target is CO2EMISSIONS. Target-derived CO2_per_MPG is excluded. Predictions assume fuel-consumption measurements are already available. Identical raw predictor rows are grouped; random split evaluation does not prove unseen-manufacturer or future-year performance. MAE/RMSE units are the source CO2EMISSIONS units, conventionally g/km for this dataset; the original file provides no unit metadata.

## Reproduce

Activate an environment with the project's requirements, change to the repository root, and run:

```cmd
python -m mlproject train --data "D:\data\Fuel_Consumption\FuelConsumption.csv" --output artifacts --folds 5 --seed 42 --tune
```

Results were obtained with the dependency versions recorded in the JSON report. Different versions can change results slightly. Negative MAE selection scores are higher-is-better. CV scores are selection estimates and optimistic; the holdout is the final internal result. Hyperparameter tuning is enabled on the smaller datasets and diabetes; for credit, soccer, student and churn the default bounded candidate comparison is used. Scores are not directly comparable across datasets or tasks.

Do not tune against the now-observed holdout or claim improvement over the legacy notebooks: leakage, split policies and preprocessing differ. Raw data, row-level predictions and trained pickle artifacts are not included in this evidence commit.
