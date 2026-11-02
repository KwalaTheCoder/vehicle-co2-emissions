# Audit and refactor

Correctly names the target: the original predicts CO2EMISSIONS, not fuel consumption. Excludes CO2_per_MPG because it contains the answer. Keeps nominal fuel/vehicle categories, adds a city-highway consumption gap, and removes test-driven PCA selection. Excluding MODEL avoids memorizing vehicle names; validate by model/year groups for deployment.

Original algorithms and historical outputs remain in legacy/. No production dataset was available during refactoring; new real-data metrics remain unmeasured. See README.md for the reproducible evaluation protocol.
