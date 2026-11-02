"""Deterministic domain transforms; learned statistics belong in the pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DomainFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, domain):
        self.domain = domain

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in df.select_dtypes(include=["bool", "boolean", "category", "object", "string"]):
            df[col] = df[col].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        if self.domain == "diabetes":
            for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)
            for a, b in [("Pregnancies", "Age"), ("Glucose", "BloodPressure")]:
                if a in df and b in df:
                    df[a + "_per_" + b] = df[a] / df[b].replace(0, np.nan)
        elif self.domain == "credit-risk":
            if {"loan_amnt", "person_income"} <= set(df):
                df["debt_to_income_ratio"] = df.loan_amnt / df.person_income.replace(0, np.nan)
                df["remaining_income"] = df.person_income - df.loan_amnt
        elif self.domain == "house-prices" and "Area" in df:
            df["Area"] = pd.to_numeric(df.Area.astype(str).str.replace(",", "", regex=False), errors="coerce")
            df.loc[df.Area <= 0, "Area"] = np.nan
        elif self.domain == "fuel-emissions":
            cols = ["FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY"]
            if set(cols) <= set(df):
                df["city_highway_gap"] = df[cols[0]] - df[cols[1]]
        elif self.domain == "soccer-ratings":
            if {"pace", "physic"} <= set(df):
                df["pace_x_physic"] = df.pace * df.physic
        elif self.domain == "student-performance":
            if {"Hours Studied", "Sample Question Papers Practiced"} <= set(df):
                df["practice_x_hours"] = df["Hours Studied"] * df["Sample Question Papers Practiced"]
        return df.replace([np.inf, -np.inf], np.nan)
