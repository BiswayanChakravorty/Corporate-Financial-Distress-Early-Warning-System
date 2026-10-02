from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES = [
    "current_ratio","debt_to_assets","debt_to_equity","gross_margin",
    "operating_margin","roa","fcf_margin","revenue_yoy",
    "operating_income_yoy","market_volatility",
]

def build_logistic_baseline(feature_names: list[str] | None = None) -> Pipeline:
    features = feature_names or BASELINE_FEATURES
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    preprocess = ColumnTransformer([("numeric", numeric, features)], remainder="drop")
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])

def coefficient_table(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coefficients = model.named_steps["model"].coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
