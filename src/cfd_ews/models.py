from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES = [
    "current_ratio","debt_to_assets","debt_to_equity","gross_margin",
    "operating_margin","roa","fcf_margin","revenue_yoy",
    "operating_income_yoy","market_volatility",
]

def _preprocess(features: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("numeric", numeric, features)], remainder="drop")

def build_logistic_baseline(feature_names: list[str] | None = None) -> Pipeline:
    features = feature_names or BASELINE_FEATURES
    return Pipeline([
        ("preprocess", _preprocess(features)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42)),
    ])

def build_model_suite() -> dict[str, Pipeline]:
    f = BASELINE_FEATURES
    return {
        "logistic": build_logistic_baseline(f),
        "random_forest": Pipeline([
            ("preprocess", _preprocess(f)),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=8, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", _preprocess(f)),
            ("model", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)),
        ]),
    }

def coefficient_table(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coefficients = model.named_steps["model"].coef_[0]
    return pd.DataFrame({"feature": feature_names, "coefficient": coefficients}).sort_values("coefficient", ascending=False).reset_index(drop=True)
