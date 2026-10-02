from __future__ import annotations
import json
import os
from cfd_ews.evaluation import binary_metrics
from cfd_ews.models import BASELINE_FEATURES, build_model_suite
from cfd_ews.pipeline import chronological_split, prepare_panel
from cfd_ews.synthetic import make_demo_panel

def main():
    data = prepare_panel(make_demo_panel())
    train, test = chronological_split(data)
    X_train, y_train = train[BASELINE_FEATURES], train["target"].astype(int)
    X_test, y_test = test[BASELINE_FEATURES], test["target"].astype(int)
    results = {}
    for name, model in build_model_suite().items():
        model.fit(X_train, y_train)
        results[name] = binary_metrics(y_test, model.predict_proba(X_test)[:, 1])
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/demo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("\nNOTE: metrics are from synthetic data and have no empirical interpretation.")

if __name__ == "__main__":
    main()
