from __future__ import annotations

import pandas as pd
import streamlit as st

from cfd_ews.features import build_financial_features
from cfd_ews.models import BASELINE_FEATURES, build_model_suite
from cfd_ews.synthetic import make_demo_panel
from cfd_ews.pipeline import chronological_split

st.set_page_config(page_title="Corporate Distress EWS", page_icon="📊", layout="wide")
st.title("Corporate Financial Distress Early-Warning System")
st.caption("Research dashboard — demo data is synthetic unless a point-in-time dataset is supplied.")

@st.cache_data
def demo_data():
    return make_demo_panel()

df = demo_data()
featured = build_financial_features(df)
train, test = chronological_split(featured.dropna(subset=["distress_event"]))
train["target"] = train.groupby("company_id")["distress_event"].shift(-1)
test["target"] = test.groupby("company_id")["distress_event"].shift(-1)
train = train.dropna(subset=["target"])
test = test.dropna(subset=["target"])

st.sidebar.header("Research controls")
model_name = st.sidebar.selectbox("Model", ["logistic", "random_forest", "hist_gradient_boosting"])

model = build_model_suite()[model_name]
model.fit(train[BASELINE_FEATURES], train["target"].astype(int))
test_prob = model.predict_proba(test[BASELINE_FEATURES])[:, 1]

latest = featured.sort_values("period").groupby("company_id").tail(1).copy()
latest["risk_probability"] = model.predict_proba(latest[BASELINE_FEATURES])[:, 1]
latest["risk_band"] = pd.cut(
    latest["risk_probability"],
    bins=[-0.01, 0.33, 0.66, 1.01],
    labels=["Lower", "Intermediate", "Higher"],
)

c1,c2,c3 = st.columns(3)
c1.metric("Companies", latest["company_id"].nunique())
c2.metric("Model", model_name.replace("_"," ").title())
c3.metric("Test observations", len(test))

st.subheader("Latest risk distribution")
st.bar_chart(latest["risk_band"].value_counts().sort_index())

st.subheader("Company-level screening")
show = latest[["company_id","period","risk_probability","risk_band","debt_to_assets","current_ratio","operating_margin","fcf_margin"]].sort_values("risk_probability", ascending=False)
st.dataframe(show, use_container_width=True, hide_index=True)

st.info("Synthetic data is for software validation only. Do not interpret these probabilities as evidence about real companies.")
