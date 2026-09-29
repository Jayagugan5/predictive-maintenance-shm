"""Streamlit risk dashboard.  Run:  streamlit run dashboard.py   (after python pipeline.py)"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

OUT = Path(__file__).parent / "outputs"
st.set_page_config(page_title="Infrastructure Risk Monitor", layout="wide")
st.title("Predictive Maintenance – Infrastructure Risk Monitor")

if not (OUT / "predictions.csv").exists():
    st.error("No results found. Run `python pipeline.py` first.")
    st.stop()


@st.cache_data
def load():
    return pd.read_csv(OUT / "predictions.csv")


df = load()
metrics = json.load(open(OUT / "metrics.json"))
cb = json.load(open(OUT / "cost_benefit.json"))
CLASSES = ["Normal", "Warning", "Critical"]
ICON = {0: "🟢 Normal", 1: "🟡 Warning", 2: "🔴 Critical"}

tab1, tab2, tab3 = st.tabs(["Live monitor", "Model performance", "Cost-benefit"])

with tab1:
    t = st.slider("Monitoring step (replay of sensor stream)", int(df.step.min()), int(df.step.max()), 150)
    snap = df[df.step == t].copy()
    snap["Status"] = snap.pred.map(ICON)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Structures", len(snap))
    c2.metric("🟢 Normal", int((snap.pred == 0).sum()))
    c3.metric("🟡 Warning", int((snap.pred == 1).sum()))
    c4.metric("🔴 Critical", int((snap.pred == 2).sum()))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Fleet risk ranking")
        rank = snap.sort_values("risk_score", ascending=False)[
            ["structure_id", "kind", "Status", "risk_score", "p_critical", "f1_shift_comp", "alert"]]
        st.dataframe(rank.round(3), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Alert log")
        log = df[(df.step <= t) & df.alert_onset][["step", "structure_id", "kind", "risk_score"]]
        log = log.sort_values("step", ascending=False).rename(columns={"step": "raised at step"})
        st.dataframe(log.round(3), hide_index=True, use_container_width=True)

    sid = st.selectbox("Structure detail", rank.structure_id.tolist())
    g = df[(df.structure_id == sid) & (df.step <= t)].set_index("step")
    a, b = st.columns(2)
    a.caption("Risk score (0 = normal, 1 = critical)")
    a.line_chart(g[["risk_score", "p_critical"]])
    b.caption("Temperature-compensated natural-frequency shift (%)")
    b.line_chart(g[["f1_shift_raw", "f1_shift_comp"]])

with tab2:
    m = metrics["RandomForest"]
    st.caption("All results are out-of-fold: every structure was scored by a model that never saw it in training.")
    rep = pd.DataFrame(m["report"]).T.loc[CLASSES + ["macro avg"], ["precision", "recall", "f1-score"]]
    st.dataframe(rep.round(3), use_container_width=True)
    st.write("ROC-AUC (one-vs-rest):", {k: round(v, 3) for k, v in m["roc_auc"].items()})
    st.write("Lead time (RF):", m["lead"])
    for row in [("confusion_matrix", "roc_curves"), ("lead_time_hist", "feature_importance"), ("example_timeline",)]:
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            col.image(str(OUT / "figures" / f"{name}.png"))
    st.subheader("Model comparison")
    comp = {k: {"lead_median": v["lead"]["lead_median"], "lead_p10_worst": v["lead"]["lead_p10_worst"],
                "missed": v["lead"]["missed"], "false_alert_structs": v["lead"]["structures_with_false_alert"]}
            for k, v in metrics.items()}
    st.dataframe(pd.DataFrame(comp).T, use_container_width=True)

with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("Reactive cost", f"${cb['reactive_total']/1e6:.1f}M")
    c2.metric("Predictive cost", f"${cb['predictive_total']/1e6:.1f}M")
    c3.metric("Savings", f"{cb['savings_pct']:.0f}%")
    st.write(f"{cb['caught_in_time']} of {cb['failures']} failures flagged with ≥ {cb['min_lead_steps']} steps of warning; "
             f"{cb['false_alert_structures']} structures raised a false alert.")
    st.caption("Cost inputs are placeholder assumptions – replace with figures from literature:")
    st.json(cb["assumptions"])
