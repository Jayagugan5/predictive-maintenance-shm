"""End-to-end: simulate -> features -> grouped CV (baseline RF, GB, autoencoder) -> metrics, figures,
lead-time, cost-benefit -> saved model + predictions for the dashboard.   Run:  python pipeline.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_sample_weight
from simulate import simulate_fleet
from features import build_feature_table, FEATURES

OUT = Path(__file__).parent / "outputs"
FIG = OUT / "figures"
CLASSES = ["Normal", "Warning", "Critical"]
K_CONSEC, MIN_LEAD, SEED = 3, 5, 0

# ---- Cost-benefit ASSUMPTIONS (replace with cited literature figures in your report) ----
COST = dict(reactive_failure=4_000_000,    # emergency repair + downtime + damage per failure ($)
            planned_repair=1_000_000,      # planned repair after early warning (=> reactive is 4x, cf. "3-5x")
            false_alarm_inspection=20_000, # cost of inspecting a falsely flagged structure
            sensor_system_per_structure=15_000)  # sensors + connectivity over the study horizon


def sustained(flag, k=K_CONSEC):
    c = np.convolve(flag.astype(int), np.ones(k, int), mode="full")[:len(flag)]
    return c >= k


def make_folds(meta, n_folds=5):
    rng = np.random.default_rng(SEED)
    fold = {}
    for grp in (meta[meta.fail_step >= 0].structure_id.values, meta[meta.fail_step < 0].structure_id.values):
        for i, s in enumerate(rng.permutation(grp)):
            fold[s] = i % n_folds
    return fold


def lead_table(df, col):
    rows = []
    for sid, g in df.sort_values("step").groupby("structure_id"):
        al, s, st, fail = g[col].values, g.health_s.values, g.step.values, g.fail_step.iloc[0]
        valid = al & (s >= 0.02)                       # alert once damage has really begun
        first = st[valid.argmax()] if valid.any() else np.nan
        rows.append(dict(structure_id=sid, fails=fail >= 0, false_alert=bool((al & (s < 0.02)).any()),
                         first_alert=first, lead=(fail - first) if fail >= 0 else np.nan))
    return pd.DataFrame(rows)


def lead_summary(lt):
    f = lt[lt.fails]
    det = f.lead.notna()
    d = f.lead[det]
    return dict(n_failing=int(len(f)), detected=int(det.sum()), missed=int((~det).sum()),
                lead_median=float(d.median()) if len(d) else None, lead_mean=float(d.mean()) if len(d) else None,
                lead_p10_worst=float(d.quantile(0.1)) if len(d) else None,
                lead_min=float(d.min()) if len(d) else None,
                structures_with_false_alert=int(lt.false_alert.sum()), n_structures=int(len(lt)))


def cost_benefit(lt):
    f = lt[lt.fails]
    ok = int((f.lead >= MIN_LEAD).sum())
    missed = len(f) - ok
    fa = int(lt.false_alert.sum())
    reactive = len(f) * COST["reactive_failure"]
    predictive = (ok * COST["planned_repair"] + missed * COST["reactive_failure"]
                  + fa * COST["false_alarm_inspection"] + len(lt) * COST["sensor_system_per_structure"])
    return dict(assumptions=COST, min_lead_steps=MIN_LEAD, failures=len(f), caught_in_time=ok, missed=missed,
                false_alert_structures=fa, reactive_total=reactive, predictive_total=predictive,
                savings=reactive - predictive, savings_pct=100 * (reactive - predictive) / reactive,
                roi=(reactive - predictive) / (len(lt) * COST["sensor_system_per_structure"]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Simulating fleet + extracting features ...")
    df = build_feature_table(simulate_fleet())
    df.to_csv(OUT / "features.csv", index=False)
    meta = df.groupby("structure_id").fail_step.first().reset_index()
    fold = make_folds(meta)
    df["fold"] = df.structure_id.map(fold)
    X, y = df[FEATURES].values, df.label.values
    print(f"{len(df)} readings, {df.structure_id.nunique()} structures; class counts:",
          dict(zip(CLASSES, np.bincount(y))))

    # ---------- grouped 5-fold CV (no structure appears in both train and test) ----------
    proba = {m: np.zeros((len(df), 3)) for m in ["RandomForest", "GradientBoosting"]}
    ae_score = np.zeros(len(df))
    for k in range(5):
        tr, te = (df.fold != k).values, (df.fold == k).values
        sw = compute_sample_weight("balanced", y[tr])          # class-imbalance handling
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=SEED).fit(X[tr], y[tr])
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED)
        gb.fit(X[tr], y[tr], sample_weight=sw)
        proba["RandomForest"][te], proba["GradientBoosting"][te] = rf.predict_proba(X[te]), gb.predict_proba(X[te])
        # autoencoder trained ONLY on normal data -> unsupervised anomaly score
        norm = tr & (y == 0)
        sc = StandardScaler().fit(X[norm])
        Xn = sc.transform(X[norm])
        ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), max_iter=600, random_state=SEED).fit(Xn, Xn)
        thr = np.quantile(((ae.predict(Xn) - Xn) ** 2).mean(1), 0.99)
        Xt = sc.transform(X[te])
        ae_score[te] = ((ae.predict(Xt) - Xt) ** 2).mean(1) / thr

    # ---------- evaluation ----------
    res, tables = {}, {}
    for name, p in proba.items():
        pred = p.argmax(1)
        rep = classification_report(y, pred, target_names=CLASSES, output_dict=True)
        auc = {c: roc_auc_score(y == i, p[:, i]) for i, c in enumerate(CLASSES)}
        df["_al"] = sustained_by_structure(df, pred >= 1)
        lt = lead_table(df.assign(alert=df["_al"]), "alert")
        tables[name] = lt
        res[name] = dict(report=rep, roc_auc=auc, roc_auc_macro=float(np.mean(list(auc.values()))),
                         lead=lead_summary(lt), confusion=confusion_matrix(y, pred).tolist())
        print(f"\n=== {name} (out-of-fold, grouped by structure) ===")
        print(classification_report(y, pred, target_names=CLASSES, digits=3))
        print("ROC-AUC (OvR):", {k: round(v, 3) for k, v in auc.items()})
        print("Lead time:", res[name]["lead"])
    df["_al"] = sustained_by_structure(df, ae_score > 1)
    lt_ae = lead_table(df.assign(alert=df["_al"]), "alert")
    res["Autoencoder(unsupervised)"] = dict(roc_auc_anomaly=float(roc_auc_score(y > 0, ae_score)),
                                            lead=lead_summary(lt_ae))
    print("\n=== Autoencoder (unsupervised) ===", res["Autoencoder(unsupervised)"])

    # ---------- predictions for dashboard (baseline RF, out-of-fold) ----------
    p = proba["RandomForest"]
    df["pred"], df["p_normal"], df["p_warning"], df["p_critical"] = p.argmax(1), p[:, 0], p[:, 1], p[:, 2]
    df["risk_score"] = 0.5 * p[:, 1] + p[:, 2]
    df["alert"] = sustained_by_structure(df, p.argmax(1) >= 1)
    df["alert_onset"] = df.alert & ~df.groupby("structure_id").alert.shift(1, fill_value=False)
    df["ae_score"] = ae_score
    df.drop(columns=["_al"]).to_csv(OUT / "predictions.csv", index=False)

    # ---------- cost-benefit ----------
    cb = cost_benefit(tables["RandomForest"])
    print("\nCost-benefit:", json.dumps({k: v for k, v in cb.items() if k != "assumptions"}, indent=1))
    json.dump(cb, open(OUT / "cost_benefit.json", "w"), indent=1)
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1, default=float)

    # ---------- figures ----------
    figures(df, res, tables, proba["RandomForest"], y)

    # ---------- final model on all data (for API / live use) ----------
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED).fit(X, y)
    joblib.dump(dict(model=rf, features=FEATURES, classes=CLASSES), OUT / "model.joblib")
    imp = pd.Series(rf.feature_importances_, FEATURES).sort_values()
    imp.plot.barh(figsize=(6, 5), title="RandomForest feature importance")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=150); plt.close()
    print("\nDone. Outputs in", OUT)


def sustained_by_structure(df, flag):
    out = np.zeros(len(df), bool)
    for _, idx in df.groupby("structure_id").indices.items():
        idx = idx[np.argsort(df.step.values[idx])]
        out[idx] = sustained(np.asarray(flag)[idx])
    return out


def figures(df, res, tables, p, y):
    cm = np.array(res["RandomForest"]["confusion"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (RF, out-of-fold)")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y == i, p[:, i])
        plt.plot(fpr, tpr, label=f"{c} (AUC={roc_auc_score(y == i, p[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.7); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC (one-vs-rest)")
    plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    for name, lt in tables.items():
        plt.hist(lt.lead.dropna(), bins=12, alpha=.55, label=name)
    plt.axvline(MIN_LEAD, color="r", ls="--", label=f"min useful lead = {MIN_LEAD}")
    plt.xlabel("Lead time (steps before failure)"); plt.ylabel("# structures"); plt.legend()
    plt.title("Detection lead-time distribution"); plt.tight_layout()
    plt.savefig(FIG / "lead_time_hist.png", dpi=150); plt.close()

    sid = tables["RandomForest"].query("fails").sort_values("lead").structure_id.iloc[len(tables["RandomForest"].query("fails")) // 2]
    g = df[df.structure_id == sid]
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax[0].plot(g.step, g.f1_shift_raw, lw=.8, label="raw"); ax[0].plot(g.step, g.f1_shift_comp, label="temp-compensated")
    ax[0].set_ylabel("Mode-1 freq shift (%)"); ax[0].legend(); ax[0].set_title(f"{sid}: natural-frequency drift & risk")
    ax[1].plot(g.step, g.risk_score, color="crimson"); ax[1].plot(g.step, g.label / 2, "k:", label="true state")
    ax[1].scatter(g.step[g.alert_onset], g.risk_score[g.alert_onset], color="k", zorder=3, label="alert raised")
    ax[1].set_ylabel("Risk score"); ax[1].set_xlabel("Monitoring step"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "example_timeline.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
