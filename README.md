# Predictive Maintenance for Urban Infrastructure – Software Pipeline

    pip install -r requirements.txt
    python pipeline.py             # simulate -> features -> models -> metrics/figures (about 1 min)
    streamlit run dashboard.py     # risk dashboard
    uvicorn api:app --reload       # optional inference API

| Master-plan section | File |
|---|---|
| Data (synthetic strategy) | `simulate.py` |
| Preprocessing + features (low-pass, windowing, FFT/Welch, RMS/crest/kurtosis, strain, temp compensation) | `features.py` |
| Baseline RF, GradientBoosting, autoencoder, class weights, evaluation, lead time, cost-benefit | `pipeline.py` |
| Serving layer | `api.py` |
| Dashboard | `dashboard.py` |

Outputs (`outputs/`): `figures/*.png`, `metrics.json`, `cost_benefit.json`, `predictions.csv`, `features.csv`, `model.joblib`.

## Design notes for the report
* **Fleet**: 40 structures (bridges + pipelines), 300 readings each, 8 never degrade (false-alarm test).
* **Physics**: f = f0*sqrt(1-s) (stiffness loss s), damping rises with s, temperature shifts frequency, random load, noise, spike anomalies.
* **Labels**: Normal s<5%, Warning 5-15%, Critical >=15% stiffness loss.
* **Validation**: 5-fold GROUPED CV by structure (no leakage). All reported numbers are out-of-fold.
* **Alert rule**: >=3 consecutive readings predicted Warning/Critical. Lead time = failure step - first alert made after damage began.
* **Imbalance**: class-weighted RF / sample-weighted GB (SMOTE optional if imbalanced-learn is installed).
* **Autoencoder**: sklearn MLP trained on Normal data only; anomaly score = reconstruction error / 99th-percentile threshold.

## Using the Z24 dataset (not included)
Build, per structure, arrays `signals (T, N)`, `temp (T,)`, `strain (T,)`, `health_s`, `label`, `fail_step` in the same dict format as `simulate_structure`, then call `build_feature_table`. Z24 has no strain and no failure, so use its damage stages as labels and skip strain/lead-time.
