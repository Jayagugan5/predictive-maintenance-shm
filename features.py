"""Preprocessing + time/frequency/strain feature extraction (numpy / scipy)."""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis
from simulate import FS, BASE_N

FEATURES = ["rms_rel", "peak", "crest", "kurt", "spec_energy", "spec_centroid", "spec_entropy",
            "f1_peak", "f1_shift_raw", "f1_shift_comp", "f1_bw", "f2_shift_comp",
            "strain_resid", "strain_rate", "temp"]


def _band(f, P, fb):
    m = (f >= 0.7 * fb) & (f <= 1.25 * fb)
    Pm, fm = P[:, m], f[m]
    peak = fm[Pm.argmax(1)]
    w = Pm ** 2
    cent = (w * fm).sum(1) / w.sum(1)                       # sharpened centroid = stable freq estimate
    bw = (Pm >= 0.5 * Pm.max(1, keepdims=True)).sum(1) * (f[1] - f[0])   # half-power width ~ damping
    return peak, cent, bw


def _comp_shift(y, temp):
    """% deviation of y from a temperature-regression fitted on the healthy baseline."""
    c = np.polyfit(temp[:BASE_N], y[:BASE_N], 1)
    pred = np.polyval(c, temp)
    return 100 * (y - pred) / pred


def extract_structure(st):
    sig, temp, strain = st["signals"].astype(float), st["temp"], st["strain"]
    # 1) denoise: 4th-order zero-phase low-pass @ 60 Hz
    b, a = butter(4, 60 / (FS / 2))
    x = filtfilt(b, a, sig, axis=1)
    # 2) time-domain features
    rms = np.sqrt((x ** 2).mean(1))
    peak = np.abs(x).max(1)
    kurt = kurtosis(x, axis=1)
    # 3) frequency-domain (Welch PSD, zero-padded for fine frequency grid)
    f, P = welch(x, fs=FS, nperseg=500, noverlap=250, nfft=8192, axis=1)
    keep = f <= 60
    f, P = f[keep], P[:, keep]
    base = np.median(P[:BASE_N], axis=0)
    f1b = f[(f > 1.5)][base[f > 1.5].argmax()]
    hi = f > 1.6 * f1b
    f2b = f[hi][base[hi].argmax()]
    f1p, f1c, f1bw = _band(f, P, f1b)
    _, f2c, _ = _band(f, P, f2b)
    Pn = P / P.sum(1, keepdims=True)
    # 4) strain: remove thermal expansion using baseline regression, then rate of change
    c = np.polyfit(temp[:BASE_N], strain[:BASE_N], 1)
    resid = strain - np.polyval(c, temp)
    rate = np.zeros_like(resid)
    rate[4:] = (resid[4:] - resid[:-4]) / 4
    df = pd.DataFrame(dict(
        structure_id=st["structure_id"], kind=st["kind"], step=np.arange(len(rms)),
        rms_rel=rms / np.median(rms[:BASE_N]), peak=peak, crest=peak / rms, kurt=kurt,
        spec_energy=P.sum(1), spec_centroid=(P * f).sum(1) / P.sum(1),
        spec_entropy=-(Pn * np.log(Pn + 1e-12)).sum(1),
        f1_peak=f1p, f1_shift_raw=100 * (f1c - np.median(f1c[:BASE_N])) / np.median(f1c[:BASE_N]),
        f1_shift_comp=_comp_shift(f1c, temp), f1_bw=f1bw, f2_shift_comp=_comp_shift(f2c, temp),
        strain_resid=resid, strain_rate=rate, temp=temp,
        health_s=st["health_s"], label=st["label"], fail_step=st["fail_step"]))
    # per-structure normalisation of absolute quantities (z-score vs healthy baseline)
    for col in ["peak", "spec_energy", "spec_centroid", "spec_entropy", "f1_peak", "f1_bw"]:
        mu, sd = df[col][:BASE_N].mean(), df[col][:BASE_N].std() + 1e-9
        df[col] = (df[col] - mu) / sd
    return df


def build_feature_table(fleet):
    return pd.concat([extract_structure(s) for s in fleet], ignore_index=True)
