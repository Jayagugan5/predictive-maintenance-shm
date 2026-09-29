"""Synthetic SHM data generator: accelerometer window + temperature + strain per monitoring step.

Physics used (so the ML has something real to learn):
  * natural frequency f ~ sqrt(stiffness): stiffness loss s  ->  f = f0*sqrt(1-s)
  * damping ratio rises with damage
  * temperature shifts frequency slightly (operational variability -> needs compensation)
  * traffic/flow load scales vibration amplitude randomly
  * sensor noise + random spike anomalies
  * strain drifts with damage on top of thermal expansion
"""
import numpy as np
from scipy.signal import lfilter

FS = 200          # sampling rate (Hz)
WIN_S = 5         # window length (s)  -> 1000 samples
N = FS * WIN_S
T = 300           # monitoring steps per structure (1 step = 1 reading, e.g. daily)
BASE_N = 50       # first 50 readings = healthy baseline calibration
WARN_S, CRIT_S = 0.05, 0.15   # stiffness-loss thresholds for Warning / Critical labels


def _resonator(f, zeta, n, rng):
    wn = 2 * np.pi * f
    dt = 1 / FS
    r = np.exp(-zeta * wn * dt)
    wd = wn * np.sqrt(1 - zeta ** 2) * dt
    x = lfilter([1.0], [1, -2 * r * np.cos(wd), r * r], rng.standard_normal(n + 400))[400:]
    return x / x.std()


def simulate_structure(sid, rng, degrade):
    kind = rng.choice(["bridge", "pipeline"])
    f1_0 = rng.uniform(4, 9) if kind == "bridge" else rng.uniform(12, 20)
    f2_0 = f1_0 * rng.uniform(2.4, 3.0)
    t = np.arange(T)
    temp = 15 + 10 * np.sin(2 * np.pi * t / 40 + rng.uniform(0, 6.28)) + rng.normal(0, 1.5, T)

    s = np.zeros(T)
    if degrade:
        onset = int(rng.integers(90, 170))
        u = np.clip((t - onset) / (T - 1 - onset), 0, 1)
        s = rng.uniform(0.25, 0.35) * u ** rng.uniform(1.0, 2.0)
    fail_step = T - 1 if degrade else -1

    alpha = -rng.uniform(0.001, 0.002)              # freq change per deg C
    f1 = f1_0 * np.sqrt(1 - s) * (1 + alpha * (temp - 15)) * (1 + rng.normal(0, 0.002, T))
    f2 = f2_0 * np.sqrt(1 - s) * (1 + alpha * (temp - 15)) * (1 + rng.normal(0, 0.002, T))
    zeta = 0.02 + 0.06 * s

    sig = np.empty((T, N), np.float32)
    for i in range(T):
        load = rng.lognormal(0, 0.3)
        x = load * (_resonator(f1[i], zeta[i], N, rng) + 0.5 * _resonator(f2[i], zeta[i], N, rng))
        x += rng.normal(0, 0.15, N)
        if rng.random() < 0.03:                      # spike anomaly (sensor glitch / impact)
            idx = rng.integers(0, N, rng.integers(1, 4))
            x[idx] += rng.choice([-1, 1], len(idx)) * rng.uniform(6, 10)
        sig[i] = x

    strain = 8 * (temp - 15) + 400 * s + rng.normal(0, 3, T)   # microstrain
    label = np.where(s < WARN_S, 0, np.where(s < CRIT_S, 1, 2))
    return dict(structure_id=sid, kind=kind, signals=sig, temp=temp, strain=strain,
                health_s=s, label=label, fail_step=fail_step)


def simulate_fleet(n_structures=40, n_healthy=8, seed=42):
    rng = np.random.default_rng(seed)
    flags = [False] * n_healthy + [True] * (n_structures - n_healthy)
    rng.shuffle(flags)
    return [simulate_structure(f"S{i:02d}", rng, d) for i, d in enumerate(flags)]
