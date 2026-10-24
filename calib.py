"""Bench calibration of every taxel against a reference load cell, and the force estimators.

Bench (simulated): a small indenter on a reference load cell (0.01 N rms noise) presses one
taxel at a time through two up/down cycles from 0 to 20 N, at 25 C and again at 40 C.
For each taxel a linear least-squares fit in log space gives

    log G = c_T + n log F + b s        s = +1 while loading, -1 while unloading

so n is the power-law exponent, b captures hysteresis and c_25, c_40 give the temperature
coefficient. Estimators, from simplest to best:
    datasheet   one nominal curve for all taxels (R1 = 8 kohm, n = 1)
    per-taxel   per-taxel c_25 and n
    + temp      divides out (1 + alpha (T_meas - 25)) using the on-board temperature sensor
    + hyst      inverts the hysteresis with an inverse play operator using b
"""
import numpy as np
from sensor import FS, F_ACT

DT = 0.05              # bench sample period, s
TEMPS = (25.0, 40.0)
F_MIN_FIT = 0.5


def bench_profile(cycles=2, steps=40):
    up = np.linspace(0, FS, steps + 1)
    one = np.concatenate([up, up[-2::-1]])
    return np.tile(one[:-1], cycles)


def direction(F_cmd):
    """+1 while the commanded force rises, -1 while it falls (held through flat parts)."""
    s = np.zeros(len(F_cmd)); last = 1.0
    for k in range(1, len(F_cmd)):
        d = F_cmd[k] - F_cmd[k - 1]
        if d > 1e-9: last = 1.0
        elif d < -1e-9: last = -1.0
        s[k] = last
    s[0] = 1.0
    return s


def run_taxel(sensor, readout, i, j, F_cmd, T, rng, t_noise=0.3):
    """Press taxel (i, j) with F_cmd; return measured G, reference force, measured temperature."""
    sensor.reset()
    G_meas, F_ref = [], []
    F = np.zeros(sensor.shape)
    for f in F_cmd:
        F[i, j] = f
        G = sensor.step(F, DT, T)
        G_meas.append(readout.G_est(readout.frame(G))[i, j])
        F_ref.append(f + rng.normal(0, 0.01))
    F[i, j] = 0
    T_meas = T + rng.normal(0, t_noise, len(F_cmd))
    return np.array(G_meas), np.array(F_ref), T_meas


class Calibration:
    def __init__(self, shape):
        self.c25 = np.zeros(shape); self.n = np.ones(shape)
        self.b = np.zeros(shape); self.alpha = np.zeros(shape)

    def fit(self, sensor, readout, seed=3):
        rng = np.random.default_rng(seed)
        F_cmd = bench_profile(); s = direction(F_cmd)
        rows, cols = sensor.shape
        for i in range(rows):
            for j in range(cols):
                X, y = [], []
                for k, T in enumerate(TEMPS):
                    G, Fr, _ = run_taxel(sensor, readout, i, j, F_cmd, T, rng)
                    m = (Fr >= F_MIN_FIT) & (G > 0)
                    onehot = np.zeros((m.sum(), len(TEMPS))); onehot[:, k] = 1
                    X.append(np.column_stack([onehot, np.log(Fr[m]), s[m]]))
                    y.append(np.log(G[m]))
                coef, *_ = np.linalg.lstsq(np.vstack(X), np.concatenate(y), rcond=None)
                c25, c40, n, b = coef
                self.c25[i, j], self.n[i, j], self.b[i, j] = c25, n, b
                self.alpha[i, j] = (np.exp(c40 - c25) - 1) / (TEMPS[1] - TEMPS[0])
        return self


class Estimator:
    """Turns measured conductances into forces, one frame at a time (keeps hysteresis state)."""
    MODES = ("datasheet", "per-taxel", "per-taxel + temp", "per-taxel + temp + hyst")

    def __init__(self, cal, mode):
        self.cal, self.mode = cal, mode
        self.reset()

    EPS = 0.02

    def reset(self):
        self.qref = np.zeros(self.cal.c25.shape)
        self.dir = np.ones(self.cal.c25.shape)

    def __call__(self, G, T_meas=25.0):
        cal, mode = self.cal, self.mode
        G = np.maximum(G, 1e-12)
        if mode == "datasheet":
            F = G * 8e3
        else:
            if "temp" in mode:
                G = G / (1 + cal.alpha * (T_meas - 25.0))
            q = np.exp((np.log(G) - cal.c25) / cal.n)      # mid-line force
            if "hyst" in mode:
                # Inverse of the play operator: a rising reading means the loading branch
                # (true force above the mid-line), a falling one the unloading branch.
                # Changes smaller than EPS are treated as noise and keep the last branch.
                k = np.abs(cal.b) / cal.n
                up = q > self.qref * (1 + self.EPS)
                down = q < self.qref * (1 - self.EPS)
                self.dir = np.where(up, 1.0, np.where(down, -1.0, self.dir))
                self.qref = np.where(up | down, q, self.qref)
                F = q * np.exp(k * self.dir)
            else:
                F = q
        return np.where(F < F_ACT, 0.0, F)


def eval_profile(rng, levels=24):
    """Random staircase between 0.5 and 20 N with returns to zero, 0.5 s per level."""
    seq = []
    for _ in range(levels):
        f = rng.uniform(F_MIN_FIT, FS)
        seq += list(np.linspace(seq[-1] if seq else 0, f, 4)) + [f] * 10
        if rng.random() < 0.3:
            seq += list(np.linspace(f, 0, 4)) + [0] * 4
    return np.array(seq)


def evaluate(sensor, readout, cal, T, taxels, seed=7):
    """RMS force error (% of full scale) for each estimator, over the listed taxels."""
    rng = np.random.default_rng(seed)
    err = {m: [] for m in Estimator.MODES}
    for (i, j) in taxels:
        F_cmd = eval_profile(rng)
        G, _, T_meas = run_taxel(sensor, readout, i, j, F_cmd, T, rng)
        for m in Estimator.MODES:
            est = Estimator(cal, m)
            Fe = np.array([est(np.full(sensor.shape, g), t)[i, j] for g, t in zip(G, T_meas)])
            keep = F_cmd >= F_MIN_FIT
            err[m].append(Fe[keep] - F_cmd[keep])
    return {m: 100 * np.sqrt(np.mean(np.concatenate(e) ** 2)) / FS for m, e in err.items()}
