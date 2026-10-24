"""Run-time self-tests for the zero-potential readout, plus the fault-injection campaigns.

Checks done on every frame (1 kHz):
  reference column   each row's 1 kohm reference reads within 50-150 % of its power-up value,
                     else "row fault" (open driver or line, or a short pulling the row down)
  reference row      each column's reference reads within 50-150 %, else "column fault"
                     (only if the design has a reference row)
  saturation         a TIA output at full scale for 3 frames: shorted taxel or overload
  frozen data        3 frames with every ADC code identical to the last frame
  dead taxel         reads 0 N for 20 frames while two opposite neighbours read more than
                     0.3 N: an open taxel inside a contact patch
"""
import numpy as np
from readout import ADC_MAX
from sensor import contact_forces

FRAME_DT = 1e-3


class Monitor:
    def __init__(self, zp, startup_codes):
        self.zp = zp
        r, c = zp.rows, zp.cols
        self.ref_col0 = startup_codes[:, c].copy()                     # per driven row
        self.ref_row0 = startup_codes[r, :].copy() if zp.ref_row else None
        self.cnt = {}
        self.prev = None
        self.flags = set()

    def _count(self, key, cond, need):
        self.cnt[key] = self.cnt.get(key, 0) + 1 if cond else 0
        if self.cnt[key] >= need:
            self.flags.add(key)

    def update(self, codes, F):
        zp = self.zp
        r, c = zp.rows, zp.cols
        ratio = codes[:, c] / np.maximum(self.ref_col0, 1)
        for i in range(zp.R):
            self._count(("row fault", i), not 0.5 <= ratio[i] <= 1.5, 2)
        if zp.ref_row:
            ratio = codes[r, :] / np.maximum(self.ref_row0, 1)
            for j in range(zp.C):
                self._count(("column fault", j), not 0.5 <= ratio[j] <= 1.5, 2)
        self._count(("saturated",), codes[:r, :c].max() >= ADC_MAX, 3)
        self._count(("frozen ADC",), self.prev is not None and np.array_equal(codes, self.prev), 3)
        self.prev = codes
        # A healthy taxel lying between two live neighbours (left and right, or above and
        # below) cannot read zero: the contact footprint is log-concave, so its force is at
        # least the smaller of the two. Corner taxels have no opposite pair and are not covered.
        pad = np.pad(F > 0.3, 1)
        lr = pad[1:-1, :-2] & pad[1:-1, 2:]
        ud = pad[:-2, 1:-1] & pad[2:, 1:-1]
        dead = (F <= 0) & (lr | ud)
        self.dead_cnt = np.where(dead, getattr(self, "dead_cnt", 0) + 1, 0)
        for i, j in zip(*np.nonzero(self.dead_cnt >= 20)):
            self.flags.add(("dead taxel", int(i), int(j)))
        return self.flags


class Rig:
    """Sensor + readout + estimator + monitor, stepped one 1 ms frame at a time."""

    def __init__(self, sensor, zp, est, T=25.0, seed=11):
        self.s, self.zp, self.est, self.T = sensor, zp, est, T
        self.rng = np.random.default_rng(seed)
        sensor.reset(); est.reset()
        start = np.mean([zp.frame(sensor.static_G(np.zeros(sensor.shape), T)) for _ in range(20)], axis=0)
        self.mon = Monitor(zp, start)

    def step(self, F):
        G = self.s.step(F, FRAME_DT, self.T)
        codes = self.zp.frame(G)
        Fe = self.est(self.zp.G_est(codes), self.T + self.rng.normal(0, 0.3))
        flags = self.mon.update(codes, Fe)
        return Fe, flags


def press(cx, cy, total, ramp=20, hold=100, release=10, rest=20):
    """Force frames (1 ms each) for one indentation."""
    peak = contact_forces(cx, cy, total)
    k = np.concatenate([np.linspace(0, 1, ramp), np.ones(hold), np.linspace(1, 0, release), np.zeros(rest)])
    return [a * peak for a in k]


def random_traffic(rng, frames, rows=8, cols=8):
    out = []
    while len(out) < frames:
        out += press(rng.uniform(0, cols - 1), rng.uniform(0, rows - 1), rng.uniform(3, 15),
                     hold=int(rng.integers(60, 200)))
    return out[:frames]
