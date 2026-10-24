"""Piezoresistive force-sensitive-resistor (FSR) taxel array model.

Each taxel's conductance follows G = F_eff^n / R1 above an activation force, with:
  - part-to-part spread in R1 (resistance at 1 N), exponent n, hysteresis and temperature coefficient
  - hysteresis: a play (backlash) operator, so loading reads low and unloading reads high
  - creep: conductance slowly rises under constant load (first-order, time constant tau)
  - temperature: conductance scales by (1 + alpha (T - 25 C))
  - faults: open taxel (no conduction) and shorted taxel (about 1 ohm)
Parameters are in the range of published FSR datasheet curves; they are not a specific part.
"""
import numpy as np

FS = 20.0          # full-scale force per taxel, N
F_ACT = 0.2        # activation force, N
R_OFF = 10e6       # unpressed resistance, ohms


class FSRArray:
    def __init__(self, rows=8, cols=8, seed=1, creep=0.05, tau=20.0):
        rng = np.random.default_rng(seed)
        s = (rows, cols)
        self.shape = s
        self.R1 = 8e3 * rng.normal(1, 0.15, s)            # ohms at 1 N
        self.n = rng.normal(1.0, 0.05, s)                 # power-law exponent
        self.h = np.clip(rng.normal(0.08, 0.01, s), 0.03, None)   # +/- hysteresis fraction
        self.alpha = rng.normal(0.004, 0.0004, s)         # conductance tempco, 1/C
        self.creep_amp, self.tau = creep, tau
        self.open = np.zeros(s, bool)
        self.short = np.zeros(s, bool)
        self.reset()

    def reset(self):
        self.p = np.zeros(self.shape)   # play-operator state (force-equivalent)
        self.x = np.zeros(self.shape)   # creep state

    def static_G(self, F, T=25.0):
        """Conductance with no hysteresis or creep (used for circuit-only tests)."""
        F = np.asarray(F, float)
        on = F > F_ACT
        G = 1 / R_OFF + np.where(on, np.power(np.maximum(F, 1e-9), self.n) / self.R1, 0.0)
        return self._faults(G * (1 + self.alpha * (T - 25.0)))

    def step(self, F, dt, T=25.0):
        F = np.clip(np.asarray(F, float), 0, None)
        self.p = np.clip(self.p, F * (1 - self.h), F * (1 + self.h))
        target = np.where(F > F_ACT, self.creep_amp, 0.0)
        self.x += (target - self.x) * (1 - np.exp(-dt / self.tau))
        fe = self.p * (1 + self.x)
        on = fe > F_ACT
        G = 1 / R_OFF + np.where(on, np.power(np.maximum(fe, 1e-9), self.n) / self.R1, 0.0)
        return self._faults(G * (1 + self.alpha * (T - 25.0)))

    def _faults(self, G):
        G = np.where(self.open, 1e-12, G)
        return np.where(self.short, 1.0, G)


def contact_forces(cx, cy, total, rows=8, cols=8, pitch=4.0, sigma=3.0):
    """Force on each taxel from a round indenter with a Gaussian pressure footprint.

    cx, cy are in taxel units (column, row). Returns an array of forces in N.
    """
    yy, xx = np.mgrid[0:rows, 0:cols]
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) * pitch ** 2
    return total * pitch ** 2 / (2 * np.pi * sigma ** 2) * np.exp(-d2 / (2 * sigma ** 2))
