"""Scan circuits for a row/column (passive matrix) taxel array.

A  divider, floating:  selected row driven to VCC, other lines left floating, selected
                       column read through a 1 kohm sense resistor. Cheapest; ghosting.
B  divider, grounded:  same, but unselected rows driven to 0 V.
C  zero potential:     every column held at virtual ground by its own transimpedance amplifier
                       (TIA), unselected rows driven to 0 V, so no current can take a sneak
                       path. Adds a reference column (1 kohm per row) for ratiometric readout
                       and, optionally, a reference row for self-test of each column.

Values: VCC 3.3 V, driver output 30 ohm, mux on-resistance 5 ohm, TIA Rf 330 ohm (1 %),
op-amp open-loop gain 1e5, 12-bit ADC with 1.5 LSB rms noise.
"""
import numpy as np
from circuit import Circuit

VCC, R_DRV, R_ON, R_SENSE = 3.3, 30.0, 5.0, 1e3
RF, A_OL, R_COL = 330.0, 1e5, 0.05
G_REF = 1e-3                 # reference resistor 1 kohm (0.1 %)
ADC_MAX, VREF, ADC_NOISE = 4095, 3.3, 1.5
T_ADC, T_SETTLE_DIV, T_SETTLE_TIA = 8e-6, 2e-6, 10e-6
OPEN = 1e12


# ---------- methods A and B: one netlist per selected taxel ----------
def divider_circuit(G, sel, ground_unselected):
    rows, cols = G.shape
    i0, j0 = sel
    c = Circuit(f"Passive matrix, divider readout, {'grounded' if ground_unselected else 'floating'} "
                f"unselected lines, selected taxel r{i0} c{j0}")
    for i in range(rows):
        for j in range(cols):
            c.r(f"Rt{i}_{j}", f"r{i}", f"c{j}", 1 / G[i, j])
    for i in range(rows):
        if i == i0:
            c.v(f"Vd{i}", f"d{i}", 0, VCC); c.r(f"Rdrv{i}", f"d{i}", f"r{i}", R_DRV)
        elif ground_unselected:
            c.v(f"Vd{i}", f"d{i}", 0, 0.0); c.r(f"Rdrv{i}", f"d{i}", f"r{i}", R_DRV)
    c.r("Rmux", f"c{j0}", "out", R_ON)
    c.r("Rsense", "out", 0, R_SENSE)
    return c


def divider_G_est(v_out):
    return v_out / (R_SENSE * (VCC - v_out))


def scan_divider(G, ground_unselected):
    """Estimated conductance of every taxel with method A or B (one MNA solve per taxel)."""
    est = np.zeros_like(G)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            v, _ = divider_circuit(G, (i, j), ground_unselected).solve()
            est[i, j] = divider_G_est(v["out"])
    return est


# ---------- method C: zero-potential readout with reference row/column ----------
class ZeroPotential:
    def __init__(self, rows=8, cols=8, ref_row=True, seed=2):
        rng = np.random.default_rng(seed)
        self.rows, self.cols, self.ref_row = rows, cols, ref_row
        self.R = rows + (1 if ref_row else 0)      # driven lines
        self.C = cols + 1                          # sensed lines (last = reference column)
        self.rf = RF * rng.normal(1, 0.01, self.C)
        self.rng = np.random.default_rng(seed + 100)
        self.row_open = np.zeros(self.R, bool)
        self.col_open = np.zeros(self.C, bool)
        self.frozen = False
        self._last = None

    def full_G(self, G):
        """Add the reference column (and row) to the sensor conductances."""
        Gf = np.full((self.R, self.C), G_REF)
        Gf[:self.rows, :self.cols] = G
        return Gf

    def circuit(self, G, sel_row):
        """Explicit netlist (V sources, resistors, VCVS op-amps) for one row step."""
        Gf = self.full_G(G)
        c = Circuit(f"Zero-potential readout, row {sel_row} selected")
        for i in range(self.R):
            c.v(f"Vd{i}", f"d{i}", 0, VCC if i == sel_row else 0.0)
            c.r(f"Rdrv{i}", f"d{i}", f"r{i}", OPEN if self.row_open[i] else R_DRV)
            for j in range(self.C):
                c.r(f"Rt{i}_{j}", f"r{i}", f"c{j}", 1 / Gf[i, j])
        for j in range(self.C):
            c.r(f"Rcol{j}", f"c{j}", f"t{j}", OPEN if self.col_open[j] else R_COL)
            c.r(f"Rf{j}", f"t{j}", f"o{j}", self.rf[j])
            c.e(f"Eamp{j}", f"o{j}", 0, 0, f"t{j}", A_OL)   # o = A (0 - v_t): inverting TIA
        return c

    def solve(self, G):
        """TIA output voltages, shape (driven rows, sensed columns), one nodal solve per frame.

        The op-amp is folded into a conductance (1 + A)/Rf from its input to ground and
        each driver into a Norton source, so all row steps share one matrix.
        """
        Gf = self.full_G(G)
        R, C = self.R, self.C
        n = R + 2 * C                         # nodes: rows, column lines, TIA inputs
        Y = np.zeros((n, n))
        gdrv = np.where(self.row_open, 1 / OPEN, 1 / R_DRV)
        ri, cj = np.arange(R), R + np.arange(C)
        tj = R + C + np.arange(C)
        Y[ri, ri] += gdrv + Gf.sum(1)
        Y[cj, cj] += Gf.sum(0)
        Y[np.ix_(ri, cj)] -= Gf
        Y[np.ix_(cj, ri)] -= Gf.T
        gcol = np.where(self.col_open, 1 / OPEN, 1 / R_COL)
        Y[cj, cj] += gcol; Y[tj, tj] += gcol; Y[cj, tj] -= gcol; Y[tj, cj] -= gcol
        Y[tj, tj] += (1 + A_OL) / self.rf
        I = np.zeros((n, R))
        I[ri, ri] = VCC * gdrv
        X = np.linalg.solve(Y, I)
        return (-A_OL * X[tj, :]).T           # (row step, column)

    def adc(self, vout):
        if self.frozen and self._last is not None:
            return self._last.copy()
        v = np.clip(np.abs(vout), 0, VREF)    # TIA output saturates at the rail
        code = np.rint(v / VREF * ADC_MAX + self.rng.normal(0, ADC_NOISE, v.shape))
        code = np.clip(code, 0, ADC_MAX)
        self._last = code
        return code.copy()

    def frame(self, G):
        """Scan one frame: returns ADC codes (driven rows x sensed columns)."""
        return self.adc(self.solve(G))

    def G_est(self, codes):
        """Ratiometric conductance: each row is scaled by its own reference-column reading."""
        ref = np.maximum(codes[:self.rows, self.cols:self.cols + 1], 1.0)
        return G_REF * codes[:self.rows, :self.cols] / ref

    def frame_time(self):
        return self.R * (T_SETTLE_TIA + self.C * T_ADC)


def frame_time_divider(rows=8, cols=8):
    return rows * cols * (T_SETTLE_DIV + T_ADC)
