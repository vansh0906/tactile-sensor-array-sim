"""Run every experiment, print the requirement table, save results.png and the SPICE netlists.

    python run_all.py            # about 2 minutes
Exit code is 1 if any requirement fails, so CI catches regressions.
"""
import sys
import numpy as np
from sensor import FSRArray, FS, contact_forces
from readout import (ZeroPotential, scan_divider, divider_circuit, divider_G_est,
                     frame_time_divider, G_REF)
from calib import Calibration, Estimator, run_taxel, direction, DT
from diagnostics import Rig, press, random_traffic, FRAME_DT

PITCH = 4.0
BEST = "per-taxel + temp + hyst"
req = []


def check(rid, text, value, limit, unit, ok):
    req.append((rid, text, value, limit, unit, ok))


def section(t):
    print(f"\n[{t}]")


# ---------------------------------------------------------------- 1. readout circuits
section("1  Readout circuits: ghosting and loading (no ADC noise, true sensor curve)")
sensor = FSRArray()


def G_to_F(G):
    return np.where(G > 2e-6, (np.maximum(G, 1e-12) * sensor.R1) ** (1 / sensor.n), 0.0)


def zp_noiseless(G, zp):
    v = np.abs(zp.solve(G))
    return G_REF * v[:zp.rows, :zp.cols] / v[:zp.rows, zp.cols:zp.cols + 1]


zp = ZeroPotential()
F3 = np.zeros((8, 8)); F3[1, 1] = F3[1, 5] = F3[5, 1] = 10.0
G3 = sensor.static_G(F3)
maps = {"A divider, floating lines": G_to_F(scan_divider(G3, False)),
        "B divider, grounded rows": G_to_F(scan_divider(G3, True)),
        "C zero potential (TIA)": G_to_F(zp_noiseless(G3, zp))}
print("Three taxels pressed at 10 N: (1,1) (1,5) (5,1). Taxel (5,5) is untouched.")
print(f"  {'method':28s} {'reads on (5,5)':>15s} {'reads on (1,1)':>15s}")
for k, m in maps.items():
    print(f"  {k:28s} {m[5, 5]:12.3f} N {m[1, 1]:12.2f} N")

rng = np.random.default_rng(21)
worst = {k: [0.0, 0.0] for k in maps}
for _ in range(40):
    F = np.zeros((8, 8))
    idx = rng.choice(64, int(rng.integers(3, 6)), replace=False)
    F.flat[idx] = rng.uniform(2, FS, len(idx))
    G = sensor.static_G(F)
    pressed = F > 0
    readers = (lambda G, ij: G_to_F(divider_G_est(divider_circuit(G, ij, False).solve()[0]["out"]))[ij],
               lambda G, ij: G_to_F(divider_G_est(divider_circuit(G, ij, True).solve()[0]["out"]))[ij],
               lambda G, ij: G_to_F(zp_noiseless(G, zp))[ij])
    for k, est, read in zip(maps, (G_to_F(scan_divider(G, False)), G_to_F(scan_divider(G, True)),
                                   G_to_F(zp_noiseless(G, zp))), readers):
        worst[k][0] = max(worst[k][0], 100 * est[~pressed].max() / FS)
        for ij in zip(*np.nonzero(pressed)):
            alone = sensor.static_G(np.where(np.arange(64).reshape(8, 8) == ij[0] * 8 + ij[1], F, 0))
            worst[k][1] = max(worst[k][1], 100 * abs(est[ij] / read(alone, ij) - 1))
print("40 random patterns with 3-5 taxels pressed at 2-20 N:")
print(f"  {'method':28s} {'worst ghost':>12s} {'worst crosstalk on pressed':>28s}")
print("  (crosstalk = change in a pressed taxel's reading caused by the other pressed taxels)")
for k, (g, p) in worst.items():
    print(f"  {k:28s} {g:9.2f} %FS {p:25.2f} %")
check("R3", "Ghost reading on untouched taxels (zero potential)", worst["C zero potential (TIA)"][0], 1.0, "%FS", worst["C zero potential (TIA)"][0] <= 1.0)
check("R4", "Crosstalk on pressed taxels (zero potential)", worst["C zero potential (TIA)"][1], 1.0, "%", worst["C zero potential (TIA)"][1] <= 1.0)

# SPICE netlists for LTspice (same circuits, selected taxel (5,5) / row 5)
notes = []
for name, gu in (("A_divider_floating", False), ("B_divider_grounded", True)):
    c = divider_circuit(G3, (5, 5), gu)
    v, _ = c.solve()
    open(f"spice/{name}.cir", "w").write(c.netlist([f"Python MNA result: V(out) = {v['out']:.6f} V"]))
    notes.append(f"  spice/{name}.cir   V(out) = {v['out']:.6f} V")
c = zp.circuit(G3, 5)
v, _ = c.solve()
open("spice/C_zero_potential.cir", "w").write(c.netlist(
    [f"Python MNA result: V(o1) = {v['o1']:.6f} V, V(o5) = {v['o5']:.6f} V, V(o8) = {v['o8']:.6f} V"]))
notes.append(f"  spice/C_zero_potential.cir   V(o1) = {v['o1']:.6f} V  V(o5) = {v['o5']:.6f} V  V(o8) = {v['o8']:.6f} V")
print("SPICE netlists written (open in LTspice, run, compare the .op values):")
print("\n".join(notes))

# ---------------------------------------------------------------- 2. calibration
section("2  Calibration against the reference load cell (all 64 taxels)")
sensor = FSRArray(); zp = ZeroPotential()
cal = Calibration(sensor.shape).fit(sensor, zp)
h_est = np.tanh(np.abs(cal.b) / cal.n)
print(f"  exponent n: fitted vs true, mean abs diff {np.mean(np.abs(cal.n - sensor.n)):.3f}")
print(f"  temperature coefficient: fitted vs true, mean abs diff {1e6 * np.mean(np.abs(cal.alpha - sensor.alpha)):.0f} ppm/C "
      f"(true values {1e6 * sensor.alpha.min():.0f}-{1e6 * sensor.alpha.max():.0f} ppm/C)")
print(f"  hysteresis: fitted {100 * h_est.mean():.1f} % vs true {100 * sensor.h.mean():.1f} % (array mean)")

from calib import evaluate
taxels = [(i, j) for i in range(8) for j in range(8)]
table = {T: evaluate(sensor, zp, cal, T, taxels, seed=7 + int(T)) for T in (15.0, 25.0, 40.0)}
print("RMS force error over 0.5-20 N, % of full scale (20 N), new random load profiles:")
print(f"  {'estimator':26s} {'15 C':>7s} {'25 C':>7s} {'40 C':>7s}")
for m in Estimator.MODES:
    print(f"  {m:26s} " + " ".join(f"{table[T][m]:7.2f}" for T in table))
check("R1", "Force error at 25 C, 0.5-20 N", table[25.0][BEST], 5.0, "%FS", table[25.0][BEST] <= 5.0)
w = max(table[15.0][BEST], table[40.0][BEST])
check("R2", "Force error at 15 C and 40 C (temperature compensated)", w, 5.0, "%FS", w <= 5.0)

# hysteresis and creep on one taxel
i, j = 2, 3
tri = np.concatenate([np.linspace(0, FS, 81), np.linspace(FS, 0, 81)[1:]])
G, _, Tm = run_taxel(sensor, zp, i, j, tri, 25.0, np.random.default_rng(4))
hyst = {}
for m in ("per-taxel + temp", BEST):
    est = Estimator(cal, m)
    Fe = np.array([est(np.full(sensor.shape, g), t)[i, j] for g, t in zip(G, Tm)])
    up, down = Fe[:81], Fe[80:][::-1]
    hyst[m] = 100 * np.mean(np.abs(up - down)[tri[:81] >= 0.5]) / FS
print(f"  Hysteresis (mean up/down gap, 0-20-0 N): {hyst['per-taxel + temp']:.1f} %FS static, {hyst[BEST]:.1f} %FS with compensation")
hold = np.concatenate([np.linspace(0, 10, 10), np.full(int(60 / DT), 10.0)])
G, _, Tm = run_taxel(sensor, zp, i, j, hold, 25.0, np.random.default_rng(5))
est = Estimator(cal, BEST)
Fe = np.array([est(np.full(sensor.shape, g), t)[i, j] for g, t in zip(G, Tm)])
k1 = 10 + int(1 / DT)
creep = 100 * (np.mean(Fe[-20:]) / np.mean(Fe[k1:k1 + 20]) - 1)
print(f"  Creep: 10 N held 60 s, reading rises {creep:.1f} % from t = 1 s to t = 60 s (not compensated)")

# ---------------------------------------------------------------- 3. timing
section("3  Scan timing (8 us ADC conversion, 10 us TIA settling per row, 2 us mux settling)")
t_div = frame_time_divider(); t_v1 = ZeroPotential(ref_row=False).frame_time(); t_v2 = zp.frame_time()
print(f"  divider (64 conversions)                      {1e6 * t_div:6.0f} us  -> {1 / t_div:5.0f} Hz")
print(f"  zero potential, reference column (8 x 9)       {1e6 * t_v1:6.0f} us  -> {1 / t_v1:5.0f} Hz")
print(f"  zero potential, reference row + column (9 x 9) {1e6 * t_v2:6.0f} us  -> {1 / t_v2:5.0f} Hz")
check("R5", "Full frame time incl. reference row and column", 1e3 * t_v2, 1.0, "ms", t_v2 <= 1e-3)

# ---------------------------------------------------------------- 4. contact estimation
section("4  Contact location and total force (300 presses, 4 mm pitch, Gaussian footprint sigma 3 mm)")
rig = Rig(FSRArray(), ZeroPotential(), Estimator(cal, BEST))
yy, xx = np.mgrid[0:8, 0:8]


def presses(rng, n):
    out = []
    for _ in range(n):
        cx, cy, tot = rng.uniform(1.5, 5.5), rng.uniform(1.5, 5.5), rng.uniform(2, 15)
        for k, F in enumerate(press(cx, cy, tot)):
            Fe, _ = rig.step(F)
            if k == 20 + 99:                   # last frame of the hold
                snap = Fe.copy()
        out.append((snap, cx, cy, tot))
    return out


# contact-level calibration: 100 presses on the reference load cell fit total = a * sum + b
calset = presses(np.random.default_rng(1), 100)
a, b = np.polyfit([c[0].sum() for c in calset], [c[3] for c in calset], 1)
test = presses(np.random.default_rng(31), 300)
loc = np.array([PITCH * np.hypot((sn * xx).sum() / sn.sum() - cx, (sn * yy).sum() / sn.sum() - cy)
                for sn, cx, cy, _ in test])
raw = np.array([100 * (sn.sum() / tot - 1) for sn, _, _, tot in test])
fixed = np.array([100 * ((a * sn.sum() + b) / tot - 1) for sn, _, _, tot in test])
sn, cx, cy, tot = test[3]
example = (sn, cx, cy, (sn * xx).sum() / sn.sum(), (sn * yy).sum() / sn.sum(), tot)
rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
print(f"  location error (force-weighted centroid): RMS {rms(loc):.2f} mm, 95th percentile {np.percentile(loc, 95):.2f} mm")
print(f"  total force, sum of taxels: RMS error {rms(raw):.1f} %, mean {raw.mean():+.1f} % "
      "(taxels below the 0.2 N activation force are lost)")
print(f"  total force after contact-level calibration (total = {a:.3f} x sum + {b:.2f} N, fit on a "
      f"separate 100-press set): RMS error {rms(fixed):.1f} %, worst {np.abs(fixed).max():.1f} %")
check("R6", "Contact location error (RMS)", rms(loc), 1.0, "mm", rms(loc) <= 1.0)
check("R7", "Total contact force error (RMS)", rms(fixed), 10.0, "%", rms(fixed) <= 10.0)

# ---------------------------------------------------------------- 5. fault injection
section("5  Fault injection (1 kHz frames)")


def latency(make_fault, design_ref_row=True, frames=None, limit=50, seed=0):
    s = FSRArray(); z = ZeroPotential(ref_row=design_ref_row)
    rig = Rig(s, z, Estimator(cal, BEST), seed=seed)
    for _ in range(5):
        rig.step(np.zeros((8, 8)))
    make_fault(s, z)
    frames = frames if frames is not None else [np.zeros((8, 8))] * limit
    for k, F in enumerate(frames):
        _, fl = rig.step(F)
        if fl:
            return k + 1, fl
    return None, set()


def campaign(name, cases, design_ref_row=True, frame_fn=None, limit=50):
    lat = []
    for n, fault in enumerate(cases):
        frames = frame_fn(n) if frame_fn else None
        k, _ = latency(fault, design_ref_row, frames, limit, seed=n)
        lat.append(k)
    det = [x for x in lat if x is not None]
    return len(det) / len(lat), (max(det) if det else None), len(lat)


def short(i, j): return lambda s, z: s.short.__setitem__((i, j), True)
def open_tax(i, j): return lambda s, z: s.open.__setitem__((i, j), True)
def row_open(i): return lambda s, z: z.row_open.__setitem__(i, True)
def col_open(j): return lambda s, z: z.col_open.__setitem__(j, True)
def frozen(): return lambda s, z: setattr(z, "frozen", True)


corners = {(0, 0), (0, 7), (7, 0), (7, 7)}
inner = [t for t in taxels if t not in corners]
results = {}
for design in (False, True):
    tag = "v2 (ref row + col)" if design else "v1 (ref col only)"
    r = {}
    r["shorted taxel"] = campaign("short", [short(*t) for t in taxels], design)
    r["open row line"] = campaign("row", [row_open(i) for i in range(8)], design)
    r["open column line"] = campaign("col", [col_open(j) for j in range(8)], design)
    r["frozen ADC"] = campaign("frozen", [frozen() for _ in range(10)], design)
    results[tag] = r
    print(f"  design {tag}: detected within 50 ms (worst latency)")
    for k, (cov, worst_lat, n) in r.items():
        print(f"    {k:18s} {100 * cov:5.0f} % of {n:2d} cases   worst {worst_lat if worst_lat else '-'} ms")
v2 = results["v2 (ref row + col)"]
w5 = all(cov == 1.0 and lat <= 5 for cov, lat, _ in v2.values())
check("R8", "Short, open row, open column, frozen ADC flagged (v2)", max(lat for _, lat, _ in v2.values()), 5, "ms", w5)

cov_c, lat_c, _ = campaign("open centred", [open_tax(*t) for t in inner], True,
                           frame_fn=lambda n: press(inner[n][1], inner[n][0], 5.0), limit=150)
print(f"  open taxel, 5 N press centred on it: {100 * cov_c:.0f} % of {len(inner)} non-corner taxels, worst {lat_c} ms after contact starts")
check("R9", "Open taxel flagged during a 5 N contact (60 non-corner)", lat_c if cov_c == 1 else float("inf"), 100, "ms", cov_c == 1 and lat_c <= 100)

traffic_rng = np.random.default_rng(77)
cov_t, _, _ = campaign("open traffic", [open_tax(*t) for t in taxels], True,
                       frame_fn=lambda n: random_traffic(np.random.default_rng(1000 + n), 2000), limit=2000)
print(f"  open taxel under 2 s of random contacts: flagged in {100 * cov_t:.0f} % of 64 cases")

rig = Rig(FSRArray(), ZeroPotential(), Estimator(cal, BEST), T=30.0, seed=99)
false = set()
for F in random_traffic(np.random.default_rng(88), 10000):
    _, fl = rig.step(F)
    false |= fl
print(f"  false flags in 10,000 frames of normal contacts at 30 C: {len(false)}")
check("R10", "False flags in 10,000 normal frames", len(false), 0, "", len(false) == 0)

# ---------------------------------------------------------------- 6. fault tree
section("6  Fault tree: undetected wrong force data (failure rates are assumed, in FIT = failures per 1e9 h)")
lam = {"open taxel": (5, 64), "shorted taxel": (2, 64), "open row line": (10, 8),
       "open column line": (10, 8), "frozen ADC": (20, 1)}
print("  TOP: wrong force data reaches the controller without a fault flag  (OR gate)")
for tag, r in results.items():
    cov = {k: r[k][0] for k in r}
    cov["open taxel"] = cov_t
    tot = sum(a * n for a, n in lam.values())
    resid = {k: a * n * (1 - cov[k]) for k, (a, n) in lam.items()}
    print(f"  design {tag}: total {tot} FIT, residual undetected {sum(resid.values()):.0f} FIT "
          f"(diagnostic coverage {100 * (1 - sum(resid.values()) / tot):.0f} %)")
    for k, (a, n) in lam.items():
        print(f"     +- {k:17s} {n:2d} x {a:2d} FIT = {a * n:4d}   coverage {100 * cov[k]:5.1f} %   residual {resid[k]:6.1f} FIT")

# ---------------------------------------------------------------- requirement table
print("\nID   Requirement                                              Result      Limit    Status")
for rid, text, val, lim, unit, ok in sorted(req, key=lambda r: int(r[0][1:])):
    vs = f"{val:.2f}" if isinstance(val, float) else str(val)
    print(f"{rid:4s} {text:56s} {vs:>7s} {unit:4s} <= {lim:<5} {'PASS' if ok else 'FAIL'}")
n_ok = sum(r[-1] for r in req)
print(f"{n_ok}/{len(req)} requirements pass")

# ---------------------------------------------------------------- plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    s = FSRArray(); z = ZeroPotential()
    for T, col in ((25.0, "tab:blue"), (40.0, "tab:red")):
        G, Fr, _ = run_taxel(s, z, 2, 3, tri, T, np.random.default_rng(6))
        ax[0, 0].plot(tri, 1e3 * G, color=col, label=f"{T:.0f} C (up: lower branch, down: upper)")
    ax[0, 0].set(xlabel="applied force (N)", ylabel="measured conductance (mS)",
                 title="One taxel: hysteresis loop and temperature shift")
    ax[0, 0].legend(fontsize=8)
    xs = np.arange(3)
    for k, m in enumerate(Estimator.MODES):
        ax[0, 1].bar(xs + (k - 1.5) * 0.2, [table[T][m] for T in table], 0.2, label=m)
    ax[0, 1].axhline(5, color="k", ls="--", lw=1)
    ax[0, 1].set(xticks=xs, xticklabels=["15 C", "25 C", "40 C"], ylabel="RMS error (% of 20 N)",
                 title="Force error by estimator (all 64 taxels)")
    ax[0, 1].set_ylim(0, 1.35 * max(max(v.values()) for v in table.values()))
    ax[0, 1].legend(fontsize=8, ncol=2, loc="upper center")
    for a, (k, m) in zip((ax[0, 2], ax[1, 0], ax[1, 1]), maps.items()):
        im = a.imshow(m, vmin=0, vmax=10, cmap="viridis")
        a.add_patch(plt.Rectangle((4.5, 4.5), 1, 1, fill=False, ec="w", lw=2))
        a.set_title(f"{k}\nuntouched (5,5) reads {m[5, 5]:.2f} N", fontsize=10)
        a.set(xlabel="column", ylabel="row")
    fig.colorbar(im, ax=ax[1, 1], label="estimated force (N)")
    snap, cx, cy, ex, ey, tot = example
    ax[1, 2].imshow(snap, cmap="magma")
    ax[1, 2].plot(cx, cy, "+", color="lime", ms=18, mew=2.5, label="true centre")
    ax[1, 2].plot(ex, ey, "o", mfc="none", mec="deepskyblue", ms=16, mew=2.5, label="estimated")
    ax[1, 2].set(title=f"Contact estimate ({tot:.1f} N press)", xlabel="column", ylabel="row")
    ax[1, 2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results.png", dpi=110)
    print("saved results.png")
except ImportError:
    print("matplotlib not installed; skipped the plot")

sys.exit(0 if n_ok == len(req) else 1)
