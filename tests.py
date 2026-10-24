"""Quick self-checks (a few seconds). Run: python tests.py"""
import numpy as np
from circuit import Circuit, parse
from sensor import FSRArray, contact_forces
from readout import ZeroPotential, scan_divider, ADC_MAX
from calib import Calibration
from diagnostics import Monitor

passed = 0


def test(name, cond):
    global passed
    assert cond, name
    passed += 1
    print(f"ok  {name}")


# circuit solver
c = Circuit("divider"); c.v("V1", "a", 0, 10); c.r("R1", "a", "b", 3e3); c.r("R2", "b", 0, 1e3)
v, i = c.solve()
test("voltage divider gives 2.5 V and -2.5 mA", abs(v["b"] - 2.5) < 1e-12 and abs(i["V1"] + 2.5e-3) < 1e-12)
c = Circuit("inverting amp"); c.v("V1", "in", 0, 1); c.r("R1", "in", "m", 1e3); c.r("R2", "m", "o", 1e4)
c.e("E1", "o", 0, 0, "m", 1e6)
test("ideal inverting amplifier gain -10", abs(c.solve()[0]["o"] + 10) < 1e-3)

s = FSRArray()
F = np.zeros((8, 8)); F[1, 1] = F[1, 5] = F[5, 1] = 10.0
G = s.static_G(F)
zp = ZeroPotential()
for row in (0, 5, 8):
    ref = zp.circuit(G, row)
    back = parse(ref.netlist())
    v1, v2 = ref.solve()[0], back.solve()[0]   # netlist keeps 9 significant digits
    fast = zp.solve(G)[row]
    test(f"row {row}: netlist round trip and fast solver match full MNA",
         max(abs(v1[k] - v2[k]) for k in v1) < 1e-6 and
         max(abs(fast[j] - v1[f"o{j}"]) for j in range(zp.C)) < 1e-9)

# ghosting: method A shows the classic ghost, zero potential does not
estA = scan_divider(G, False)
v = np.abs(zp.solve(G))
estC = 1e-3 * v[:8, :8] / v[:8, 8:9]
test("floating-line divider reads a ghost on (5,5)", estA[5, 5] > 100 * G[5, 5])
test("zero-potential readout has no ghost on (5,5)", estC[5, 5] < 2 * G[5, 5])

# sensor model
s = FSRArray(rows=1, cols=1, creep=0.0)
up = [s.step(np.array([[f]]), 0.01)[0, 0] for f in np.linspace(0, 10, 50)]
down = [s.step(np.array([[f]]), 0.01)[0, 0] for f in np.linspace(10, 0, 50)]
test("hysteresis: unloading reads higher than loading at 5 N", down[25] > up[24])
test("unpressed taxel is about 10 Mohm", abs(1 / s.step(np.zeros((1, 1)), 0.01)[0, 0] - 10e6) < 1)
test("contact footprint integrates to the applied force", abs(contact_forces(3.5, 3.5, 10.0).sum() - 10.0) < 0.05)

# calibration recovers the model when hysteresis and creep are off
s = FSRArray(rows=2, cols=2, creep=0.0); s.h[:] = 0.0
z = ZeroPotential(rows=2, cols=2)
cal = Calibration(s.shape).fit(s, z)
test("calibration recovers exponent within 0.01", np.abs(cal.n - s.n).max() < 0.01)
test("calibration recovers temperature coefficient within 5 %", np.abs(cal.alpha / s.alpha - 1).max() < 0.05)

# ADC and monitor
z = ZeroPotential()
codes = z.adc(np.full((9, 9), 5.0))
test("ADC clips at full scale", codes.max() == ADC_MAX)
z = ZeroPotential(); s = FSRArray()
start = z.frame(s.static_G(np.zeros((8, 8))))
mon = Monitor(z, start)
for _ in range(5):
    mon.update(z.frame(s.static_G(np.zeros((8, 8)))), np.zeros((8, 8)))
test("no flags on an idle healthy array", not mon.flags)
z.frozen = True
for _ in range(4):
    mon.update(z.frame(s.static_G(np.zeros((8, 8)))), np.zeros((8, 8)))
test("frozen ADC is flagged", ("frozen ADC",) in mon.flags)
print(f"\n{passed} tests passed")
