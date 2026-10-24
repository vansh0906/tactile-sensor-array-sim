"""Small DC circuit solver (modified nodal analysis) that reads and writes SPICE netlists.

Supports the three element types the readout circuits need:
    Rname n1 n2 value          resistor (ohms)
    Vname n+ n- value          DC voltage source (volts)
    Ename n+ n- nc+ nc- gain   voltage-controlled voltage source (ideal op-amp model)
Node 0 is ground. The same netlist text opens in LTspice, so the two solvers can be compared.
"""
import numpy as np


class Circuit:
    def __init__(self, title="circuit"):
        self.title = title
        self.nodes = {"0": -1}
        self.R, self.V, self.E = [], [], []

    def _n(self, name):
        name = str(name)
        if name not in self.nodes:
            self.nodes[name] = len(self.nodes) - 1
        return self.nodes[name]

    def r(self, name, a, b, ohms):
        self._n(a); self._n(b); self.R.append((name, str(a), str(b), float(ohms)))

    def v(self, name, p, n, volts):
        self._n(p); self._n(n); self.V.append((name, str(p), str(n), float(volts)))

    def e(self, name, p, n, cp, cn, gain):
        for x in (p, n, cp, cn):
            self._n(x)
        self.E.append((name, str(p), str(n), str(cp), str(cn), float(gain)))

    @staticmethod
    def _num(x):
        return f"{x:.9g}".replace("e+", "e")      # 1e+07 -> 1e07, which every SPICE accepts

    def netlist(self, comments=()):
        out = [f"* {self.title}"] + [f"* {c}" for c in comments]
        f = self._num
        out += [f"{n} {a} {b} {f(v)}" for n, a, b, v in self.R]
        out += [f"{n} {a} {b} {f(v)}" for n, a, b, v in self.V]
        out += [f"{n} {p} {m} {cp} {cn} {f(g)}" for n, p, m, cp, cn, g in self.E]
        out += [".op", ".end", ""]
        return "\n".join(out)

    def solve(self):
        """Return {node: voltage} and {source: current} from one MNA solve."""
        nn = len(self.nodes) - 1
        m = len(self.V) + len(self.E)
        A = np.zeros((nn + m, nn + m))
        z = np.zeros(nn + m)
        idx = self.nodes

        def add(i, j, val):
            if i >= 0 and j >= 0:
                A[i, j] += val

        for _, a, b, ohms in self.R:
            g = 1.0 / ohms
            ia, ib = idx[a], idx[b]
            add(ia, ia, g); add(ib, ib, g); add(ia, ib, -g); add(ib, ia, -g)
        k = nn
        for _, p, n, volts in self.V:
            ip, im = idx[p], idx[n]
            add(ip, k, 1); add(im, k, -1); add(k, ip, 1); add(k, im, -1)
            z[k] = volts
            k += 1
        for _, p, n, cp, cn, gain in self.E:
            ip, im, icp, icn = idx[p], idx[n], idx[cp], idx[cn]
            add(ip, k, 1); add(im, k, -1); add(k, ip, 1); add(k, im, -1)
            add(k, icp, -gain); add(k, icn, gain)
            k += 1
        x = np.linalg.solve(A, z)
        volts = {name: (0.0 if i < 0 else x[i]) for name, i in idx.items()}
        names = [s[0] for s in self.V] + [s[0] for s in self.E]
        currents = dict(zip(names, x[nn:]))
        return volts, currents


def parse(text):
    """Read a netlist written by Circuit.netlist (or a hand-written one with R, V, E lines)."""
    c = Circuit()
    for line in text.splitlines():
        t = line.split()
        if not t or t[0].startswith(("*", ".")):
            if t and t[0].startswith("*") and c.title == "circuit":
                c.title = line[1:].strip()
            continue
        kind = t[0][0].upper()
        if kind == "R":
            c.r(t[0], t[1], t[2], float(t[3]))
        elif kind == "V":
            c.v(t[0], t[1], t[2], float(t[-1]))
        elif kind == "E":
            c.e(t[0], t[1], t[2], t[3], t[4], float(t[5]))
        else:
            raise ValueError(f"unsupported element: {line}")
    return c
