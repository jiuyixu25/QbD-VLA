"""Probe decision vs full-protocol success along a search's accepted path.

For every accepted point i of a search (search/<name>/result.json), reads the 100-episode verification
cell results/libero/<suite>/<model>/path-<tag>-NN.jsonl (initial states 0-9), and reports the probe score
that accepted the move next to the full-protocol success on the same states, the Wilson interval, whether
the verified success falls below tau * (full-precision success on the same 100 episodes), and the
Spearman correlation between probe and verification along the path. Optionally writes a figure.

    python scripts/path_verification.py --search search/pi05-spatial-deep --tag deep --tau 0.75 --fig figs/path_pi05.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def episodes(path: Path, max_state: int | None = None):
    out = []
    for l in path.read_text().splitlines():
        if '"kind": "episode"' not in l:
            continue
        e = json.loads(l)
        if max_state is None or int(e["episode"]) < max_state:
            out.append(e)
    return out


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    return float("nan") if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True)
    ap.add_argument("--tag", required=True, help="cell tag infix: path-<tag>-NN.jsonl")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--fp-cell", default="rtn-BASELINE-e2e")
    ap.add_argument("--states", type=int, default=10, help="verification used initial states [0, states)")
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    r = json.load(open(Path(a.search) / "result.json")); model, suite = r["model"], r["suite"]
    cdir = Path("results/libero") / suite / model
    fp = episodes(cdir / f"{a.fp_cell}.jsonl", a.states)
    fp_sr = sum(bool(e["success"]) for e in fp) / len(fp); floor = a.tau * fp_sr
    print(f"{model}/{suite}: FP on states 0-{a.states-1}: {100*fp_sr:.1f}% (n={len(fp)}) -> full-protocol floor tau*FP = {100*floor:.1f}%")
    rows = []
    for i, p in enumerate(r["path"]):
        if i == 0:
            continue
        cell = cdir / f"path-{a.tag}-{i:02d}.jsonl"
        if not cell.exists():
            continue
        eps = episodes(cell, a.states); k = sum(bool(e["success"]) for e in eps); n = len(eps)
        lo, hi = wilson(k, n)
        rows.append({"i": i, "probe": p["score"], "probe_n": p["n"], "full": k / n, "n": n, "lo": lo, "hi": hi,
                     "below_floor": (k / n) < floor, "W": p["w_bits_eff"], "A": p["a_bits_eff"]})
    print(f"{'i':>3} {'W':>5} {'A':>5} {'probe':>9} {'full SR':>8} {'95% CI':>14}  below-floor")
    for q in rows:
        print(f"{q['i']:3d} {q['W']:5.2f} {q['A']:5.2f} {q['probe']:6.2f}/{q['probe_n']:<2d} {100*q['full']:7.1f}% [{100*q['lo']:5.1f},{100*q['hi']:5.1f}]  {'YES' if q['below_floor'] else '-'}")
    if len(rows) >= 3:
        rho = spearman([q["probe"] for q in rows], [q["full"] for q in rows])
        print(f"points verified: {len(rows)}/{len(r['path'])-1}; accepted-but-below-floor: {sum(q['below_floor'] for q in rows)}; "
              f"Spearman(probe, full) along the path = {rho:.2f}; min full SR on path = {100*min(q['full'] for q in rows):.1f}%")
    if a.fig and rows:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.3, 2.6))
        ax.errorbar([q["probe"] for q in rows], [q["full"] for q in rows],
                    yerr=[[q["full"] - q["lo"] for q in rows], [q["hi"] - q["full"] for q in rows]], fmt="o", ms=3, lw=0.8, capsize=2, color="#2b4c7e")
        for q in rows:
            ax.annotate(str(q["i"]), (q["probe"], q["full"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.axhline(floor, ls="--", lw=0.8, color="#b23a48"); ax.axvline(a.tau * r["path"][0]["score"], ls=":", lw=0.8, color="#b23a48")
        ax.set_xlabel("probe score that accepted the move"); ax.set_ylabel(f"full-protocol success ({a.states*10} episodes)")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); fig.tight_layout(); fig.savefig(a.fig); print("wrote", a.fig)


if __name__ == "__main__":
    main()
