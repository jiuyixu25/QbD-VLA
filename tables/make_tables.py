#!/usr/bin/env python3
"""Regenerate the record-derived tables of the paper from records/ (Python 3.8+, standard library only).

Usage
    python tables/make_tables.py [--records <path to records/>] [--out <dir>]

The records directory is found automatically (records/ of this repository). Outputs go to tables/generated/ by default:

    tab_libero.tex      Table 1  success on the four LIBERO suites (full precision, uniform W4A4/W4A6, QbD-VLA)
    tab_released.tex    Table 2  released VLA quantization methods on pi0.5 under the same evaluation
    tab_allocation.tex  Table 3  QbD-VLA against sensitivity-based allocations and the two-stage allocator (UFR)
    tab_simpler.tex     Table 4  X-VLA on the SIMPLER visual-matching split
    tab_groups.tex      Table 7  layer groups, parameter shares and the selected level of every group
    interaction.txt     the group-interaction numbers quoted in the text (Sections 1 and 4.3)
    cells.csv           episodes, successes and success rate of every evaluation cell in the records

Tables 5 and 6 (real-robot trials) and Table 8 (per-model settings) are not derived from these records.
"""
import csv, json, os, sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent


def find_records():
    cands = []
    if "--records" in sys.argv:
        cands.append(Path(sys.argv[sys.argv.index("--records") + 1]))
    if os.environ.get("QBD_RECORDS"):
        cands.append(Path(os.environ["QBD_RECORDS"]))
    cands += [PKG / "records", PKG.parent / "QbD-VLA_data" / "records", PKG.parent / "records",
              Path.cwd() / "records", Path.cwd() / "QbD-VLA_data" / "records"]
    for c in cands:
        if (c / "results").is_dir() and (c / "search").is_dir():
            return c.resolve()
    sys.exit("records/ not found; pass --records <path>.")


ROOT = find_records()
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(__file__).resolve().parent / "generated"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [("pi05", r"$\pi_{0.5}$"), ("openvla_oft", "OpenVLA-OFT"), ("xvla", "X-VLA"), ("pi0", r"$\pi_0$")]
SUITES = [("libero_spatial", "Spatial"), ("libero_object", "Object"), ("libero_goal", "Goal"), ("libero_10", "Long")]
# verified cell of the configuration each allocation run selects on LIBERO-Spatial, and the same configuration on the other suites
HEADLINE = {"pi05": "qbd-deep-final", "openvla_oft": "qbd-deep-final", "xvla": "qbd-final", "pi0": "qbd-select-final"}
TRANSFER = {"pi05": "qbd-transfer-deep-final", "openvla_oft": "qbd-transfer-final", "xvla": "qbd-transfer-final", "pi0": "qbd-transfer-final"}
SEARCH = {"pi05": "pi05-spatial-deep", "openvla_oft": "oft-spatial", "xvla": "xvla-spatial", "pi0": "pi0-spatial"}
RELEASED = {"pi05": "BF16", "openvla_oft": "BF16", "xvla": "FP32", "pi0": "BF16"}
GROUP_NAMES = {
    "pi05": [("ve", "Vision encoder"), ("llm", "Language backbone"), ("ah_attn", "Action head: attention projections"),
             ("ah_adarms_in", "Action head: pre-attention modulators"), ("ah_adarms_post", "Action head: post-attention and final modulators"),
             ("ah_mlp", "Action head: MLP layers"), ("ah_plumbing", "Action head: action and time projections")],
    "openvla_oft": [("ve", "Vision encoder"), ("mp", "Vision-language projector"), ("llm", "Language backbone"),
                    ("ah_fc1", "Action head: input projection fc1"), ("ah_blocks", "Action head: residual-block layers"),
                    ("ah_fc2", "Action head: output projection fc2")],
    "xvla": [("ve", "Vision encoder"), ("llm", "Language backbone"), ("ah_qkv", "Action head: attention QKV projections"),
             ("ah_proj", "Action head: attention output projections"), ("ah_fc1", "Action head: MLP fc1 layers"),
             ("ah_fc2", "Action head: MLP fc2 layers"), ("ah_rest", "Action head: remaining quantizable layers")],
    "pi0": [("ve", "Vision encoder"), ("llm", "Language backbone"), ("ah_attn", "Action head: attention projections"),
            ("ah_mlp", "Action head: MLP layers"), ("ah_plumbing", "Action head: action and time projections")],
}
WARN = []


def read_cell(path):
    """(header, episodes) of one evaluation cell; None if the file is absent."""
    path = Path(path)
    if not path.exists():
        return None
    header, eps = None, []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "header":
                header = r
            elif r.get("kind") == "episode":
                eps.append(r)
    return header, eps


def rate(eps):
    return 100.0 * sum(1 for e in eps if e.get("success")) / len(eps)


def cell(suite, model, tag, benchmark="libero", expect=200):
    c = read_cell(ROOT / "results" / benchmark / suite / model / f"{tag}.jsonl")
    if c is None:
        WARN.append(f"missing cell {benchmark}/{suite}/{model}/{tag}")
        return None
    h, eps = c
    if expect and len(eps) != expect:
        WARN.append(f"{benchmark}/{suite}/{model}/{tag}: {len(eps)} episodes (expected {expect})")
    return rate(eps) if eps else None


def budget(suite, model, tag, benchmark="libero"):
    c = read_cell(ROOT / "results" / benchmark / suite / model / f"{tag}.jsonl")
    if c is None or c[0] is None:
        return None
    b = (c[0].get("extra") or {}).get("budget") or {}
    return (b.get("w_bits_eff"), b.get("a_bits_eff")) if b else None


def levels(suite, model, tag, benchmark="libero"):
    c = read_cell(ROOT / "results" / benchmark / suite / model / f"{tag}.jsonl")
    return (c[0].get("extra") or {}).get("qbd_config") if c and c[0] else None


def f1(x):
    return "--" if x is None else f"{x:.1f}"


def avg(vals):
    return None if any(v is None for v in vals) else sum(vals) / len(vals)


def bits_name(wa):
    w, a = wa
    ws = "4" if abs(w - 4) < 0.005 else f"{w:.2f}"
    return f"W{ws}A{a:.2f}"


def table_libero():
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}clrrrrr@{}}", r"\toprule",
         r"Policy & Configuration & Spatial & Object & Goal & Long & Avg. \\", r"\midrule"]
    for i, (m, lab) in enumerate(MODELS):
        rows = [("Full precision", lambda s: cell(s, m, "rtn-BASELINE-e2e")),
                ("Uniform W4A4", lambda s: cell(s, m, "rtn-W4A4-e2e")),
                ("Uniform W4A6", lambda s: cell(s, m, "rtn-W4A6-e2e")),
                ("QbD-VLA " + bits_name(budget("libero_spatial", m, HEADLINE[m])),
                 lambda s: cell(s, m, HEADLINE[m] if s == "libero_spatial" else TRANSFER[m]))]
        for j, (name, fn) in enumerate(rows):
            vals = [fn(s) for s, _ in SUITES]
            lead = r"\multirow{4}{*}{%s}" % lab if j == 0 else ""
            L.append(f"{lead} & {name} & " + " & ".join(f1(v) for v in vals) + f" & {f1(avg(vals))} \\\\")
        L.append(r"\midrule" if i < len(MODELS) - 1 else r"\bottomrule")
    L += [r"\end{tabular}",
          r"\caption{Success rate (\%) on the four LIBERO suites. QbD-VLA allocations are selected on LIBERO-Spatial and applied unchanged "
          r"to the other suites. Their bit widths are parameter-weighted nominal values over all layer groups.}",
          r"\label{tab:libero}", r"\end{table*}", ""]
    return "\n".join(L)


def table_released():
    m = "pi05"
    rows = [("FP", "16 / 16", lambda s: cell(s, m, "rtn-BASELINE-e2e")),
            ("ActQuant", "3.09 / 16", lambda s: cell(s, m, "actquant-3bpw")),
            ("HoloQ-VLA", "4 / 4", lambda s: cell(s, m, "omegaqvla-w4a4")),
            ("QbD-VLA", "%.2f / %.2f" % budget("libero_spatial", m, HEADLINE[m]),
             lambda s: cell(s, m, HEADLINE[m] if s == "libero_spatial" else TRANSFER[m]))]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcrrrrr@{}}", r"\toprule",
         r"Configuration & W / A & Spatial & Object & Goal & Long & Avg. \\", r"\midrule"]
    for name, wa, fn in rows:
        vals = [fn(s) for s, _ in SUITES]
        L.append(f"{name} & {wa} & " + " & ".join(f1(v) for v in vals) + f" & {f1(avg(vals))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{VLA quantization methods on $\pi_{0.5}$ under our evaluation on the four LIBERO suites, using their released code or weights. "
          r"The ActQuant checkpoint averages 3.09 weight bits. QbD-VLA applies the allocation selected on Spatial to every suite.}",
          r"\label{tab:released}", r"\end{table}", ""]
    # full-precision references of the released runtimes (quoted in the text)
    refs = []
    for name, tag in (("HoloQ-VLA runtime, FP16", "omegaqvla-fp16"), ("ActQuant runtime, FP16 export", "actquant-fp16")):
        vals = []
        for s, lab in SUITES:
            c = read_cell(ROOT / "results/libero" / s / m / f"{tag}.jsonl")
            vals.append(f"{lab} {rate(c[1]):.1f} (n={len(c[1])})" if c and c[1] else f"{lab} --")
        refs.append(f"% {name}: " + ", ".join(vals))
    return "\n".join(L + refs + [""])


def table_allocation():
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Allocation method & $\pi_{0.5}$ & OpenVLA-OFT & X-VLA & $\pi_0$ \\", r"\midrule"]
    ufr_tag, ufr_cost = {}, []
    for m, _ in MODELS:
        ufr_tag[m] = "ufr2-final" if (ROOT / f"results/libero/libero_spatial/{m}/ufr2-final.jsonl").exists() else "ufr-final"
        ufr_cost.append("W%.2f / A%.2f" % budget("libero_spatial", m, ufr_tag[m]))

    def proxy(m, p):
        for pre in ("e3sel", "e3"):   # allocation at the budget of the selected point; the endpoint-budget run when they coincide
            if (ROOT / f"results/libero/libero_spatial/{m}/{pre}-{p}.jsonl").exists():
                return cell("libero_spatial", m, f"{pre}-{p}")
        WARN.append(f"missing proxy cell {m} {p}")
        return None

    rows = [("QbD-VLA", lambda m: cell("libero_spatial", m, HEADLINE[m])),
            ("Action error", lambda m: proxy(m, "action_error")),
            ("Activation kurtosis", lambda m: proxy(m, "act_stats")),
            ("Reconstruction error", lambda m: proxy(m, "recon_error")),
            ("UFR", lambda m: cell("libero_spatial", m, ufr_tag[m]))]
    for name, fn in rows:
        if name == "UFR":
            L.append(r"\midrule")
        L.append(f"{name} & " + " & ".join(f1(fn(m)) for m, _ in MODELS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Success rate (\%) on LIBERO-Spatial. The three sensitivity-based rows share a greedy allocator and the target cost of the "
          r"QbD-VLA allocation of each policy. UFR reaches its own cost: " + ", ".join(ufr_cost[:-1]) + ", and " + ufr_cost[-1] + ".}",
          r"\label{tab:allocation}", r"\end{table}", ""]
    return "\n".join(L)


def table_simpler():
    fam = [("pick_coke_can", "Coke can"), ("move_near", "Move near"), ("drawer", "Drawer"), ("place_in_drawer", "Put in drawer")]
    rows = [("FP", "16 / 16", "rtn-BASELINE-e2e"), ("Uniform W4A8", "4 / 8", "rtn-W4A8-e2e"), ("Uniform W4A6", "4 / 6", "rtn-W4A6-e2e"),
            ("Uniform W4A4", "4 / 4", "rtn-W4A4-e2e"), (r"QbD-VLA, 24 tasks, $\tau = 0.90$", None, "qbd-final"),
            (r"QbD-VLA, 40 tasks, $\tau = 0.95$", None, "qbd-t95-final")]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcrrrrr@{}}", r"\toprule",
         r"X-VLA configuration & W / A & " + " & ".join(l for _, l in fam) + r" & Overall \\", r"\midrule"]
    for name, wa, tag in rows:
        c = read_cell(ROOT / "results/simpler/google_robot_vm/xvla" / f"{tag}.jsonl")
        if c is None:
            WARN.append(f"missing SIMPLER cell {tag}")
            continue
        h, eps = c
        if len(eps) != 864:
            WARN.append(f"SIMPLER {tag}: {len(eps)} episodes (expected 864)")
        if wa is None:
            b = (h.get("extra") or {}).get("budget") or {}
            wa = "%.2f / %.2f" % (b.get("w_bits_eff", float("nan")), b.get("a_bits_eff", float("nan")))
        vals = []
        for key, _ in fam:
            sub = [e for e in eps if e.get("task_name", "").split("/")[0] == key]
            vals.append(rate(sub) if sub else None)
        L.append(f"{name} & {wa} & " + " & ".join(f1(v) for v in vals) + f" & {rate(eps):.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{X-VLA on the SIMPLER visual-matching split.}", r"\label{tab:simpler}", r"\end{table}", ""]
    return "\n".join(L)


def table_groups():
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
         r"Model & Group & Layers & Params (M) & Share (\%) & QbD-VLA \\", r"\midrule"]
    for i, (m, lab) in enumerate(MODELS):
        groups = json.load(open(ROOT / "search" / SEARCH[m] / "result.json"))["groups"]
        lv = levels("libero_spatial", m, HEADLINE[m])
        total = sum(g["params"] for g in groups.values())
        names = GROUP_NAMES[m]
        for j, (key, gname) in enumerate(names):
            g = groups[key]
            share = 100.0 * g["params"] / total
            share_s = "<0.001" if share < 0.001 else f"{share:.1f}"
            params_s = f"{g['params'] / 1e6:.3f}" if g["params"] < 1e6 else f"{g['params'] / 1e6:.1f}"
            level = lv[key] if lv and key in lv else "BASELINE"
            level = RELEASED[m] if level == "BASELINE" else level
            lead = r"\multirow{%d}{*}{%s}" % (len(names) + 1, lab) if j == 0 else ""
            L.append(f"{lead} & {gname} & {g['layers']} & {params_s} & {share_s} & {level} \\\\")
        b = budget("libero_spatial", m, HEADLINE[m])
        L.append(r" & \emph{All groups} & %d & %.1f & 100.0 & W%.2fA%.2f \\" % (sum(g["layers"] for g in groups.values()), total / 1e6, b[0], b[1]))
        L.append(r"\midrule" if i < len(MODELS) - 1 else r"\bottomrule")
    L += [r"\end{tabular}",
          r"\caption{Layer groups used for bit allocation, with the number of quantizable linear layers in each group, their parameter count, "
          r"their share of the model's quantizable parameters, and the level QbD-VLA selects for the group on LIBERO-Spatial. BF16 and FP32 mark "
          r"groups kept at the released precision. The \emph{All groups} rows give the parameter-weighted nominal bit widths used in Table~\ref{tab:libero}.}",
          r"\label{tab:groups}", r"\end{table*}", ""]
    return "\n".join(L)


def interaction():
    """Group-interaction numbers quoted in the text, all on pi0.5 / LIBERO-Spatial (200 episodes per cell)."""
    m = "pi05"
    fp = cell("libero_spatial", m, "rtn-BASELINE-e2e")
    rows = [("full precision", "rtn-BASELINE-e2e"),
            ("action-head attention projections at W4A4, everything else at full precision", "ablate-W4A4-ah-onlyself_attn"),
            ("action-head modulators at W4A4, everything else at full precision", "ablate-W4A4-ah-onlyadarms"),
            ("both groups at W4A4, everything else at full precision", "ablate-W4A4-ah-attn_adarms"),
            ("uniform W4A6 (MLP at A6, attention / post-attention modulators / I-O projections at A6)", "rtn-W4A6-e2e"),
            ("MLP at A4, those three groups at A6", "ufr-relax-mlp-only"),
            ("MLP at A6, those three groups at A4", "ufr-final"),
            ("MLP at A4, those three groups at A4 (selected allocation)", HEADLINE[m])]
    L = [f"Group interactions on pi0.5, LIBERO-Spatial, 200 episodes per cell (success %, and loss relative to full precision {fp:.1f})", ""]
    for desc, tag in rows:
        r = cell("libero_spatial", m, tag)
        L.append(f"{f1(r):>6}  (loss {f1(fp - r) if r is not None else '--':>5})  {desc}   [{tag}]")
    return "\n".join(L) + "\n"


def cells_csv():
    rows = []
    for p in sorted((ROOT / "results").rglob("*.jsonl")):
        c = read_cell(p)
        if c is None or not c[1]:
            continue
        rel = p.relative_to(ROOT / "results").parts
        b = ((c[0] or {}).get("extra") or {}).get("budget") or {}
        rows.append([rel[0], rel[1], rel[2], p.stem, len(c[1]), sum(1 for e in c[1] if e.get("success")), f"{rate(c[1]):.1f}",
                     f"{b['w_bits_eff']:.2f}" if b else "", f"{b['a_bits_eff']:.2f}" if b else ""])
    with open(OUT / "cells.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["benchmark", "suite", "model", "tag", "episodes", "successes", "success_rate", "w_bits", "a_bits"]); w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print(f"records: {ROOT}\noutput:  {OUT}\n")
    for name, fn in [("tab_libero", table_libero), ("tab_released", table_released), ("tab_allocation", table_allocation),
                     ("tab_simpler", table_simpler), ("tab_groups", table_groups)]:
        s = fn(); (OUT / f"{name}.tex").write_text(s); print(s)
    s = interaction(); (OUT / "interaction.txt").write_text(s); print(s)
    print(f"cells.csv: {cells_csv()} cells")
    if WARN:
        print("\nWARNINGS"); print("\n".join(WARN))
