"""A deliberately simple allocator that uses QbD-VLA's probe but none of its search structure.

"Uniform floor, then relax": (A) descend the uniform levels W8A8 -> W4A8 -> W4A6 -> W4A4 with the same
racing test and the same floor tau*s0 as QbD-VLA and keep the last uniform level that passes, U;
(B) from uniform U, lower each group one level in isolation and record which single relaxations pass;
(C) apply every passing relaxation jointly and race that configuration once; if it fails, fall back
to U plus the single passing relaxation that saves the most bits. The output is verified under the
full protocol by the caller. The question it answers: does the joint, retrying descent of QbD-VLA find
anything that this two-stage procedure with an identical probe signal cannot?

    python scripts/uniform_floor_relax.py --model pi05 --suite libero_spatial --tau 0.75 --out search/ufr-pi05-spatial/ufr.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbdvla.cli import build_adapter, build_parser, resolve_checkpoint  # noqa: E402
from qbdvla.registry import get_runner_cls  # noqa: E402
from qbdvla.search.greedy import ProbeCache, WeightVault, race  # noqa: E402
from qbdvla.search.probe import make_run_block  # noqa: E402
from qbdvla.search.space import MODEL_GROUPS, Config, SearchSpace  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, required=True, help="same relative floor as the QbD-VLA search of this policy")
    ap.add_argument("--blocks", type=int, default=3, help="max probe blocks per race (QbD-VLA: 3)")
    ap.add_argument("--probe-from", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--remote-sim", action="store_true", help="X-VLA: probe through per-task client processes")
    ap.add_argument("--client-python", default=None)
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--anchor-from", default=None,
                    help="search directory whose probes.jsonl supplies the full-precision anchor, so the floor is the "
                         "one the compared QbD-VLA search used (frozen) instead of a fresh draw")
    ap.add_argument("--scan-all", action="store_true",
                    help="race every uniform level instead of stopping at the first failure; U = the cheapest that passes")
    ap.add_argument("--relax-from-baseline", action="store_true",
                    help="run the relaxation stages from full precision when no uniform level passes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    args = build_parser().parse_args(["run", "--model", a.model, "--benchmark", a.benchmark, "--suite", a.suite,
                                      "--preset", "BASELINE", "--scope", "e2e", "--seed", str(a.seed), "--out", "/dev/null"]
                                     + (["--checkpoint", a.checkpoint] if a.checkpoint else []))
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner = get_runner_cls(args.benchmark)(args.suite, episodes_per_task=a.probe_from + a.blocks,
                                            seed=a.seed, max_steps_override=a.horizon)
    adapter = build_adapter(args)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)
        runner.max_steps_override = a.horizon
    space = SearchSpace(adapter.component_map(), MODEL_GROUPS[a.model])
    vault = WeightVault(space)
    n_tasks = len(runner.tasks())
    if a.remote_sim:
        import os
        from qbdvla.search.remote_probe import make_remote_run_block, start_server_thread
        start_server_thread(adapter, a.port)
        raw_block = make_remote_run_block(space, vault, port=a.port, client_python=a.client_python or sys.executable,
                                          benchmark=a.benchmark, suite=a.suite, n_tasks=n_tasks, horizon=a.horizon,
                                          episodes_from=a.probe_from, env={"MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    else:
        raw_block = make_run_block(adapter, runner, space, vault, episodes_from=a.probe_from)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out.with_suffix(".cache.jsonl")
    if a.anchor_from:
        # append the compared search's records of the full-precision configuration; the cache keeps the last record
        # per key, so the frozen anchor replaces any anchor this script drew itself
        key = space.initial().key(); n_added = 0
        with open(cache_path, "a") as f:
            for line in open(Path(a.anchor_from) / "probes.jsonl"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("key") == key:
                    f.write(json.dumps(rec) + "\n"); n_added += 1
        print(f"anchor frozen from {a.anchor_from}: {n_added} records appended", flush=True)
    cache = ProbeCache(cache_path)
    spent = {"eps": 0}

    def run_block(cfg, i):
        spent["eps"] += n_tasks
        return raw_block(cfg, i)

    def R(cfg, floor):
        verdict, rec = race(cfg, floor, run_block, cache, block_size=n_tasks, max_blocks=a.blocks)
        print(f"  race {cfg.pretty()} -> {verdict} ({rec['successes']}/{rec['n']})", flush=True)
        return verdict, rec

    t0 = time.time()
    log: dict = {"model": a.model, "suite": a.suite, "checkpoint": args.checkpoint, "tau": a.tau,
                 "probe": {"blocks": a.blocks, "from": a.probe_from, "horizon": a.horizon, "n_tasks": n_tasks}}
    _, base = R(space.initial(), None)
    s0 = base["score"]; floor = a.tau * s0
    log["anchor"] = base; log["floor"] = floor
    print(f"anchor {s0:.3f} (n={base['n']}) -> floor {floor:.3f}", flush=True)

    def uniform(level: str) -> Config:
        return Config({g.name: (level if level in g.levels else g.levels[-1]) for g in space.groups})

    # (A) uniform floor
    levels = list(space.groups[0].levels)          # BASELINE first, descending
    U = "BASELINE"; log["stage_a"] = []
    for L in levels[1:]:
        verdict, rec = R(uniform(L), floor)
        log["stage_a"].append({"level": L, "verdict": verdict, **rec, "budget": space.budget(uniform(L))})
        if verdict == "safe":
            U = L                      # levels descend in cost, so the last safe one is the cheapest that passes
        elif not a.scan_all:
            break
    log["uniform_floor"] = U; log["scan_all"] = a.scan_all; log["relax_from_baseline"] = a.relax_from_baseline
    log["anchor_from"] = a.anchor_from
    print(f"uniform floor: {U}", flush=True)

    chosen = uniform(U); log["stage_b"] = []; safe_relax = []
    if U != "BASELINE" or a.relax_from_baseline:
        # (B) single relaxations from uniform U, each judged in isolation
        base_budget = space.budget(chosen)
        for g in space.groups:
            cur = chosen.levels[g.name]; idx = g.levels.index(cur)
            if idx + 1 >= len(g.levels):
                continue
            cand = Config(dict(chosen.levels)); cand.levels[g.name] = g.levels[idx + 1]
            verdict, rec = R(cand, floor)
            b = space.budget(cand)
            saved = sum(base_budget.values()) - sum(b.values())
            log["stage_b"].append({"group": g.name, "to": g.levels[idx + 1], "verdict": verdict, **rec, "bits_saved": saved})
            if verdict == "safe":
                safe_relax.append((saved, g.name, g.levels[idx + 1]))
        # (C) all passing relaxations jointly, once
        if len(safe_relax) >= 2:
            joint = Config(dict(chosen.levels))
            for _, gname, lv in safe_relax:
                joint.levels[gname] = lv
            verdict, rec = R(joint, floor)
            log["stage_c"] = {"joint_levels": joint.levels, "verdict": verdict, **rec, "budget": space.budget(joint)}
            if verdict == "safe":
                chosen = joint
            else:
                saved, gname, lv = max(safe_relax)
                chosen = Config(dict(chosen.levels)); chosen.levels[gname] = lv
                log["stage_c"]["fallback"] = {"group": gname, "to": lv}
        elif len(safe_relax) == 1:
            _, gname, lv = safe_relax[0]
            chosen = Config(dict(chosen.levels)); chosen.levels[gname] = lv
            log["stage_c"] = {"single": {"group": gname, "to": lv}}
        else:
            log["stage_c"] = {"none_safe": True}

    log.update({"levels": chosen.levels, "budget": space.budget(chosen), "probe_episodes": spent["eps"],
                "wall_s": round(time.time() - t0, 1)})
    out.write_text(json.dumps(log, indent=1))
    print(json.dumps({"levels": chosen.levels, "budget": log["budget"], "probe_episodes": spent["eps"],
                      "wall_s": log["wall_s"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
