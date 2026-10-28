#!/usr/bin/env python
"""Full-protocol verification of a searched mixed-precision configuration.

Applies a QbD-VLA config (from result.json's path/frontier or an inline JSON
mapping) and runs the official evaluation (episodes 0..N-1, full horizon),
writing a normal results JSONL whose header records the config.

    python scripts/verify_config.py --model pi05 --suite libero_spatial \
        --config '{"ve":"W4A6","llm":"W4A6",...}' --episodes 20 --out results/...jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qbdvla.cli import build_adapter, resolve_checkpoint  # noqa: E402
from qbdvla.registry import get_runner_cls  # noqa: E402
from qbdvla.results import RunHeader, RunWriter  # noqa: E402
from qbdvla.search.space import MODEL_GROUPS, Config, SearchSpace  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-kwargs", nargs="*", metavar="K=V")
    ap.add_argument("--config", required=True, help="JSON mapping group->level")
    ap.add_argument("--episodes", type=int, default=20, help="evaluate init states [--episodes-from, --episodes) of every task")
    ap.add_argument("--episodes-from", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runner-kwargs", nargs="*", metavar="K=V", help="runner options, e.g. camera_size=256")
    ap.add_argument("--groups", choices=("fine", "coarse"), default="fine")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: all); one task per process is the X-VLA harness")
    ap.add_argument("--act-calib", nargs="*", metavar="COMP", default=None, help="verify on the calibrated lattice (same flags as `qbdvla run`)")
    ap.add_argument("--act-calib-alpha", type=float, default=0.5)
    ap.add_argument("--act-calib-quantile", type=float, default=0.999)
    ap.add_argument("--act-calib-from", type=int, default=20)
    ap.add_argument("--act-calib-episodes", type=int, default=2)
    ap.add_argument("--act-calib-tasks", type=int, default=1)
    args = ap.parse_args()

    levels = json.loads(args.config)
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    import yaml
    rk = {kv.split("=", 1)[0]: yaml.safe_load(kv.split("=", 1)[1]) for kv in (args.runner_kwargs or [])}
    runner = get_runner_cls(args.benchmark)(args.suite, episodes_per_task=args.episodes, seed=args.seed, **rk)
    adapter = build_adapter(args)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)

    from qbdvla.search.space import coarse_groups
    space = SearchSpace(adapter.component_map(), coarse_groups(args.model) if args.groups == "coarse" else MODEL_GROUPS[args.model])
    if args.act_calib:
        from qbdvla.cli import _collect_act_calib
        space.act_calib = _collect_act_calib(adapter, runner, args)
    known = {g.name for g in space.groups}
    cfg = Config({g.name: levels.get(g.name, "BASELINE") for g in space.groups})
    unknown = set(levels) - known
    if unknown:
        raise SystemExit(f"unknown groups in --config: {unknown}; known: {sorted(known)}")
    applied = space.apply(cfg)
    budget = space.budget(cfg)

    header = RunHeader(
        model=args.model, checkpoint=adapter.checkpoint, benchmark=args.benchmark,
        suite=args.suite, preset="SEARCHED", scope="mixed", method="qbd",
        baseline_dtype=str(adapter.dtype).replace("torch.", ""),
        episodes_per_task=args.episodes, seed=args.seed,
        quantize_lm_head=False, include_conv=False,
        quant_report={g: {"layers": n} for g, n in applied.items()},
        extra={"qbd_config": cfg.levels, "budget": budget, "runner_options": rk,
               "act_calib": (getattr(args, "_act_calib_provenance", None) if args.act_calib else None)},
    )
    with RunWriter(Path(args.out), header, resume=True) as writer:
        kw = {}
        if args.episodes_from:
            kw["episodes"] = range(args.episodes_from, args.episodes)
        if args.tasks:
            ids = [int(x) for x in args.tasks.split(",") if x.strip()]
            kw["tasks"] = [runner.tasks()[i] for i in ids]
        summary = runner.run(adapter, writer, **kw)
    if hasattr(runner, "close"):
        runner.close()
    summary["budget"] = budget
    summary["config"] = cfg.levels
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
