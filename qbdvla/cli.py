"""``qbdvla`` command line.

    qbdvla search    --model pi05 --benchmark libero --suite libero_spatial --tau 0.75 --probe-horizon 120 --out search/pi05-spatial
    qbdvla run       --model pi05 --benchmark libero --suite libero_spatial --preset W4A6 --scope e2e --episodes 20
    qbdvla baselines --model pi05 --benchmark libero --suite libero_spatial --target-sum 9.74 --out search/pi05-spatial-baselines
    qbdvla inspect   --model pi05 [--list-layers]
    qbdvla serve     --model xvla --benchmark libero --preset W4A6 --port 8010
    qbdvla summarize results/ [--format md|latex|csv|json] [--ci]
    qbdvla presets
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .components import quantize_scope, resolve_scope
from .quant.presets import PRESETS, parse_spec
from .registry import BENCHMARKS, MODELS, get_adapter_cls, get_runner_cls
from .results import RunHeader, RunWriter

log = logging.getLogger("qbdvla")
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_model_config(model: str) -> dict[str, Any]:
    path = CONFIG_DIR / "models" / f"{model}.yaml"
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def resolve_checkpoint(model: str, checkpoint: str | None, benchmark: str, suite: str) -> str:
    if checkpoint and checkpoint != "auto":
        return checkpoint
    if model == "remote":
        return "remote"  # the server knows its checkpoint; recorded from /info
    cfg = load_model_config(model)
    ckpts = cfg.get("checkpoints", {})
    bench = ckpts.get(benchmark)
    if isinstance(bench, dict):
        if suite in bench:
            return bench[suite]
        if "default" in bench:
            return bench["default"]
    elif isinstance(bench, str):
        return bench
    raise SystemExit(
        f"no checkpoint configured for model={model} benchmark={benchmark} suite={suite}; "
        f"pass --checkpoint or edit configs/models/{model}.yaml"
    )


def default_out_path(model: str, benchmark: str, suite: str, method: str, preset: str, scope: str,
                     lm_head: bool, seed: int = 0) -> Path:
    tag = f"{method}-{preset}-{scope}" + ("-lmhead" if lm_head else "") + (f"-seed{seed}" if seed else "")
    return REPO_ROOT / "results" / benchmark / suite / model / f"{tag}.jsonl"


def build_adapter(args) -> Any:
    cls = get_adapter_cls(args.model)
    cfg = load_model_config(args.model)
    extra = dict(cfg.get("adapter_kwargs", {}))
    for kv in args.model_kwargs or []:
        k, _, v = kv.partition("=")
        extra[k] = yaml.safe_load(v)
    dtype = args.dtype or cfg.get("dtype")
    adapter = cls(args.checkpoint, device=args.device, dtype=dtype, seed=args.seed, **extra)
    adapter.load()
    return adapter



def _collect_act_calib(adapter, runner, args) -> dict:
    """Run calibration episodes with the un-quantized policy and build per-layer
    smoothing/clip vectors for the components named in ``--act-calib``.

    Statistics are cached per (model, checkpoint, suite, components, episodes);
    the six cells of the calibration experiment share one collection run.
    """
    import torch

    from .quant.act_calib import LayerCalib, build_calibration, cache_path, collect_stats
    from .quant.apply import DEFAULT_EXCLUDE_PATTERNS, iter_quantizable

    comps = list(dict.fromkeys(args.act_calib))
    cmap = adapter.component_map()
    layers: dict = {}
    for c in comps:
        exclude = list(DEFAULT_EXCLUDE_PATTERNS) + list(cmap.exclude.get(c, []))
        allow = list(cmap.allow.get(c, []))
        for qual, mod in iter_quantizable(cmap.roots(c), exclude_patterns=exclude, allow_patterns=allow):
            layers[qual] = mod
    if not layers:
        raise SystemExit(f"--act-calib {comps}: no quantizable layers found")

    from .results import git_commit

    cache = cache_path(args.model, adapter.checkpoint, args.suite, comps, args.act_calib_episodes,
                       episodes_from=args.act_calib_from, tasks=args.act_calib_tasks, seed=args.seed,
                       commit=git_commit())
    if cache.exists():
        raw = torch.load(cache, weights_only=False)
        log.info("act-calib: loaded cached statistics for %d layers from %s", len(raw), cache)
    else:
        cal_tasks = runner.tasks()[: max(1, args.act_calib_tasks)]
        log.info(
            "act-calib: collecting statistics over %d episode(s) x %d task(s) with the un-quantized policy",
            args.act_calib_episodes, len(cal_tasks),
        )
        ep0 = args.act_calib_from
        raw = collect_stats(
            layers,
            lambda: runner.run(
                adapter, None, tasks=cal_tasks, episodes=range(ep0, ep0 + args.act_calib_episodes), log_every=100
            ),
        )
        torch.save(raw, cache)
        log.info("act-calib: cached statistics to %s", cache)
    bank = build_calibration(layers, raw, alpha=args.act_calib_alpha, clip_quantile=args.act_calib_quantile)
    log.info("act-calib: calibrating %d layers in components %s", len(bank), "+".join(comps))
    args._act_calib_provenance = {
        "components": comps, "episodes": args.act_calib_episodes, "episodes_from": args.act_calib_from,
        "tasks": args.act_calib_tasks, "seed": args.seed, "alpha": args.act_calib_alpha,
        "clip_quantile": args.act_calib_quantile, "cache": cache.name, "commit": git_commit(),
    }
    return bank


def apply_quantization(adapter, args, act_calib: dict | None = None) -> dict[str, Any]:
    if getattr(adapter, "name", "") == "remote":
        # quantization lives on the server; record what it reports
        return {"remote": adapter.url, **adapter.info.get("quant_report", {})}
    if getattr(adapter, "quantization_is_remote", False):
        # a third-party policy server: the served checkpoint already carries its own quantization
        return {"served_by": getattr(adapter, "url", "?"), "server_metadata": getattr(adapter, "metadata", {})}
    cmap = adapter.component_map()
    spec = parse_spec(args.preset)
    method = (args.method or "rtn").lower()
    if method == "rtn":
        # `serve` and other callers may not define the run-only flags
        for pattern in (getattr(args, "exclude", None) or []):
            for comp in resolve_scope(args.scope, cmap):
                cmap.exclude.setdefault(comp, []).append(pattern)
        if getattr(args, "only", None):
            # keep everything that does NOT match, i.e. quantize only the named layers
            for comp in resolve_scope(args.scope, cmap):
                cmap.exclude.setdefault(comp, []).append("*")
                cmap.allow.setdefault(comp, []).extend(getattr(args, "only", []))
        reports = quantize_scope(
            cmap, spec, scope=args.scope, include_conv=args.include_conv,
            quantize_lm_head=args.quantize_lm_head, compute_error=args.report_error,
            act_calib=act_calib,
        )
        out = {c: {"layers": r.n_layers, "params": r.n_params, "mean_rel_mse": r.mean_rel_mse()} for c, r in reports.items()}
        if act_calib:
            for c, r in reports.items():
                out[c]["calibrated_layers"] = sum(1 for l in r.layers if l.act_calibrated)
            out["act_calib"] = getattr(args, "_act_calib_provenance", None) or {
                "components": list(args.act_calib), "episodes": getattr(args, "act_calib_episodes", None)}
        return out
    raise SystemExit(f"unknown method {method!r}: this package implements round-to-nearest quantization ('rtn') only")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_presets(_: argparse.Namespace) -> None:
    for name, spec in PRESETS.items():
        print(f"{name:9s} {spec.describe()}")


def cmd_inspect(args: argparse.Namespace) -> None:
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark or "libero", args.suite or "default")
    adapter = build_adapter(args)
    info = adapter.describe()
    print(json.dumps(info, indent=2))
    if args.list_layers:
        from .quant.apply import iter_quantizable

        cmap = adapter.component_map()
        for comp in cmap.present():
            print(f"\n== {comp} ==")
            for name, mod in iter_quantizable(cmap.roots(comp)):
                print(f"  {name:70s} {tuple(mod.weight.shape)}")


def cmd_run(args: argparse.Namespace) -> None:
    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner_cls = get_runner_cls(args.benchmark)
    runner_kwargs: dict[str, Any] = {}
    for kv in args.bench_kwargs or []:
        k, _, v = kv.partition("=")
        runner_kwargs[k] = yaml.safe_load(v)
    runner = runner_cls(
        args.suite, episodes_per_task=args.episodes, seed=args.seed, max_steps_override=args.max_steps,
        video_dir=args.video_dir, **runner_kwargs,
    )
    adapter = build_adapter(args)
    out_model = getattr(adapter, "served_name", None) if args.model == "remote" else args.model
    out = Path(args.out) if args.out else default_out_path(
        out_model or args.model, args.benchmark, args.suite, args.method or "rtn", args.preset.upper(),
        args.scope, args.quantize_lm_head, args.seed
    )
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)  # adopt the model's official benchmark conventions before listing tasks
    act_calib = None
    if getattr(args, "act_calib", None):
        act_calib = _collect_act_calib(adapter, runner, args)
    quant_report = apply_quantization(adapter, args, act_calib=act_calib)
    model_name = getattr(adapter, "served_name", None) if args.model == "remote" else args.model
    header = RunHeader(
        model=model_name or args.model, checkpoint=adapter.checkpoint, benchmark=args.benchmark, suite=args.suite,
        preset=args.preset.upper(), scope=args.scope, method=(args.method or "rtn"),
        baseline_dtype=str(adapter.dtype).replace("torch.", ""), episodes_per_task=runner.episodes_per_task,
        seed=args.seed, quantize_lm_head=args.quantize_lm_head, include_conv=args.include_conv,
        quant_report=quant_report,
        extra={"adapter_options": adapter.options, "runner_options": runner_kwargs, "exclude": args.exclude, "only": args.only,
               "act_calib": (list(args.act_calib) if getattr(args, "act_calib", None) else None)},
    )
    tasks = None
    if args.tasks:
        wanted = {int(t) for t in args.tasks.split(",")}
        tasks = [t for t in runner.tasks() if t.task_id in wanted]
    episodes = range(args.episodes_from, runner.episodes_per_task) if args.episodes_from else None
    with RunWriter(out, header, resume=not args.no_resume) as writer:
        summary = runner.run(adapter, writer, tasks=tasks, episodes=episodes)
    if hasattr(runner, "close"):
        runner.close()
    summary["out"] = str(out)
    print(json.dumps(summary, indent=2))



def cmd_search(args: argparse.Namespace) -> None:
    """QbD-VLA mixed-precision search."""
    from .search.greedy import ProbeCache, WeightVault, qbd_search
    from .search.probe import make_run_block
    from .search.space import MODEL_GROUPS, DEFAULT_LEVELS, GroupSpec, SearchSpace

    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner = get_runner_cls(args.benchmark)(
        args.suite, episodes_per_task=args.probe_from + args.probe_blocks, seed=args.seed,
        max_steps_override=args.probe_horizon,
    )
    adapter = build_adapter(args)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)
        runner.max_steps_override = args.probe_horizon  # keep the probe horizon authoritative

    from .search.space import coarse_groups
    groups = coarse_groups(args.model) if args.groups == "coarse" else MODEL_GROUPS[args.model]
    if args.levels:
        menu = tuple(args.levels)
        groups = [GroupSpec(g.name, g.component, g.only, menu) for g in groups]
    space = SearchSpace(adapter.component_map(), groups)
    if getattr(args, "act_calib", None):
        # calibrated lattice: smoothing/clip vectors from two un-quantized episodes (cached), applied to every level
        space.act_calib = _collect_act_calib(adapter, runner, args)
        log.info("search on the calibrated lattice: %d calibrated layers", len(space.act_calib))
    vault = WeightVault(space)
    out = Path(args.out or f"search/{args.model}-{args.suite}")
    out.mkdir(parents=True, exist_ok=True)
    cache = ProbeCache(out / "probes.jsonl")
    probe_tasks = None
    if args.probe_tasks:
        from .search.probe import select_probe_tasks

        probe_tasks = select_probe_tasks(runner, args.probe_tasks, blocks=args.probe_blocks,
                                         episodes_from=args.probe_from, seed=args.seed)
    n_tasks = len(probe_tasks) if probe_tasks is not None else len(runner.tasks())
    if args.remote_sim:
        # simulator in one short-lived client process per task (X-VLA's EGL-leak workaround)
        from .search.remote_probe import make_remote_run_block, start_server_thread

        start_server_thread(adapter, args.port)
        run_block = make_remote_run_block(
            space, vault, port=args.port, client_python=args.client_python or sys.executable,
            benchmark=args.benchmark, suite=args.suite, n_tasks=n_tasks,
            horizon=args.probe_horizon, episodes_from=args.probe_from,
            env={"MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
        )
    else:
        run_block = make_run_block(adapter, runner, space, vault, episodes_from=args.probe_from, tasks=probe_tasks)
    result = qbd_search(space, run_block, cache, tau=args.tau,
                        block_size=n_tasks, max_blocks=args.probe_blocks,
                        max_episodes=args.max_episodes, retry_rejected=not args.no_retry)
    payload = {
        "model": args.model, "suite": args.suite, "checkpoint": args.checkpoint,
        "probe": {"blocks_max": args.probe_blocks, "from": args.probe_from,
                  "horizon": args.probe_horizon, "tau": args.tau, "n_tasks": n_tasks,
                  "task_ids": [t.task_id for t in probe_tasks] if probe_tasks is not None else None,
                  "retry_rejected": not args.no_retry,
                  "act_calib": (getattr(args, "_act_calib_provenance", None) if getattr(args, "act_calib", None) else None)},
        "groups": {g.name: {"layers": len(space.group_layers(g.name)),
                            "params": space.group_params(g.name), "levels": list(g.levels)}
                   for g in space.groups},
        "path": result.path, "frontier": result.frontier,
        "probe_episodes": result.probes_episodes, "wall_s": round(result.wall_s, 1),
        "stopped_on_budget": result.stopped_on_budget,
    }
    from .search.greedy import select_point
    payload["selected"] = select_point(payload)      # frozen operating-point rule (see select_point)
    (out / "result.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({"frontier": result.frontier, "selected": payload["selected"]["levels"], "probe_episodes": result.probes_episodes,
                      "wall_min": round(result.wall_s / 60, 1), "out": str(out)}, indent=1))



def cmd_baselines(args: argparse.Namespace) -> None:
    """proxy-guided allocations at a matched budget (see search/baselines.py)."""
    from .search.baselines import (allocate, budget_sum, proxy_act_stats, proxy_action_error,
                                   proxy_recon_error, record_trajectory)
    from .search.greedy import WeightVault
    from .search.space import MODEL_GROUPS, SearchSpace

    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark, args.suite)
    runner = get_runner_cls(args.benchmark)(args.suite, episodes_per_task=args.episode + 1, seed=args.seed)
    adapter = build_adapter(args)
    if hasattr(runner, "configure_for"):
        runner.configure_for(adapter)
    space = SearchSpace(adapter.component_map(), MODEL_GROUPS[args.model])
    vault = WeightVault(space)

    # target budget: QbD-VLA's final configuration for this model (result.json), or explicit
    if args.target_sum is None:
        res = json.loads(Path(args.search_result).read_text())
        fin = res["path"][-1]
        target = fin["w_bits_eff"] + fin["a_bits_eff"]
    else:
        target = args.target_sum
    log.info("baselines: target budget sum = %.3f bits", target)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    configs = {}

    task, obs_seq, ref = record_trajectory(adapter, runner, episode=args.episode, seed=args.seed + 1234)
    sens_ae = proxy_action_error(adapter, space, vault, task, obs_seq, ref, seed=args.seed + 1234)
    configs["action_error"] = allocate(space, sens_ae, target)

    vault.restore()
    sens_st = proxy_act_stats(adapter, runner, space, episode=args.episode)
    configs["act_stats"] = allocate(space, sens_st, target)

    vault.restore()
    sens_re = proxy_recon_error(space)
    configs["recon_error"] = allocate(space, sens_re, target)
    vault.restore()

    payload = {"model": args.model, "suite": args.suite, "target_sum": target,
               "configs": {k: {"levels": c.levels, **space.budget(c)} for k, c in configs.items()},
               "sensitivities": {k: {f"{g}|{l}": v for (g, l), v in s.items()}
                                 for k, s in (("action_error", sens_ae), ("act_stats", sens_st),
                                              ("recon_error", sens_re))}}
    (out / "baselines.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["configs"], indent=1))


def cmd_serve(args: argparse.Namespace) -> None:
    from .remote import PolicyServer

    args.checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.benchmark or "libero", args.suite or "default")
    adapter = build_adapter(args)
    if getattr(args, "config", None):
        # a searched/transferred assignment: the same path verify_config.py uses, so served and verified cells match
        from .search.space import MODEL_GROUPS, Config, SearchSpace, coarse_groups
        levels = json.loads(args.config)
        groups = MODEL_GROUPS[args.model] if args.groups == "fine" else coarse_groups(args.model)
        space = SearchSpace(adapter.component_map(), groups)
        known = {g.name for g in space.groups}
        unknown = set(levels) - known
        if unknown:
            raise SystemExit(f"unknown groups in --config: {unknown}; known: {sorted(known)}")
        cfg = Config({g.name: levels.get(g.name, "BASELINE") for g in space.groups})
        applied = space.apply(cfg)
        quant_report = {"qbd_config": cfg.levels, "budget": space.budget(cfg), "applied": applied, "method": "rtn-groups"}
    else:
        quant_report = apply_quantization(adapter, args)
    log.info("serving %s %s | %s %s -> %s", args.model, args.checkpoint, args.method, args.preset, json.dumps(quant_report))
    PolicyServer(adapter, quant_report, host=args.host, port=args.port).serve_forever()


def cmd_tasks(args: argparse.Namespace) -> None:
    runner_kwargs: dict[str, Any] = {}
    for kv in args.bench_kwargs or []:
        k, _, v = kv.partition("=")
        runner_kwargs[k] = yaml.safe_load(v)
    runner = get_runner_cls(args.benchmark)(args.suite, **runner_kwargs)
    tasks = runner.tasks()
    total = 0
    for t in tasks:
        n = len(runner.episodes_for(t))
        total += n
        print(f"{t.task_id:4d}  {t.task_name[:78]:78s}  eps={n:4d}  max_steps={t.max_steps}")
    print(f"# {len(tasks)} tasks, {total} episodes ({args.benchmark}/{args.suite})", file=sys.stderr)


def cmd_summarize(args: argparse.Namespace) -> None:
    from .summarize import collect, pivot, render_csv, render_latex, render_markdown, to_json

    cells = collect(args.root)
    if args.benchmark:
        cells = {k: v for k, v in cells.items() if k[0] == args.benchmark}
    if args.model:
        cells = {k: v for k, v in cells.items() if k[2] == args.model}
    if args.format == "json":
        print(to_json(cells))
        return
    rows = tuple(args.rows.split(","))
    fields, cols, table = pivot(cells, rows=rows)
    if args.format == "md":
        print(render_markdown(fields, cols, table, ci=args.ci))
    elif args.format == "latex":
        print(render_latex(fields, cols, table, ci=args.ci, caption=args.caption))
    else:
        print(render_csv(fields, cols, table))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qbdvla", description="QbD-VLA CLI")
    p.add_argument("--version", action="version", version=f"qbdvla {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="list quantization presets").set_defaults(func=cmd_presets)

    def add_model_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, choices=sorted(MODELS))
        sp.add_argument("--checkpoint", default="auto", help="HF id / path, or 'auto' to read configs/models/<model>.yaml")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--dtype", default=None, help="baseline precision override (bf16/fp16/fp32)")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--model-kwargs", nargs="*", metavar="K=V", help="extra adapter options")

    ins = sub.add_parser("inspect", help="load a model and print its component decomposition")
    add_model_args(ins)
    ins.add_argument("--benchmark", default=None, help="used only to resolve 'auto' checkpoints")
    ins.add_argument("--suite", default=None)
    ins.add_argument("--list-layers", action="store_true")
    ins.set_defaults(func=cmd_inspect)

    run = sub.add_parser("run", help="evaluate one (model, benchmark, suite, preset, scope) cell")
    add_model_args(run)
    run.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    run.add_argument("--suite", required=True)
    run.add_argument("--preset", default="BASELINE", help="W3 | W4 | W8 | W4A4 | W4A8 | W8A8 | W2 | BASELINE | custom e.g. W3G64")
    run.add_argument("--scope", default="e2e", help="e2e | ve | mp | llm | ah | llm+ah ...")
    run.add_argument("--method", default="rtn", help="rtn (round-to-nearest; part of the output file name)")
    run.add_argument("--quantize-lm-head", action="store_true")
    run.add_argument("--include-conv", action="store_true", help="also quantize Conv2d weights (patch embeddings)")
    run.add_argument("--act-calib", nargs="*", metavar="COMP", default=None,
                     help="calibrate activation quantization for these components (SmoothQuant-style "
                          "smoothing folded into weights + clipping) using closed-loop rollouts of the "
                          "un-quantized policy; other quantized components keep plain per-token absmax")
    run.add_argument("--act-calib-alpha", type=float, default=0.5,
                     help="SmoothQuant migration strength (default 0.5)")
    run.add_argument("--act-calib-quantile", type=float, default=0.999,
                     help="clip quantile: 0.999 (default) or 0.99 for harder clipping")
    run.add_argument("--act-calib-from", type=int, default=20,
                     help="first init-state index used for calibration rollouts (default 20: disjoint from the "
                          "evaluation states 0-19; the 2026-08 cells were collected with --act-calib-from 0)")
    run.add_argument("--act-calib-episodes", type=int, default=2,
                     help="episodes per calibration task (default 2)")
    run.add_argument("--act-calib-tasks", type=int, default=1,
                     help="number of suite tasks used for calibration (default 1)")
    run.add_argument("--only", nargs="*", metavar="GLOB", default=None,
                     help="quantize ONLY layers matching these patterns (the dual of --exclude); "
                          "use to isolate a layer group, e.g. --only '*layernorm.dense*'")
    run.add_argument("--exclude", nargs="*", metavar="GLOB", default=None,
                     help="keep layers matching these name patterns at baseline precision "
                          "(ablation, e.g. --exclude '*layernorm.dense*'); recorded in the run header")
    run.add_argument("--report-error", action="store_true", help="record per-layer weight reconstruction error")
    run.add_argument("--episodes", type=int, default=None, help="episodes per task (default: official protocol)")
    run.add_argument("--episodes-from", type=int, default=0)
    run.add_argument("--tasks", default=None, help="comma-separated task ids subset")
    run.add_argument("--max-steps", type=int, default=None)
    run.add_argument("--video-dir", default=None)
    run.add_argument("--bench-kwargs", nargs="*", metavar="K=V")
    run.add_argument("--out", default=None)
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(func=cmd_run)

    srv = sub.add_parser("serve", help="host an adapter (with quantization applied) for `--model remote` clients")
    add_model_args(srv)
    srv.add_argument("--benchmark", default=None, help="used to resolve 'auto' checkpoints")
    srv.add_argument("--suite", default=None)
    srv.add_argument("--preset", default="BASELINE")
    srv.add_argument("--scope", default="e2e")
    srv.add_argument("--method", default="rtn")

    srch = sub.add_parser("search", help="QbD-VLA mixed-precision search")
    srch.add_argument("--model", required=True)
    srch.add_argument("--benchmark", default="libero")
    srch.add_argument("--suite", default="libero_spatial")
    srch.add_argument("--checkpoint", default=None)
    srch.add_argument("--dtype", default=None)
    srch.add_argument("--device", default="cuda")
    srch.add_argument("--seed", type=int, default=0)
    srch.add_argument("--model-kwargs", nargs="*", metavar="K=V")
    srch.add_argument("--probe-blocks", type=int, default=3, help="max blocks per race (1 block = 1 episode/task)")
    srch.add_argument("--probe-from", type=int, default=20, help="first init-state index for probes (disjoint from eval's 0..19)")
    srch.add_argument("--probe-horizon", type=int, default=120)
    srch.add_argument("--act-calib", nargs="*", metavar="COMP", default=None,
                      help="search on the calibrated lattice: SmoothQuant-style smoothing/clip for these components (e.g. ve llm ah)")
    srch.add_argument("--act-calib-alpha", type=float, default=0.5)
    srch.add_argument("--act-calib-quantile", type=float, default=0.999)
    srch.add_argument("--act-calib-from", type=int, default=20)
    srch.add_argument("--act-calib-episodes", type=int, default=2)
    srch.add_argument("--act-calib-tasks", type=int, default=1)
    srch.add_argument("--no-retry", action="store_true",
                      help="ablation: never reconsider a move once it has been rejected (the default retries "
                           "rejected moves after every accept, so a move can pass after another group has changed)")
    srch.add_argument("--probe-tasks", type=int, default=None,
                      help="probe on this many tasks, stratified over families (default: every task; "
                           "needed for suites with hundreds of heterogeneous tasks, e.g. SIMPLER)")
    srch.add_argument("--tau", type=float, default=0.90)
    srch.add_argument("--max-episodes", type=int, default=6000)
    srch.add_argument("--groups", choices=("fine", "coarse"), default="fine",
                      help="group granularity: fine = MODEL_GROUPS, coarse = one group per component (ablation)")
    srch.add_argument("--levels", nargs="*", default=None, help="override the per-group level menu")
    srch.add_argument("--remote-sim", action="store_true",
                      help="run probe episodes in per-task client processes against an in-process policy server")
    srch.add_argument("--client-python", default=None, help="interpreter for probe clients (default: this one)")
    srch.add_argument("--port", type=int, default=8020)
    srch.add_argument("--out", default=None)
    srch.set_defaults(func=cmd_search)

    bl = sub.add_parser("baselines", help="sensitivity-based allocations at a matched budget")
    bl.add_argument("--model", required=True)
    bl.add_argument("--benchmark", default="libero")
    bl.add_argument("--suite", default="libero_spatial")
    bl.add_argument("--checkpoint", default=None)
    bl.add_argument("--dtype", default=None)
    bl.add_argument("--device", default="cuda")
    bl.add_argument("--seed", type=int, default=0)
    bl.add_argument("--model-kwargs", nargs="*", metavar="K=V")
    bl.add_argument("--episode", type=int, default=20, help="init-state index used for the recorded trajectory / stats")
    bl.add_argument("--search-result", default=None, help="result.json of a search whose final budget is matched")
    bl.add_argument("--target-sum", type=float, default=None, help="explicit w+a effective-bit target")
    bl.add_argument("--out", required=True)
    bl.set_defaults(func=cmd_baselines)


    srv.add_argument("--config", default=None,
                     help="serve a group-level configuration (JSON group->level, as verify_config.py takes) instead of a uniform preset")
    srv.add_argument("--groups", choices=("fine", "coarse"), default="fine")
    srv.add_argument("--quantize-lm-head", action="store_true")
    srv.add_argument("--include-conv", action="store_true")
    srv.add_argument("--report-error", action="store_true")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8010)
    srv.set_defaults(func=cmd_serve)

    sm = sub.add_parser("summarize", help="aggregate results into tables")
    sm.add_argument("root", nargs="?", default=str(REPO_ROOT / "results"))
    sm.add_argument("--format", choices=("md", "latex", "csv", "json"), default="md")
    sm.add_argument("--rows", default="benchmark,suite,model,method,scope")
    sm.add_argument("--benchmark", default=None)
    sm.add_argument("--model", default=None)
    sm.add_argument("--ci", action="store_true", help="print 95%% confidence intervals")
    sm.add_argument("--caption", default="")
    sm.set_defaults(func=cmd_summarize)

    lt = sub.add_parser("tasks", help="list a benchmark suite's official task table")
    lt.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    lt.add_argument("--suite", required=True)
    lt.add_argument("--bench-kwargs", nargs="*", metavar="K=V")
    lt.set_defaults(func=cmd_tasks)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
