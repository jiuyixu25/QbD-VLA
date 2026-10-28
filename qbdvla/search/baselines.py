"""Proxy-guided precision allocation baselines.

Each baseline is one *signal* the field actually uses to decide where bits
go, fed to one shared budget-matched greedy allocator, so the comparison
against QbD-VLA isolates the signal:

* ``action_error``   – open-loop deviation from the full-precision policy's
  actions when a group is quantized (the QVLA-style "action-space
  sensitivity"). Measured by replaying one recorded observation sequence
  through the policy under each (group, level), with identical RNG seeding so
  stochastic heads (flow matching) are compared like-for-like.
* ``act_stats``      – on-policy activation kurtosis per group (calibration-
  time outlier statistics; groups with the fewest outliers are lowered first).
* ``recon_error``    – parameter-weighted weight relative-MSE at the level's
  weight grid plus an activation-grid relative-error estimate (the
  reconstruction objective of GPTQ/OBQ-style methods).

The allocator descends from the all-baseline configuration, always taking the
lowest-sensitivity legal one-level-down move, until the effective bit budget
reaches the target (QbD-VLA's final budget for the same model). The resulting
configuration is then verified under the full closed-loop protocol.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable

import numpy as np
import torch

from ..quant.fake_quant import weight_quant_error
from ..quant.presets import parse_spec
from .greedy import WeightVault
from .space import Config, SearchSpace

log = logging.getLogger(__name__)

__all__ = ["record_trajectory", "proxy_action_error", "proxy_act_stats",
           "proxy_recon_error", "allocate", "budget_sum"]


# ---------------------------------------------------------------- observation replay
def _safe_copy(obj):
    """Deep-copy arrays/containers; pass simulator handles (env objects, ctypes) by reference."""
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _safe_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_safe_copy(v) for v in obj)
    if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        return obj
    return obj


def record_trajectory(adapter, runner, *, episode: int = 20, task_index: int = 0, seed: int = 1234):
    """Run one full-precision episode and record every observation the policy saw
    and every action it returned. Returns (task, [obs...], [action...])."""
    task = runner.tasks()[task_index]
    obs_log, act_log = [], []
    orig_act = adapter.act

    def recorder(obs, *a, **k):
        obs_log.append(_safe_copy(obs))
        out = orig_act(obs, *a, **k)
        act_log.append(np.asarray(out, dtype=np.float64).copy())
        return out

    adapter.act = recorder
    torch.manual_seed(seed)
    try:
        runner.run(adapter, None, tasks=[task], episodes=range(episode, episode + 1), log_every=1000)
    finally:
        adapter.act = orig_act
    log.info("recorded %d observations from task %s", len(obs_log), task.task_name)
    return task, obs_log, act_log


@torch.no_grad()
def replay_actions(adapter, task, obs_seq, *, seed: int = 1234) -> list[np.ndarray]:
    """Open-loop replay of ``obs_seq`` under a fixed RNG seed; returns the action sequence."""
    # Adapters with stochastic heads advance a private episode seed inside ``reset``
    # (so consecutive episodes draw fresh flow-matching noise). Freeze it here so every
    # replay -- reference and quantized -- draws the *same* noise sequence.
    frozen = getattr(adapter, "_episode_seed", None)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    adapter.reset(task)
    if frozen is not None:
        adapter._episode_seed = frozen
    return [np.asarray(adapter.act(_safe_copy(obs), task), dtype=np.float64).copy() for obs in obs_seq]


def _mean_dev(a_seq, b_seq) -> float:
    devs = []
    for a, b in zip(a_seq, b_seq):
        n = min(len(a), len(b))
        devs.append(float(np.mean(np.abs(a[:n] - b[:n]))))
    return float(np.mean(devs))


def proxy_action_error(adapter, space: SearchSpace, vault: WeightVault, task, obs_seq, ref_actions,
                       *, seed: int = 1234) -> dict[tuple[str, str], float]:
    """sens[(group, level)] = action deviation from the *full-precision replay* when only
    that group is at that level. Using an FP replay (same seed, same call sequence) as the
    reference — rather than the recorded actions — removes the RNG-position mismatch of
    stochastic heads; the FP-vs-FP repeatability is logged as the true noise floor."""
    vault.restore()
    ref = replay_actions(adapter, task, obs_seq, seed=seed)
    ref2 = replay_actions(adapter, task, obs_seq, seed=seed)
    floor = _mean_dev(ref, ref2)
    log.info("action-error proxy: FP replay repeatability %.5f (vs recorded actions %.5f)",
             floor, _mean_dev(ref, ref_actions))
    if floor > 1e-4:
        log.warning("action-error proxy: replay is not deterministic (floor %.4f); rankings will be noisy", floor)
    sens = {}
    for g in space.groups:
        for level in g.levels[1:]:
            cfg = space.initial()
            cfg.levels[g.name] = level
            vault.restore()
            space.apply(cfg)
            try:
                d = _mean_dev(replay_actions(adapter, task, obs_seq, seed=seed), ref)
            finally:
                vault.restore()
            sens[(g.name, level)] = d
            log.info("  action-error %-16s %-8s dev %.5f", g.name, level, d)
    return sens


# ---------------------------------------------------------------- activation statistics
class _KurtObserver:
    def __init__(self):
        self.sum_k, self.n = 0.0, 0

    def __call__(self, module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            return
        x = args[0]
        if not x.is_floating_point() or x.shape[-1] != module.weight.shape[1]:
            return
        with torch.no_grad():
            xa = x.reshape(-1, x.shape[-1]).abs().float()
            if xa.numel() < 8:
                return
            mu, sd = xa.mean(), xa.std().clamp_min(1e-12)
            self.sum_k += float((((xa - mu) / sd) ** 4).mean())
            self.n += 1


def proxy_act_stats(adapter, runner, space: SearchSpace, *, episode: int = 20, task_index: int = 0):
    """sens[(group, level)] = mean activation kurtosis of the group (level-independent)."""
    observers, handles = {}, []
    for g in space.groups:
        for qual, mod in space.group_layers(g.name):
            obs = _KurtObserver()
            observers[qual] = (g.name, obs)
            handles.append(mod.register_forward_pre_hook(obs))
    task = runner.tasks()[task_index]
    try:
        runner.run(adapter, None, tasks=[task], episodes=range(episode, episode + 1), log_every=1000)
    finally:
        for h in handles:
            h.remove()
    per_group: dict[str, list[float]] = {}
    for qual, (gname, obs) in observers.items():
        if obs.n:
            per_group.setdefault(gname, []).append(obs.sum_k / obs.n)
    sens = {}
    for g in space.groups:
        k = float(np.mean(per_group.get(g.name, [0.0])))
        for level in g.levels[1:]:
            sens[(g.name, level)] = k
        log.info("  act-stats %-16s mean kurtosis %.1f", g.name, k)
    return sens


# ---------------------------------------------------------------- reconstruction error
def proxy_recon_error(space: SearchSpace, act_ref_bits: int = 4, act_rel_err_ref: float | None = None):
    """sens[(group, level)] = param-weighted weight rel-MSE at the level's weight grid
    + an activation-grid relative-error estimate (scaled from a reference width)."""
    sens = {}
    for g in space.groups:
        layers = space.group_layers(g.name)
        for level in g.levels[1:]:
            spec = parse_spec(level)
            w_err = 0.0
            tot = 0
            if spec.weight is not None:
                for _, mod in layers:
                    n = mod.weight.numel()
                    w_err += n * weight_quant_error(mod.weight.data, spec.weight)["rel_mse"]
                    tot += n
                w_err = w_err / max(tot, 1)
            a_err = 0.0
            if spec.act is not None:
                ref = 0.35 if act_rel_err_ref is None else act_rel_err_ref  # measured A4 rel-err scale
                a_err = ref * (2.0 ** (act_ref_bits - spec.act.bits))
            sens[(g.name, level)] = w_err + a_err
            log.info("  recon-error %-16s %-8s w %.4f a %.4f", g.name, level, w_err, a_err)
    return sens


# ---------------------------------------------------------------- shared allocator
def budget_sum(space: SearchSpace, cfg: Config) -> float:
    b = space.budget(cfg)
    return b["w_bits_eff"] + b["a_bits_eff"]


def allocate(space: SearchSpace, sens: dict[tuple[str, str], float], target_sum: float) -> Config:
    """Greedy budget-matched descent: always take the lowest-sensitivity legal
    one-level-down move until the summed effective bits reach ``target_sum``."""
    cfg = space.initial()
    while budget_sum(space, cfg) > target_sum + 1e-9:
        moves = space.moves(cfg)
        if not moves:
            break
        gname, level, nxt = min(moves, key=lambda m: sens.get((m[0], m[1]), float("inf")))
        cfg = nxt
        log.info("  allocate: %s -> %s  (sens %.5f)  budget %.3f", gname, level,
                 sens.get((gname, level), float("nan")), budget_sum(space, cfg))
    return cfg
