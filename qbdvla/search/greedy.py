"""Task-guided bit allocation (Section 3.3 of the paper) and configuration selection (Section 3.4).

Starting from full precision, every candidate lowers one layer group by one level while every other group keeps its
current level, so each candidate is scored as a *joint* configuration, and every score is *closed-loop* task success
from short evaluation batches ("probes": one episode per task, see ``probe.py``). Candidates are tried in order of
decreasing cost saving, and the first one that passes the sequential test of Section 3.2 (:func:`race`) against the
floor ``tau`` x the full-precision probe score is accepted. After every accept the previously rejected moves become
eligible again, because the effect of lowering a group depends on the levels of the other groups.

Weight mutation is in place, so the runner snapshots every group weight once (CPU, original dtype) and restores between
configurations; activation hooks are removed between configurations. Every probe is appended to a JSONL log keyed by
configuration, which makes a search resumable: re-running skips everything already scored.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from ..quant.apply import remove_activation_hooks
from .space import Config, SearchSpace

log = logging.getLogger(__name__)


class _BudgetExhausted(RuntimeError):
    """Raised inside a probe block when the episode budget runs out.

    Exhaustion is a *stopping condition*, not a failure: the descent so far is a valid
    (if budget-limited) result, so :func:`qbd_search` catches this, records that the run stopped on
    budget, and returns the path it has. Letting it propagate would throw away every probe
    episode spent -- on SIMPLER that was a day of compute with no ``result.json`` written.
    """

__all__ = ["WeightVault", "ProbeCache", "race", "qbd_search", "select_point", "SearchResult"]


class WeightVault:
    """CPU snapshot of every group layer's weights; restore() undoes quantization."""

    def __init__(self, space: SearchSpace):
        self.space = space
        self._saved: dict[str, torch.Tensor] = {}
        for g in space.groups:
            for qual, mod in space.group_layers(g.name):
                self._saved[qual] = mod.weight.detach().to("cpu", copy=True)
        log.info("weight vault: %d layers snapshotted (%.1f MB)",
                 len(self._saved),
                 sum(t.numel() * t.element_size() for t in self._saved.values()) / 2 ** 20)

    @torch.no_grad()
    def restore(self) -> None:
        for g in self.space.groups:
            for qual, mod in self.space.group_layers(g.name):
                mod.weight.data.copy_(self._saved[qual].to(mod.weight.device))
                if hasattr(mod, "_qbd_quantized_with"):
                    delattr(mod, "_qbd_quantized_with")
            remove_activation_hooks(self.space.cmap.roots(g.component))


class ProbeCache:
    """JSONL-backed memo of config -> probe result; makes searches resumable."""

    def __init__(self, path: Path):
        self.path = path
        self._mem: dict[str, dict] = {}
        if path.exists():
            for line in path.open():
                try:
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
            log.info("probe cache: %d entries loaded from %s", len(self._mem), path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, cfg: Config) -> dict | None:
        return self._mem.get(cfg.key())

    def put(self, cfg: Config, rec: dict) -> dict:
        rec = {"key": cfg.key(), "levels": cfg.levels, **rec}
        self._mem[cfg.key()] = rec
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec


@dataclass
class SearchResult:
    path: list[dict]        # accepted configs in order, with budgets and scores
    frontier: list[dict]    # same, deduplicated by budget (monotone)
    probes_episodes: int    # total probe episodes spent
    wall_s: float
    stopped_on_budget: bool = False   # True if the descent ended on --max-episodes, not on the space


def _wilson(successes: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def select_point(result: dict, *, z: float = 1.28) -> dict:
    """The operating point reported for a search: the *cheapest* accepted configuration whose probe score is
    statistically indistinguishable from the best probe score on the accepted path, i.e. whose one-sided Wilson
    upper bound (at the racing z, 90 %) reaches that maximum. A point accepted after one block at 10/10 and a
    point raced to 27/30 are thereby compared on their uncertainty, not on raw fractions: a late move that lowers
    the probe score beyond noise (X-VLA) is not reported as the operating point, while noise-level fluctuations
    among early, expensive points do not pull the selection away from the endpoint.
    Returns the path entry plus ``index``, ``best_score`` and ``rule``."""
    path = result["path"]

    def score_of(p):
        return float(p["score"])

    best = max(score_of(p) for p in path[1:]) if len(path) > 1 else score_of(path[0])
    cands = []
    for i, p in enumerate(path):
        if i == 0 and len(path) > 1:
            continue                            # the unquantized anchor is never a compression result
        n = int(p["n"]); k = round(score_of(p) * n)
        _, hi = _wilson(k, n, z)
        if hi >= best - 1e-9:
            cands.append((p["w_bits_eff"] + p["a_bits_eff"], -i, p))
    bud, negi, p = min(cands, key=lambda t: (t[0], t[1]))
    return {**p, "index": -negi, "best_score": best, "rule": f"cheapest accepted point whose one-sided Wilson upper bound (z={z}) on the "
            f"probe score reaches the best probe score on the path ({best:.3f})"}


def race(
    cfg: Config,
    floor: float | None,
    run_block,                    # (cfg, block_idx) -> successes out of block_size
    cache: ProbeCache,
    *,
    block_size: int,
    max_blocks: int,
    z_accept: float = 1.28,       # ~80%: a hasty accept is caught by the verify stage
    z_reject: float = 1.96,       # ~95%: a hasty reject silently truncates the frontier
) -> tuple[str, dict]:
    """Sequential test of one joint config against the acceptance floor (Section 3.2).

    Adds probe episodes block by block until a Wilson interval separates the
    config from ``floor`` or the budget is spent; at budget the point estimate
    decides. The test is asymmetric: rejecting needs 95% confidence while
    accepting needs only 80%, because the error costs differ — a false accept
    is caught by the full-protocol verify stage, but a false reject silently
    truncates the frontier (observed in the first pi0.5 run: a true-0.9 move
    was killed by a 6/10 draw under a symmetric 80% band). Cached counts
    resume across calls, so a retried config only pays for what it still
    needs.
    """
    st = cache.get(cfg) or {}
    succ, n = int(st.get("successes", 0)), int(st.get("n", 0))
    while True:
        if floor is None:                       # anchor mode: exhaust the budget
            if n >= max_blocks * block_size:
                verdict = "anchor"
                break
        elif n > 0:
            if _wilson(succ, n, z_accept)[0] >= floor:
                verdict = "safe"
                break
            if _wilson(succ, n, z_reject)[1] <= floor:
                verdict = "unsafe"
                break
            if n >= max_blocks * block_size:
                verdict = "safe" if succ / n >= floor else "unsafe"
                break
        succ += int(run_block(cfg, n // block_size))
        n += block_size
        cache.put(cfg, {"successes": succ, "n": n, "score": succ / n})
    rec = {"successes": succ, "n": n, "score": succ / max(n, 1), "verdict": verdict}
    cache.put(cfg, rec)
    return verdict, rec


def qbd_search(
    space: SearchSpace,
    run_block,                    # (cfg, block_idx) -> successes out of block_size
    cache: ProbeCache,
    *,
    tau: float = 0.90,
    min_anchor: float = 0.5,
    block_size: int = 10,
    max_blocks: int = 3,
    max_episodes: int = 6000,
    retry_rejected: bool = True,
) -> SearchResult:
    """Task-guided bit allocation with sequential (racing) probes.

    Candidates are tried in order of decreasing bit saving and the first one
    that proves safe is accepted; close calls automatically receive more
    episodes, clear calls almost none. Rejected moves become eligible again
    after every accept (they are new joint configs then), so a move that
    fails at first can pass after another group has changed.
    """
    t0 = time.time()
    spent = {"eps": 0}

    def counted_block(cfg, i):
        if spent["eps"] >= max_episodes:
            raise _BudgetExhausted(f"probe episode budget exhausted ({max_episodes})")
        got = run_block(cfg, i)
        spent["eps"] += block_size
        return got

    current = space.initial()
    # the anchor sets the floor -> estimate it at full budget
    try:
        _, base = race(current, None, counted_block, cache,
                       block_size=block_size, max_blocks=max_blocks)   # anchor: exhaust the budget
    except _BudgetExhausted as exc:
        raise RuntimeError(f"{exc}: the budget did not even cover the full-precision anchor; "
                           "raise --max-episodes or lower --probe-blocks/--probe-tasks") from None
    floor = tau * base["score"]
    log.info("QbD-VLA anchor %.3f (n=%d) -> floor %.3f", base["score"], base["n"], floor)
    if base["score"] < min_anchor:
        # A relative criterion is meaningless when the full-precision policy itself cannot
        # solve the probe tasks (e.g. a probe horizon shorter than the tasks need): every
        # move would pass and the search degenerates to uniform lowest precision (H=60).
        raise RuntimeError(
            f"full-precision probe score {base['score']:.2f} < min_anchor {min_anchor}: the probe "
            "horizon/tasks do not let the reference policy succeed, so tau-relative acceptance "
            "is vacuous; lengthen --probe-horizon or change the probe tasks")

    path = [{"levels": dict(current.levels), "score": base["score"], "n": base["n"], **space.budget(current)}]
    stopped_on_budget = False
    rejected: set[tuple[str, str]] = set()   # only consulted when retry_rejected is False
    while True:
        moves = space.moves(current)
        if not retry_rejected:
            moves = [m for m in moves if (m[0], m[1]) not in rejected]
        if not moves:
            break
        b_now = space.budget(current)
        ranked = sorted(
            moves,
            key=lambda m: (b_now["w_bits_eff"] - space.budget(m[2])["w_bits_eff"])
            + (b_now["a_bits_eff"] - space.budget(m[2])["a_bits_eff"]),
            reverse=True,
        )
        accepted = None
        try:
            for gname, level, cfg in ranked:
                verdict, rec = race(cfg, floor, counted_block, cache,
                                    block_size=block_size, max_blocks=max_blocks)
                log.info("  race %-16s -> %-8s %s (%d/%d)", gname, level, verdict, rec["successes"], rec["n"])
                if verdict == "safe":
                    accepted = (gname, level, cfg, rec)
                    break
                rejected.add((gname, level))
        except _BudgetExhausted as exc:
            stopped_on_budget = True
            log.warning("QbD-VLA: %s; returning the %d-step path found so far", exc, len(path) - 1)
            break
        if accepted is None:
            log.info("QbD-VLA: no safe move; stopping at %s", current.pretty())
            break
        gname, level, cfg, rec = accepted
        current = cfg
        log.info("QbD-VLA accept %s -> %s  score %.3f  budget %s", gname, level, rec["score"], space.budget(current))
        path.append({"levels": dict(current.levels), "score": rec["score"], "n": rec["n"], **space.budget(current)})

    frontier = []
    best = -1.0
    for step in reversed(path):
        if step["score"] > best:
            frontier.append(step)
            best = step["score"]
    frontier.reverse()
    return SearchResult(path=path, frontier=frontier, probes_episodes=spent["eps"],
                      wall_s=time.time() - t0, stopped_on_budget=stopped_on_budget)
