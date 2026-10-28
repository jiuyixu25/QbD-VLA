"""Closed-loop probe blocks: one block = one episode on every task.

Probe episodes use init-state indices *disjoint* from the full-protocol
evaluation episodes (which use indices 0..19), so the search can never overfit
the states it is finally judged on. Horizon is enforced with the runner's
``max_steps_override``; per VLAQuantBench's horizon analysis this changes
magnitudes, not orderings. Block ``i`` uses init-state index
``episodes_from + i`` on all tasks, so every configuration is scored on the
same episodes (paired comparisons, lower variance).
"""

from __future__ import annotations

import logging
import random
import time

from .greedy import WeightVault
from .space import Config, SearchSpace

log = logging.getLogger(__name__)

__all__ = ["make_run_block", "select_probe_tasks"]


def select_probe_tasks(runner, n_tasks: int, *, blocks: int = 3, episodes_from: int = 0, seed: int = 0):
    """Pick ``n_tasks`` probe tasks, stratified over task families, deterministically.

    Benchmarks whose suites hold hundreds of heterogeneous tasks (SIMPLER: 244 variant
    entries with 1-60 episodes each) make "one episode on every task" far too large for a
    probe block, and tasks with fewer episodes than the block index cannot be indexed at
    all. This filters to tasks that can serve every requested block, then round-robins over
    ``extra["family"]`` so the probe keeps the suite's task mix rather than one family's.
    """
    tasks = list(runner.tasks())
    need = episodes_from + blocks
    usable = [t for t in tasks if len(runner.episodes_for(t)) >= need]
    if not usable:
        raise ValueError(
            f"no task in {runner.suite!r} has {need} episodes (episodes_from={episodes_from} + "
            f"blocks={blocks}); lower --probe-from or --probe-blocks")
    if n_tasks >= len(usable):
        return usable
    by_family: dict[str, list] = {}
    for t in usable:
        by_family.setdefault(str(t.extra.get("family", "")), []).append(t)
    rng = random.Random(seed)
    for fam in by_family.values():
        rng.shuffle(fam)
    picked, families = [], sorted(by_family)
    while len(picked) < n_tasks:
        progressed = False
        for fam in families:
            if by_family[fam] and len(picked) < n_tasks:
                picked.append(by_family[fam].pop())
                progressed = True
        if not progressed:
            break
    picked.sort(key=lambda t: t.task_id)
    log.info("probe tasks: %d of %d usable (%d total) across %d families",
             len(picked), len(usable), len(tasks), len(families))
    return picked


def make_run_block(
    adapter,
    runner,
    space: SearchSpace,
    vault: WeightVault,
    *,
    episodes_from: int = 20,
    tasks=None,
):
    """Return ``run_block(cfg, block_idx) -> successes`` (out of #probe tasks)."""

    def run_block(cfg: Config, block_idx: int) -> int:
        t0 = time.time()
        vault.restore()
        space.apply(cfg)
        try:
            summary = runner.run(
                adapter, None,
                tasks=tasks,
                episodes=range(episodes_from + block_idx, episodes_from + block_idx + 1),
                log_every=1000,
            )
            n = int(summary["success_rate"]["n"])
            succ = int(round(summary["success_rate"]["point"] * n))
        finally:
            vault.restore()
        log.info("block %d %-52s -> %d/%d (%.0fs)", block_idx, cfg.pretty()[:52], succ, n, time.time() - t0)
        return succ

    return run_block
