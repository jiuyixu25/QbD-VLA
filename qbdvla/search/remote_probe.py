"""Probe blocks with the simulator in separate processes (X-VLA path).

X-VLA's simulator environment hangs when a second LIBERO task environment is
built inside one process (a MuJoCo/EGL context leak specific to that stack),
so the in-process ``run_block`` of :mod:`.probe` cannot run its 10-task
blocks. This module keeps the model — and therefore weight mutation, the
vault and the activation hooks — in the search process, exposes it through
the package's policy server (:mod:`qbdvla.remote`) on a background thread, and drives each
task's episode with a short-lived client process (``qbdvla run --model remote``),
one task per process, the same harness as the full X-VLA
evaluation. Clients are spawned only after a configuration is applied and
are awaited before the weights are restored, so the served model is never
mutated while a request is in flight.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from .greedy import WeightVault
from .space import Config, SearchSpace

log = logging.getLogger(__name__)

__all__ = ["start_server_thread", "make_remote_run_block"]


def start_server_thread(adapter, port: int, quant_report: dict | None = None) -> threading.Thread:
    from ..remote import PolicyServer

    server = PolicyServer(adapter, quant_report or {}, host="127.0.0.1", port=port)
    th = threading.Thread(target=server.serve_forever, name=f"policy-server-{port}", daemon=True)
    th.start()
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=2).read()
            log.info("policy server thread up on :%d", port)
            return th
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"policy server on :{port} did not come up")


def _count(path: Path) -> tuple[int, int]:
    n = ok = 0
    if not path.exists():
        return 0, 0
    for line in path.open():
        if '"kind": "episode"' in line:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            ok += bool(rec.get("success"))
    return ok, n


def make_remote_run_block(
    space: SearchSpace,
    vault: WeightVault,
    *,
    port: int,
    client_python: str,
    benchmark: str,
    suite: str,
    n_tasks: int,
    horizon: int,
    episodes_from: int,
    env: dict[str, str] | None = None,
    per_task_timeout_s: int = 900,
):
    """Return ``run_block(cfg, block_idx) -> successes`` (out of ``n_tasks``)."""
    base_env = dict(os.environ)
    base_env.update(env or {})
    scratch = Path(tempfile.mkdtemp(prefix="qbd-remote-probe-"))

    def run_tasks(block_idx: int) -> int:
        ep = episodes_from + block_idx
        successes = 0
        for task in range(n_tasks):
            out = scratch / f"b{block_idx}-t{task}.jsonl"
            out.unlink(missing_ok=True)
            cmd = [client_python, "-m", "qbdvla.cli", "run",
                   "--model", "remote", "--model-kwargs", f"url=http://127.0.0.1:{port}",
                   "--benchmark", benchmark, "--suite", suite,
                   "--preset", "BASELINE", "--scope", "e2e",
                   "--episodes", str(ep + 1), "--episodes-from", str(ep),
                   "--tasks", str(task), "--max-steps", str(horizon),
                   "--no-resume", "--out", str(out)]
            try:
                subprocess.run(cmd, env=base_env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=per_task_timeout_s, check=False)
            except subprocess.TimeoutExpired:
                log.warning("probe client for task %d timed out (%ds); counting as failure", task, per_task_timeout_s)
            ok, n = _count(out)
            if n == 0:
                log.warning("probe client for task %d produced no episode; counting as failure", task)
            successes += ok
        return successes

    def run_block(cfg: Config, block_idx: int) -> int:
        t0 = time.time()
        vault.restore()
        space.apply(cfg)
        try:
            successes = run_tasks(block_idx)
        finally:
            vault.restore()
        log.info("remote block %d %-48s -> %d/%d (%.0fs)", block_idx, cfg.pretty()[:48],
                 successes, n_tasks, time.time() - t0)
        return successes

    return run_block
