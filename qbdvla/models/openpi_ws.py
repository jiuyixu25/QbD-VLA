"""Client adapter for a policy served over the openpi WebSocket protocol.

This exists so that a *third-party* policy server can be driven by this paper's own LIBERO
protocol: the same tasks, the same official initial states, the same horizon and the same success
predicate as every other cell, with the policy itself entirely outside our code.

It is used for ActQuant (arXiv 2605.24011): their released runtime ``tools/pi0.5/serve_policy.py``
loads their GGUF checkpoint through their C++/GGML pipeline and answers on the openpi protocol
(msgpack-numpy over a WebSocket: the server sends a metadata frame on connect, then one action
chunk per observation frame). We send the observation keys their server reads
(``observation/image``, ``observation/wrist_image``, ``prompt``) and execute the returned chunk.

Nothing is quantized on this side; ``--preset BASELINE`` is the only valid preset, and what the
served checkpoint is must be recorded by the caller.
"""

from __future__ import annotations

import collections
import logging
from typing import Any, ClassVar

import numpy as np

from ..components import ComponentMap
from .base import Observation, TaskSpec, VLAAdapter

log = logging.getLogger(__name__)


class OpenPIWebsocketAdapter(VLAAdapter):
    name: ClassVar[str] = "openpi_ws"
    default_dtype: ClassVar[str] = "fp32"
    supported_benchmarks: ClassVar[tuple[str, ...]] = ("libero",)
    #: quantization lives in the served checkpoint, not in our process
    quantization_is_remote: ClassVar[bool] = True
    #: LIBERO conventions: identical to our own pi0.5 cells, so a drop measured here is measured
    #: in the same protocol (openpi's own LIBERO example instead uses 256 px and replans every 5 steps)
    libero_max_steps: ClassVar[dict[str, int]] = {
        "libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
    }
    libero_camera_size: ClassVar[int | None] = 360
    libero_action_mode: ClassVar[str] = "delta"

    def __init__(self, checkpoint: str = "served", *, device="cpu", dtype=None, seed: int = 0, **kwargs: Any):
        super().__init__(checkpoint, device="cpu", dtype="fp32", seed=seed, **kwargs)
        self.url = str(kwargs.get("url", "ws://127.0.0.1:8000"))
        self.replan_steps = int(kwargs.get("replan_steps", 50))   # 50 = our pi0.5 chunk; their script uses 5
        self.action_dim = int(kwargs.get("action_dim", 7))
        self.send_state = bool(kwargs.get("send_state", True))
        # first inference of a served policy may include torch.compile autotuning (tens of minutes)
        self.timeout = float(kwargs.get("timeout", 7200))
        self.metadata: dict[str, Any] = {}
        self._ws = None
        self._queue: collections.deque = collections.deque()
        self._chunks = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pack(obj):
        import msgpack

        def enc(o):
            if isinstance(o, np.ndarray):
                return {b"__ndarray__": True, b"data": o.tobytes(), b"dtype": o.dtype.str, b"shape": list(o.shape)}
            if isinstance(o, np.generic):
                return {b"__npgeneric__": True, b"data": o.item(), b"dtype": o.dtype.str}
            if isinstance(o, dict):
                return {k: enc(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [enc(v) for v in o]
            return o

        return msgpack.packb(enc(obj), use_bin_type=True)

    @staticmethod
    def _unpack(data):
        import msgpack

        def hook(d):
            if b"__ndarray__" in d:
                return np.frombuffer(d[b"data"], dtype=np.dtype(d[b"dtype"])).reshape(tuple(d[b"shape"]))
            if b"__npgeneric__" in d:
                return np.dtype(d[b"dtype"]).type(d[b"data"])
            return d

        return msgpack.unpackb(data, object_hook=hook, raw=False, strict_map_key=False)

    def load(self) -> None:
        from websockets.sync.client import connect

        # no keepalive: the first inference of a served policy includes warm-up and easily exceeds the
        # library's 20 s ping timeout, which would close the connection mid-episode
        self._ws = connect(self.url, max_size=None, open_timeout=self.timeout,
                           ping_interval=None, ping_timeout=None, close_timeout=self.timeout)
        try:
            self.metadata = self._unpack(self._ws.recv(timeout=self.timeout)) or {}
        except Exception as e:  # a server that sends no metadata frame is still usable
            log.warning("no metadata frame from %s: %s", self.url, e)
            self.metadata = {}
        self.model = None
        log.info("connected to %s (metadata: %s)", self.url, self.metadata)

    def build_component_map(self) -> ComponentMap:
        return ComponentMap(notes={"served": f"policy served at {self.url}; quantization is baked into that checkpoint"})

    def component_map(self) -> ComponentMap:
        if self._cmap is None:
            self._cmap = self.build_component_map()
        return self._cmap

    def describe(self) -> dict[str, Any]:
        return {"model": "openpi_ws", "url": self.url, "replan_steps": self.replan_steps,
                "server_metadata": self.metadata, "chunks_requested": self._chunks}

    def reset(self, task: TaskSpec) -> None:
        self._queue.clear()

    def act(self, obs: Observation, task: TaskSpec) -> np.ndarray:
        if not self._queue:
            payload: dict[str, Any] = {
                "observation/image": np.ascontiguousarray(obs.images["agentview"], dtype=np.uint8),
                "prompt": task.instruction,
            }
            if "wrist" in obs.images:
                payload["observation/wrist_image"] = np.ascontiguousarray(obs.images["wrist"], dtype=np.uint8)
            if self.send_state and obs.state is not None:
                payload["observation/state"] = np.asarray(obs.state, dtype=np.float32)
            with self.inference_meter.measure():
                self._ws.send(self._pack(payload))
                raw = self._ws.recv(timeout=self.timeout)
            if isinstance(raw, str):  # openpi's server sends traceback.format_exc() as text on failure
                raise RuntimeError(f"policy server error:\n{raw[-2000:]}")
            out = self._unpack(raw)
            self._chunks += 1
            if not isinstance(out, dict) or "actions" not in out:
                raise RuntimeError(f"policy server returned {type(out)} without 'actions': {str(out)[:200]}")
            chunk = np.asarray(out["actions"], dtype=np.float64)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            for a in chunk[: self.replan_steps]:
                self._queue.append(np.asarray(a[: self.action_dim], dtype=np.float64))
        return self._queue.popleft()

    def peak_vram_gb(self) -> float | None:
        return None

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:  # pragma: no cover
            pass
