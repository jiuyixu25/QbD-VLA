"""Configuration space of the QbD-VLA bit allocation (Section 3.1 of the paper).

A model is partitioned into semantic **groups** — VLAQuantBench's culprit
patterns refined below component granularity. A **configuration** assigns each
group one **level** from a small menu (preset-style strings, highest precision
first). Applying a configuration quantizes each non-baseline group with its
level's spec, restricted to the group's layers; groups are disjoint by
construction (validated at build time), so the whole assignment is one pass of
independent `quantize_modules` calls over disjoint layer sets.

The budget of a configuration is its parameter-weighted average weight bits
and activation bits over all *quantizable* parameters (unquantized groups
count at 16). Activation traffic is proportional to layer input width × calls;
parameter count is the standard proxy and what we report.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Sequence

from ..components import ComponentMap
from ..quant.apply import DEFAULT_EXCLUDE_PATTERNS, iter_quantizable, quantize_modules
from ..quant.presets import parse_spec

log = logging.getLogger(__name__)

__all__ = ["coarse_groups", "GroupSpec", "SearchSpace", "Config", "MODEL_GROUPS", "DEFAULT_LEVELS"]

#: Highest precision first. "BASELINE" = leave the group unquantized.
DEFAULT_LEVELS: tuple[str, ...] = ("BASELINE", "W8A8", "W4A8", "W4A6", "W4A4")


@dataclass(frozen=True)
class GroupSpec:
    name: str
    component: str                      # ComponentMap component the group lives in
    only: tuple[str, ...] = ()          # patterns within the component; () = the rest
    levels: tuple[str, ...] = DEFAULT_LEVELS


@dataclass
class Config:
    """Assignment: group name -> level string (index into the group's menu)."""

    levels: dict[str, str]

    def key(self) -> str:
        return "|".join(f"{g}={l}" for g, l in sorted(self.levels.items()))

    def pretty(self) -> str:
        return ", ".join(f"{g}:{l}" for g, l in sorted(self.levels.items()))


class SearchSpace:
    def __init__(self, cmap: ComponentMap, groups: Sequence[GroupSpec]):
        self.cmap = cmap
        self.groups = list(groups)
        self._layers: dict[str, list] = {}   # group -> [(qual, module)]
        self._params: dict[str, int] = {}
        #: optional per-layer activation calibration (SmoothQuant-style smoothing/clip vectors built once from
        #: the un-quantized policy); when set, every level of the lattice is a *calibrated* RTN level
        self.act_calib: dict | None = None
        self._build()

    # ------------------------------------------------------------------ #
    def _component_layers(self, component: str):
        exclude = list(DEFAULT_EXCLUDE_PATTERNS) + list(self.cmap.exclude.get(component, []))
        allow = list(self.cmap.allow.get(component, []))
        return list(iter_quantizable(self.cmap.roots(component),
                                     exclude_patterns=exclude, allow_patterns=allow))

    def _build(self) -> None:
        claimed: dict[int, str] = {}
        by_comp: dict[str, list] = {}
        for g in self.groups:
            by_comp.setdefault(g.component, None)
        for comp in by_comp:
            by_comp[comp] = self._component_layers(comp)
        # patterned groups claim first; pattern-less groups take the remainder
        for g in sorted(self.groups, key=lambda g: (not g.only,)):
            pool = by_comp[g.component]
            mine = []
            for qual, mod in pool:
                if id(mod) in claimed:
                    continue
                if g.only and not any(fnmatch.fnmatchcase(qual, p) for p in g.only):
                    continue
                mine.append((qual, mod))
                claimed[id(mod)] = g.name
            if not mine:
                raise ValueError(f"group {g.name!r}: no layers matched {g.only} in {g.component!r}")
            self._layers[g.name] = mine
            self._params[g.name] = sum(m.weight.numel() for _, m in mine)
        n_total = sum(len(v) for v in by_comp.values())
        n_claimed = len(claimed)
        if n_claimed != n_total:
            missing = [q for comp in by_comp for q, m in by_comp[comp] if id(m) not in claimed]
            raise ValueError(f"{n_total - n_claimed} layers unclaimed by any group, e.g. {missing[:4]}")
        log.info("search space: %d groups / %d layers / %.1fM params: %s",
                 len(self._layers), n_claimed, sum(self._params.values()) / 1e6,
                 {g: len(v) for g, v in self._layers.items()})

    # ------------------------------------------------------------------ #
    def initial(self) -> Config:
        return Config({g.name: g.levels[0] for g in self.groups})

    def moves(self, cfg: Config):
        """All single-group one-level-down candidates."""
        out = []
        for g in self.groups:
            i = g.levels.index(cfg.levels[g.name])
            if i + 1 < len(g.levels):
                nxt = dict(cfg.levels)
                nxt[g.name] = g.levels[i + 1]
                out.append((g.name, g.levels[i + 1], Config(nxt)))
        return out

    def group_layers(self, name: str):
        return self._layers[name]

    def group_params(self, name: str) -> int:
        return self._params[name]

    # ------------------------------------------------------------------ #
    def budget(self, cfg: Config) -> dict[str, float]:
        tot = sum(self._params.values())
        wbits = abits = 0.0
        for g in self.groups:
            spec = parse_spec(cfg.levels[g.name])
            w = 16 if spec.weight is None else spec.weight.bits
            a = 16 if spec.act is None else spec.act.bits
            wbits += self._params[g.name] * w
            abits += self._params[g.name] * a
        return {"w_bits_eff": wbits / tot, "a_bits_eff": abits / tot}

    def apply(self, cfg: Config) -> dict[str, int]:
        """Quantize the model in place per ``cfg``. Caller restores weights between configs."""
        applied = {}
        for g in self.groups:
            level = cfg.levels[g.name]
            spec = parse_spec(level)
            if spec.is_baseline:
                continue
            names = [q for q, _ in self._layers[g.name]]
            rep = quantize_modules(self.cmap.roots(g.component), spec,
                                   only_patterns=names, exclude_patterns=(), allow_patterns=(),
                                   act_calib=self.act_calib)
            if rep.n_layers != len(names):
                raise RuntimeError(f"group {g.name}: applied {rep.n_layers} != expected {len(names)}")
            applied[g.name] = rep.n_layers
        return applied


#: Group definitions per model, following VLAQuantBench's culprit structure.
MODEL_GROUPS: dict[str, list[GroupSpec]] = {
    "pi05": [
        GroupSpec("ve", "ve"),
        GroupSpec("llm", "llm"),
        GroupSpec("ah_attn", "ah", ("*self_attn*",)),
        GroupSpec("ah_adarms_in", "ah", ("*input_layernorm.dense*",)),
        GroupSpec("ah_adarms_post", "ah", ("*post_attention_layernorm.dense*", "*model.norm.dense*")),
        GroupSpec("ah_mlp", "ah", ("*.mlp.*",)),
        GroupSpec("ah_plumbing", "ah"),  # remainder: action/time projections
    ],
    "pi0": [
        GroupSpec("ve", "ve"),
        GroupSpec("llm", "llm"),
        GroupSpec("ah_attn", "ah", ("*self_attn*",)),
        GroupSpec("ah_mlp", "ah", ("*.mlp.*",)),
        GroupSpec("ah_plumbing", "ah"),
    ],
    "openvla_oft": [
        GroupSpec("ve", "ve"),
        GroupSpec("mp", "mp"),
        GroupSpec("llm", "llm"),
        GroupSpec("ah_fc1", "ah", ("*.fc1",)),
        GroupSpec("ah_blocks", "ah", ("*mlp_resnet_blocks*",)),
        GroupSpec("ah_fc2", "ah", ("*.fc2",)),
    ],
    "xvla": [
        GroupSpec("ve", "ve"),
        GroupSpec("llm", "llm"),
        GroupSpec("ah_qkv", "ah", ("*attn.qkv*",)),
        GroupSpec("ah_proj", "ah", ("*attn.proj*",)),
        GroupSpec("ah_fc1", "ah", ("*mlp.fc1*",)),
        GroupSpec("ah_fc2", "ah", ("*mlp.fc2*",)),
        GroupSpec("ah_rest", "ah"),
    ],
}


def coarse_groups(model: str) -> list[GroupSpec]:
    """Component-granularity ablation: one group per pipeline component, same level menu.
    Derived from the fine table so the component set is identical."""
    comps = list(dict.fromkeys(g.component for g in MODEL_GROUPS[model]))
    return [GroupSpec(c, c) for c in comps]
