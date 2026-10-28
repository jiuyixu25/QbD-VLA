"""Allocator + reconstruction proxy on the toy search space (no GPU)."""
from torch import nn

from qbdvla.components import ComponentMap
from qbdvla.search.baselines import allocate, budget_sum, proxy_recon_error
from qbdvla.search.space import GroupSpec, SearchSpace


def make_space():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.vis = nn.Linear(8, 8)
            self.head_attn = nn.Linear(8, 8)
            self.head_mlp = nn.Linear(8, 16)
    m = Toy()
    cmap = ComponentMap(ve=[("vis", m.vis)], ah=[("head_attn", m.head_attn), ("head_mlp", m.head_mlp)])
    groups = [GroupSpec("ve", "ve", levels=("BASELINE", "W4A8", "W4A4")),
              GroupSpec("ah_attn", "ah", ("*attn*",), levels=("BASELINE", "W4A8", "W4A4")),
              GroupSpec("ah_rest", "ah", levels=("BASELINE", "W4A8", "W4A4"))]
    return SearchSpace(cmap, groups)


def test_allocate_reaches_target_and_prefers_low_sensitivity():
    sp = make_space()
    # ah_attn is "sensitive": high cost at every level; others cheap
    sens = {}
    for g in sp.groups:
        for lvl in g.levels[1:]:
            sens[(g.name, lvl)] = 10.0 if g.name == "ah_attn" else 1.0
    full = budget_sum(sp, sp.initial())            # 32 bits
    cfg = allocate(sp, sens, target_sum=full - 8)   # need one-quarter of the way down
    assert budget_sum(sp, cfg) <= full - 8 + 1e-9
    assert cfg.levels["ah_attn"] == "BASELINE"      # the sensitive group was spared
    assert any(cfg.levels[g] != "BASELINE" for g in ("ve", "ah_rest"))


def test_allocate_exhausts_when_target_unreachable():
    sp = make_space()
    sens = {(g.name, l): 1.0 for g in sp.groups for l in g.levels[1:]}
    cfg = allocate(sp, sens, target_sum=0.0)
    assert all(v == "W4A4" for v in cfg.levels.values())   # descended everything


def test_recon_proxy_is_monotone_in_bits():
    sp = make_space()
    sens = proxy_recon_error(sp)
    for g in sp.groups:
        assert sens[(g.name, "W4A4")] > sens[(g.name, "W4A8")]   # fewer act bits -> more error
