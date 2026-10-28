"""Search-space partition/budget logic and QbD-VLA descent on a mock model."""
import torch
from torch import nn

from qbdvla.components import ComponentMap
from qbdvla.search.greedy import ProbeCache, WeightVault, qbd_search
from qbdvla.search.space import Config, GroupSpec, SearchSpace


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.head_attn = nn.Linear(8, 8)
        self.head_mlp = nn.Linear(8, 16)
        self.head_out = nn.Linear(16, 4)


def make_space():
    m = Toy()
    cmap = ComponentMap(ve=[("vis", m.vis)],
                        ah=[("head_attn", m.head_attn), ("head_mlp", m.head_mlp),
                            ("head_out", m.head_out)])
    groups = [
        GroupSpec("ve", "ve", levels=("BASELINE", "W4A8", "W4A4")),
        GroupSpec("ah_attn", "ah", ("*attn*",), levels=("BASELINE", "W4A8", "W4A4")),
        GroupSpec("ah_rest", "ah", levels=("BASELINE", "W4A8", "W4A4")),
    ]
    return m, SearchSpace(cmap, groups)


def test_partition_and_budget():
    _, sp = make_space()
    assert len(sp.group_layers("ve")) == 2
    assert len(sp.group_layers("ah_attn")) == 1
    assert len(sp.group_layers("ah_rest")) == 2      # mlp + out (remainder)
    b0 = sp.budget(sp.initial())
    assert b0["w_bits_eff"] == 16 and b0["a_bits_eff"] == 16
    cfg = Config({"ve": "W4A8", "ah_attn": "BASELINE", "ah_rest": "BASELINE"})
    b = sp.budget(cfg)
    assert 4 < b["w_bits_eff"] < 16 and 8 < b["a_bits_eff"] < 16


def test_apply_and_vault_roundtrip():
    m, sp = make_space()
    w0 = m.head_attn.weight.detach().clone()
    vault = WeightVault(sp)
    applied = sp.apply(Config({"ve": "BASELINE", "ah_attn": "W4A4", "ah_rest": "BASELINE"}))
    assert applied == {"ah_attn": 1}
    assert not torch.equal(m.head_attn.weight, w0)          # quantized in place
    assert getattr(m.head_attn, "_qbd_act_hook_handle", None) is not None
    vault.restore()
    assert torch.equal(m.head_attn.weight, w0)              # restored exactly
    assert getattr(m.head_attn, "_qbd_act_hook_handle", None) is None
    # restore also clears the quantized-mark so re-apply is legal
    sp.apply(Config({"ve": "BASELINE", "ah_attn": "W4A4", "ah_rest": "BASELINE"}))
    vault.restore()


def _truth(cfg: Config) -> float:
    # non-additive trap: ve and ah_attn individually safe at W4A4, jointly fatal
    lv = cfg.levels
    if lv["ve"] == "W4A4" and lv["ah_attn"] == "W4A4":
        return 0.05
    drops = {"BASELINE": 0.0, "W4A8": 0.02, "W4A4": 0.05}
    return 1.0 - sum(drops[l] for l in lv.values())


def test_search_respects_interaction(tmp_path):
    _, sp = make_space()
    block = lambda cfg, i: round(_truth(cfg) * 10)   # deterministic 10-episode block
    res = qbd_search(sp, block, ProbeCache(tmp_path / "probes.jsonl"), tau=0.85,
                     block_size=10, max_blocks=3)
    final = res.path[-1]["levels"]
    assert not (final["ve"] == "W4A4" and final["ah_attn"] == "W4A4")   # avoided the trap
    assert final["ah_rest"] == "W4A4"                                    # took the free savings
    assert sum(1 for l in final.values() if l == "W4A4") == 2            # one of ve/attn also low
    assert res.path[-1]["score"] >= 0.85


def test_racing_spends_fewer_episodes_on_clear_calls(tmp_path):
    _, sp = make_space()
    calls = {"n": 0}
    def block(cfg, i):
        calls["n"] += 1
        t = _truth(cfg)
        return 10 if t > 0.5 else 0     # decisive outcomes -> early CI separation
    res = qbd_search(sp, block, ProbeCache(tmp_path / "p.jsonl"), tau=0.85,
                     block_size=10, max_blocks=3)
    # a clear config should be decided in <= 2 blocks, never all 3
    n_configs = len({r["key"] for r in
                     (__import__("json").loads(l) for l in (tmp_path / "p.jsonl").open())})
    assert calls["n"] < n_configs * 3


def test_probe_cache_resume(tmp_path):
    _, sp = make_space()
    calls = {"n": 0}
    def block(cfg, i):
        calls["n"] += 1
        return 10
    qbd_search(sp, block, ProbeCache(tmp_path / "p.jsonl"), tau=0.5, block_size=10, max_blocks=2)
    first = calls["n"]
    qbd_search(sp, block, ProbeCache(tmp_path / "p.jsonl"), tau=0.5, block_size=10, max_blocks=2)
    assert calls["n"] == first


def test_budget_exhaustion_stops_gracefully_and_is_recorded(tmp_path):
    """Running out of probe episodes is a stopping condition, not a crash: the descent so far is a
    valid budget-limited result. An unhandled raise here discards every episode already spent --
    on the SIMPLER run that would have been a day of compute with no result written."""
    _, space = make_space()

    def run_block(cfg, i):
        return 10                              # everything passes; the budget is the only limit

    res = qbd_search(space, run_block, ProbeCache(tmp_path / "p.jsonl"), tau=0.9, block_size=10,
               max_blocks=1, max_episodes=40)
    assert res.stopped_on_budget is True
    assert res.probes_episodes <= 40
    assert res.path and res.frontier            # a usable, budget-limited result is still returned


def test_search_that_finishes_on_the_space_is_not_flagged_as_budget_limited(tmp_path):
    _, space = make_space()
    res = qbd_search(space, lambda cfg, i: 10, ProbeCache(tmp_path / "p.jsonl"), tau=0.9,
               block_size=10, max_blocks=1, max_episodes=10_000)
    assert res.stopped_on_budget is False


def test_no_retry_never_reconsiders_a_rejected_move():
    """--no-retry drops a move permanently; the default retries it after the next accept."""
    from qbdvla.search.greedy import ProbeCache, qbd_search
    from qbdvla.search.space import Config, GroupSpec, SearchSpace

    class FakeSpace:
        """Two groups, one level down each. Lowering 'b' only becomes safe after 'a' has been lowered."""
        groups = [GroupSpec("a", "llm", None, ("BASELINE", "W4A4")), GroupSpec("b", "llm", None, ("BASELINE", "W4A4"))]

        def initial(self): return Config({"a": "BASELINE", "b": "BASELINE"})

        #: 'b' holds three quarters of the parameters, so lowering it saves the most bits and is raced first
        W = {"a": 1.0, "b": 3.0}

        def budget(self, cfg):
            bits = sum(self.W[k] * (16 if v == "BASELINE" else 4) for k, v in cfg.levels.items()) / 4
            return {"w_bits_eff": bits, "a_bits_eff": bits}

        def moves(self, cfg):
            out = []
            for g in self.groups:
                if cfg.levels[g.name] == "BASELINE":
                    nxt = Config({**cfg.levels, g.name: "W4A4"})
                    out.append((g.name, "W4A4", nxt))
            return out

        def apply(self, cfg): pass

    def run_block(cfg, i):
        lowered = {k for k, v in cfg.levels.items() if v != "BASELINE"}
        if lowered == {"b"}:      # b alone fails
            return 0
        return 10                 # anchor, a alone, and a+b all pass

    def run(retry, tmp):
        return qbd_search(FakeSpace(), run_block, ProbeCache(tmp), tau=0.9, block_size=10, max_blocks=1,
                    retry_rejected=retry)

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        with_retry = run(True, Path(d) / "a.jsonl")
        without = run(False, Path(d) / "b.jsonl")
    assert with_retry.path[-1]["levels"] == {"a": "W4A4", "b": "W4A4"}
    assert without.path[-1]["levels"] == {"a": "W4A4", "b": "BASELINE"}


def test_race_accepts_rejects_and_decides_at_budget():
    """Accept on the lower Wilson bound, reject on the upper one, point estimate after the last block."""
    import tempfile
    from pathlib import Path
    from qbdvla.search.greedy import ProbeCache, race
    from qbdvla.search.space import Config

    with tempfile.TemporaryDirectory() as d:
        v, rec = race(Config({"a": "W4A8"}), 0.75, lambda c, i: 10, ProbeCache(Path(d) / "x.jsonl"), block_size=10, max_blocks=3)
        assert v == "safe" and rec["n"] == 10                  # 10/10 clears the floor after one block
        v, rec = race(Config({"a": "W4A4"}), 0.75, lambda c, i: 2, ProbeCache(Path(d) / "y.jsonl"), block_size=10, max_blocks=3)
        assert v == "unsafe" and rec["n"] == 10                # 2/10 is below the floor after one block
        v, rec = race(Config({"a": "W4A6"}), 0.75, lambda c, i: 8, ProbeCache(Path(d) / "z.jsonl"), block_size=10, max_blocks=3)
        assert v == "safe" and rec["n"] == 30                  # 24/30 is undecided until the budget, then 0.8 >= 0.75


def test_select_point_is_noise_aware_and_cheapest():
    from qbdvla.search.greedy import select_point

    def pt(w, a, k, n):
        return {"levels": {}, "score": k / n, "n": n, "w_bits_eff": w, "a_bits_eff": a}
    # X-VLA-like: a perfect point followed by moves that lower the score beyond noise -> the perfect point
    r = {"path": [pt(16, 16, 27, 30), pt(6, 8, 10, 10), pt(4.2, 5.9, 10, 10), pt(4.1, 5.8, 29, 30), pt(4.0, 5.6, 27, 30)]}
    assert select_point(r)["index"] == 2
    # OFT-like: best 9/10 early, endpoint 26/30 -> indistinguishable -> the cheaper endpoint
    r = {"path": [pt(16, 16, 27, 30), pt(4.0, 5.95, 9, 10), pt(4.0, 5.94, 26, 30)]}
    assert select_point(r)["index"] == 2
    # noisy early high scores under a low anchor must not select the almost unquantized point
    r = {"path": [pt(16, 16, 25, 30), pt(10.4, 10.4, 17, 20), pt(4.3, 6.3, 24, 30), pt(4.2, 6.2, 19, 30)]}
    assert select_point(r)["index"] == 2
    # the anchor itself is never selected
    r = {"path": [pt(16, 16, 30, 30), pt(4, 4, 20, 30)]}
    assert select_point(r)["index"] == 1
