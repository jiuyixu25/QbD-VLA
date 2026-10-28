# Records

Every evaluation cell and allocation run behind the simulation results of the paper. `tables/make_tables.py`
regenerates Tables 1, 2, 3, 4 and 7, the group-interaction numbers quoted in the text, and a per-cell summary from these
files.

* `results/<benchmark>/<suite>/<model>/<tag>.jsonl` — one evaluation cell per file: a header record (model,
  checkpoint, preset, group-level configuration under `extra.qbd_config`, nominal bits under `extra.budget`) followed
  by one record per episode (task, initial state, success, steps, latency). LIBERO cells have 200 episodes (10 tasks x
  initial states 0 to 19) unless the tag says otherwise; SIMPLER visual-matching cells (`google_robot_vm`) have 864.
* `search/<run>/result.json` — one allocation run: settings, layer groups with parameter counts, every accepted
  configuration with its short-batch estimate, and the selected configuration (`selected`, the rule of Section 3.4,
  `select_point` in `qbdvla/search/greedy.py`; for runs recorded before the rule was part of the output it was computed
  from the same path). `probes.jsonl` (the four main LIBERO-Spatial runs and the two SIMPLER runs) holds every
  short-batch evaluation.
* The real-robot trials of Section 4.5 (Tables 5 and 6) are not included as records; the results video attached to the
  [GitHub release](https://github.com/jiuyixu25/QbD-VLA/releases) shows them.

## Tags

| tag | cell |
|---|---|
| `rtn-BASELINE-e2e` | full precision |
| `rtn-<preset>-<scope>` | uniform round-to-nearest at that preset; scope `e2e` quantizes every component, `ve` / `llm` / `ah` one component only |
| `qbd-deep-final`, `qbd-final`, `qbd-select-final` | the configuration QbD-VLA selects on LIBERO-Spatial (Table 1) |
| `qbd-transfer-*` | the same configuration applied unchanged to Object, Goal and Long |
| `qbd-deep-final-seed1` | the selected configuration under evaluation seed 1 |
| `qbd-deep`, `qbd-mid` (X-VLA) | the endpoint (index 20) and index 18 of the accepted X-VLA path; the selected configuration is index 16 |
| `qbd-direct-final` | configuration selected by an allocation run on LIBERO-Goal |
| `qbd-tau50-final`, `qbd-tau95-final`, `qbd-k1-final`, `qbd-h60-final`, `qbd-noretry*`, `qbd-coarse-final`, `qbd-calib-final`, `qbd-knee-*` | protocol ablations on LIBERO-Spatial |
| `qbd-final`, `qbd-t95-final` under `results/simpler/` | the SIMPLER allocations: 24 tasks at tau = 0.90 and 40 tasks at tau = 0.95 (Table 4) |
| `path-*-NN` | the NN-th accepted configuration of a run, re-evaluated on initial states 0 to 9 (100 episodes) |
| `ext30-49-*` | held-out initial states 30 to 49 |
| `e6-*` | pi0 on LIBERO-Goal, full precision and the selected configuration, initial states 0 to 9 (100 episodes) |
| `e3-*`, `e3sel-*` (runs `search/e3-*`, `search/e3sel-*`) | sensitivity-based allocations (action error, activation kurtosis, reconstruction error) at the cost of the endpoint / selected configuration |
| `ufr-final`, `ufr2-final`, `ufr-relax-mlp-only` | the two-stage allocator (UFR) and the single-group relaxation used in the factorial comparison |
| `ablate-W4A4-ah-*` | single action-head groups at W4A4 (the Section 1 example) |
| `actquant-*`, `omegaqvla-*` | ActQuant and HoloQ-VLA in their own runtimes (`-fp16`: their full-precision reference) |
| `calib*-W4A6-e2e` | uniform W4A6 with activation calibration |

User names, host names, addresses and commit hashes in the records are replaced.
