# CHORD2

Quantized local reduced-order modelling with **local SINDy** dynamics.

This is a from-scratch adaptation of Colanera & Magri, *Quantized local reduced-order modeling in time*, Comput. Methods Appl. Mech. Engrg. 447 (2025) 118393 (PDF at `docs/1-s2.0-S0045782525006656-main (1).pdf`). The clustering, local-PCA, and cluster-switching machinery of the paper carry over unchanged; the per-cluster intrusive POD-Galerkin dynamics is replaced by a non-intrusive sparse regression `ȧ^k = Θ(a^k)·ξ^k` on the reduced coordinates.

The predecessor library "CHORD" implemented the POD-Galerkin variant. CHORD2 is the SINDy variant.

## Pipeline (mirrors Fig. 1 of the paper)

1. **Data** — load snapshots from `data/<problem>/stats.npz`.
2. **Quantization** — K-means in the full state space; K selected by BIC.
3. **Local basis** — per-cluster PCA centred on the cluster centroid; keep `r_k` modes; precompute change-of-coordinates `U_j^H U_i`.
4. **Dynamics** — fit `ȧ^k = Θ(a^k)·ξ^k` per cluster via STLSQ (from-scratch).
5. **Prediction** — integrate the local ODE; on cluster change, apply the change-of-coordinates and switch to the new model (Algorithm 1).

## Repo layout

- `chord2/` — library code. One module per pipeline stage.
  - `data.py`, `clustering.py`, `local_basis.py`, `sindy.py`, `integrator.py`, `diagnostics.py`
  - `baselines/galerkin.py` — one-shot reference g-ROM and ql-ROM (POD-Galerkin) for comparison curves.
- `data/<problem>/` — dataset drivers and pre-generated `stats.npz` files. Owned by the user, not by agents.
- `docs/journal/` — dated journal of decisions and milestones. Owned by `scribe`.
- `docs/notes/` — living topic writeups (SINDy design, BIC for K, derivative estimator, etc.). Owned by `scribe`.
- `.claude/agents/` — agent definitions.

## Agents

| Agent | Role | Tools |
|---|---|---|
| `software-engineer` | Implements pipeline modules and dataset drivers | Read/Edit/Write/Bash/Agent |
| `code-architect` | Repo structure, naming, dependency hygiene, one-shot baselines | Read/Edit/Write/Bash/Agent |
| `rom-specialist` | Read-only review of ROM theory | Read/Grep/WebFetch |
| `fluid-dynamics-specialist` | Read-only review of physics | Read/Grep/WebFetch |
| `scribe` | Writes `docs/journal/` and `docs/notes/` | Read/Edit/Write |

### Proactive consultant calls (the orchestrator's responsibility)

Before *any* non-trivial implementation or change touching the areas below, spawn the matching consultant via `Agent({subagent_type: ...})` and incorporate their assessment. The triggers are not optional; if you skip them, record why.

- **`rom-specialist`** — triggered by changes to:
  - SINDy feature library (polynomial degree, Fourier features, cross-terms)
  - Sparsity selection (STLSQ threshold, SR3, LASSO) — global or per-cluster
  - BIC for `K` (Eqs. 17-18) or selection of `r_k`
  - PCA / SVD truncation logic in `local_basis.py`
  - Change-of-coordinates `U_j^H U_i` and centroid shift `U_j^H (c_i - c_j)` (Eq. 14)
  - Derivative estimator for `ȧ^k`
  - Cluster-switching dynamics — affiliation function, boundary smoothness, transient behaviour on switches
- **`fluid-dynamics-specialist`** — triggered by changes whose correctness rests on physics:
  - Regime classification of any testcase (KS bursting vs chaotic, Kolmogorov quasiperiodic vs turbulent)
  - Energy spectra, PDFs, autocorrelations — both the diagnostic and its interpretation
  - Conservation laws and symmetries the ROM should preserve (incompressibility for Kolmogorov, zero-mean for KS, etc.)
  - Statistical-stationarity tests
  - Attractor visualisations (Fourier-mode projections, MDS)

### Scribe

Called *after* a non-trivial event by `software-engineer` or `code-architect`:
- Implementation merged
- Refactor merged
- Consultant assessment that materially changed a decision

Brief the scribe with what happened, what was decided, what was rejected, what the consultants said. Do not call for trivial edits.

## House style

Reference implementation: `data/KS/ks_solver.py`. New code mirrors it.

- numpy-first. Default deps: numpy, scipy, matplotlib. **No** torch / jax / pysindy / tensorflow. CuPy via `get_backend(device)` shim only.
- Single-file CLI drivers with `argparse` + `pathlib.Path`.
- Docstrings cite the paper section / equation / figure they implement.
- Standardised `stats.npz` schema: `u, t, nx, dx, L, dt_sim, dtStats, tInit, bc_type` (+ problem-specific). Do not break it.
- Plotting: matplotlib, LaTeX labels, results under `results/<problem>/`.
- Solvers and fitters ship `--diagnose` modes that split the record in halves.
- Comments only when *why* is non-obvious. Identifiers carry the *what*.

## Pipelines and tests

Two top-level folders host code that *uses* `chord2/` rather than living inside it: `scripts/` for runnable pipeline drivers and `tests/` for the automated test suite. Both sit at the repo root, parallel to `chord2/`, so neither leaks into the library namespace.

### `scripts/`

End-to-end pipeline drivers and per-stage helpers. Each script is a thin CLI wrapper around `chord2/` functions; no scientific logic lives here.

| Concern | Convention |
|---|---|
| Location | Repo root `scripts/`. Not under `chord2/` — these are not importable library code, and keeping them out of the package avoids accidentally adding CLI deps to the import path. |
| Default driver | `scripts/run_pipeline.py --dataset NAME` runs the full data→cluster→basis→sindy→integrate→diagnose chain. Per-stage drivers exist *only when a stage needs to be re-run in isolation* (typically `run_sindy_fit.py` while tuning sparsity). |
| Naming | `run_<verb>_<noun>.py`. Examples: `run_pipeline.py`, `run_sindy_fit.py`, `run_diagnostics.py`. Verbs are imperative; nouns are pipeline stages from CLAUDE.md §Pipeline. |
| Dataset selection | `--dataset NAME` argparse, where `NAME` is a `chord2.data.REGISTRY` key. `--dataset all` iterates the registry. No yaml/json config until a single CLI line stops being enough — this matches the `ks_solver.py` style anchor. |
| Outputs | `results/<NAME>/` via `chord2.data.results_dir(NAME)`. Per-stage artefacts use stable filenames (`clusters.npz`, `local_bases.npz`, `sindy_models.npz`, `prediction.npz`, `diagnostics.png`) so a re-run of one stage drops in next to the others. |
| Style | Same as `data/KS/ks_solver.py`: `argparse` + `pathlib.Path`, `--diagnose` where it makes sense, paper-cited docstrings, no torch/jax/pysindy/sklearn. |
| Shell launchers | Dataset-specific launchers like `data/KS/launch_ks.sh` stay co-located with their dataset. `scripts/` is for ROM pipeline drivers, not FOM solver wrappers. |

### `tests/`

Interactive test suite. Fast enough that the user can run it on a whim between edits.

| Concern | Convention |
|---|---|
| Framework | `pytest`. Justified: fixtures, parametrize, and `-k` selection are worth the one dependency; the stdlib `unittest` boilerplate is hostile to the numpy-first idiom. Add `pytest` to dev deps; no plugins. |
| Layout | Mirror `chord2/`: `tests/test_data.py`, `tests/test_clustering.py`, `tests/test_local_basis.py`, `tests/test_sindy.py`, `tests/test_integrator.py`, `tests/test_diagnostics.py`. One module-under-test per file. |
| Unit vs integration | Unit tests live in the mirrored files above. End-to-end pipeline tests live in `tests/test_pipeline.py` and exercise one small dataset through `scripts/run_pipeline.py` (or the library equivalent), asserting `stats.npz`-schema artefacts land in a `tmp_path` results dir. |
| Canonical small testbeds | `LOR96` (10-dim single-scale, cheap) for any test that needs real dynamics; `KS_bursting` with `stride=10` for any test that needs a PDE. No KOL/RB/EKG in the default suite — they belong behind a marker. |
| Fixtures | A single `tests/conftest.py` exposes `lor96` and `ks_bursting` fixtures wrapping `chord2.data.load(..., stride=...)` at session scope, so the load cost is paid once. |
| Runtime budget | Whole suite under 30 s on CPU, no GPU. Tests that exceed ~1 s individually are marked `@pytest.mark.slow` and skipped by default; opt in with `pytest -m slow`. |
| Config | A single `pyproject.toml [tool.pytest.ini_options]` at the repo root registers the `slow` marker and sets `testpaths = ["tests"]`. No separate `pytest.ini`. |
| Style | numpy assertions via `np.testing.assert_allclose`. Tolerances live next to the assertion with a one-line comment on what sets them. No mocking of library code; tests run against the real implementation on small inputs. |

## Do not

- Do not add error handling for impossible scenarios. Trust internal contracts.
- Do not add backward-compatibility shims. The repo has no users.
- Do not introduce dependencies silently. Justify in a journal entry.
- Do not write speculative documentation. The journal records what happened; notes record current understanding.
- Do not reintroduce dependence on the FOM operator `N(u)` anywhere downstream of snapshots — the SINDy variant is non-intrusive by design.
- Do not skip the proactive consult
