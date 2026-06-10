# RBF_ROM manifest

Verbatim copy of the L63 RBF-only-fitting thread out of `/mnt/scratch/users/sbrw610/CHORD2/`.
Created 2026-06-10. Relative layout under the repo root preserved so the tex
source's `\graphicspath{{../../}}` and every `results/LOR63/...` reference
inside the journal entries still resolves.

## Copied (by top-level directory)

- `CLAUDE.md` (1 file) — project instructions, so `RBF_ROM/` is self-explanatory.
- `pyproject.toml` (1 file) — declares the `pytest` slow marker and testpaths; same root-level metadata file.
- `.claude/` (6 files: `agents/*.md`, `settings.local.json`) — agent definitions are referenced by `CLAUDE.md`.
- `chord2/` (10 files: `backend.py`, `clustering.py`, `data.py`, `diagnostics.py`, `__init__.py`, `integrator.py`, `local_basis.py`, `sindy.py`, `baselines/__init__.py`, `baselines/galerkin.py`) — full library. `sindy.py` is the named code lever for the L63 RBF dictionary; the other modules are shared infrastructure that the L63 driver scripts import.
- `data/LOR63/` (3 files: `lor63.py`, `results/lor63/lor63.png`, `results/lor63/stats.npz`) — L63 dataset driver and pre-generated stats.
- `docs/journal/` (10 files) — every L63-named journal entry plus the two phase-0 framing entries (`2026-06-07-phase-0-rbf-design.md`, `2026-06-07-phase-0-scope.md`) that set the L63 negative-control role, and `2026-06-09-phase-0-backfit-architecture.md` because the backfit refactor it records also touched the L63 driver. The three entries named in the task spec (`2026-06-08-phase-0-l63-rbf-only-risk-surface.md`, `2026-06-08-l63-fix1-and-comparison.md`, `2026-06-09-l63-clustered-rbf-stages-A-B-E-F.md`) are included; so are the four other L63-titled milestones (`2026-06-07-phase-0-l63-robustness-shield.md`, `2026-06-08-l63-linrbf-K-sigma-ablation.md`, `2026-06-08-phase-0-l63-rbf-only-integration.md`, `2026-06-08-phase-0-l63-rbf-only.md`) which the named three repeatedly cross-reference as prior context.
- `docs/notes/` (17 files) — `rbf-only-fitting-on-l63.{md,tex,pdf,aux,log,out}` plus every topic note the L63 journals link out to: `clustered-rbf-on-l63.md`, `l63-rbf-bandwidth-rules.md`, `linrbf-identifiability.md`, `lyapunov-certificate-protocol.md`, `clustering-success-criterion.md`, `sindy-design.md`, `phase-0-design.md`, `forecast-vs-recovery.md`, `moment-orthogonalisation-shield.md`, `omitted-variable-bias-and-the-shield.md`, `backfit-architecture.md`. The last five mention L96 too but are explicitly cross-referenced from L63 entries as the load-bearing design context for the RBF dictionary, identifiability, and the moment-orthogonalisation shield used in the LinRBF thread.
- `scripts/` (7 files) — all `*l63*` named drivers and launchers: `run_phase0_l63.py`, `run_phase0_l63_rbf_only.py`, `run_phase0_l63_robustness.py`, `run_phase0_l63_backfit.py`, plus the three matching `launch_phase0_l63_*.sh`. The thread's per-step drivers live under `results/LOR63/codes/` (also copied) — the repo convention places stage-specific drivers there rather than in `scripts/`.
- `results/LOR63/` (235 files, ~27 MB) — the entire L63 results tree. Includes the named directories (`step1_nrbf_sweep`, `step1e_gamma_sweep`, `step1g_compare`, `step2a_cluster_rbf`, `step2d_diagnostics`, `step2e_ablate`, `step2f_globalsigma`) plus every other L63 step output (`step1f_compare`, `step1g_linrbf`, `step2a_lambda_1e-1`, `step2a_lambda_1e-2`, `step2b_stability`, `step2b_stability_F123`) and the prior phase-0 artefacts (`phase0_backfit`, `phase0_mix_poly_rbfs`, `phase0_rbf_only`). The `codes/` subdir contains the step1*/step2* drivers and launchers and the shared `_l63_rbf_lib.py` that the journals name as the workhorse library for the sweeps.
- `tests/` (0 files; empty directory created for layout parity) — see "Considered but left out" below.

`__pycache__/` directories were stripped on the way in; they are byte-compiled artefacts, not source.

## Considered but left out

- `tests/conftest.py`, `tests/test_data.py`, `tests/test_clustering.py`, `tests/test_diagnostics.py` — the `conftest.py` only exposes `lor96` and `ks_bursting` fixtures, and grepping the three test modules turned up zero references to `l63`/`lor63`/`sindy`. Per the task rubric ("if a test only uses lor96 it stays out, if it uses lor63 or sindy it comes in") they all stay out.
- `results/LOR96/`, `data/LOR96/`, all KS / KOL* / RB / EKG `results/` and `data/` subdirs — explicitly out of scope.
- L96-only journals: `2026-06-07-phase-0-l96-recovery-and-forecast.md`, `2026-06-08-phase-0-high-int-closure-diagnostic.md`, `2026-06-09-phase-0-l96-oracle-ridge-and-architecture-lock.md` — pure L96 / closure work.
- Notes that grepped clean of `l63`/`lor63` references: `kolmogorov-testbed-rationale.md`, `kuramoto-sivashinsky-testbed-rationale.md`, `rayleigh-benard-testbed-rationale.md`, `testbed-sweep-ranking.md`, `library-existence-test.md` (this last one is the L96 oracle-ridge writeup).
- Intro / sweep journals not L63-linked: `2026-06-06-architecture.md`, `2026-06-06-clustering-stage{1,2,3}.md`, `2026-06-07-clustering-stage4.md`, `2026-06-06-data-loader.md`, `2026-06-06-{kolmogorov,ks,rb}-intro-diagnostics.md`, `2026-06-06-scripts-and-tests.md`, `2026-06-06-testbed-sweep-ranking.md`.
- L96/KS/KOL/RB scripts: `run_phase0_l96*.py`, `launch_phase0_l96*.sh`, `make_phase0_l96_report*.py`, `plot_phase0_l96_*.py`, `run_phase0_l96_indicator.py`, `run_kmeans_sweep.py`, `launch_kmeans_sweep*.sh`, `kolmo_intro.py`, `ks_intro.py`, `rb_intro.py`, `run_inspect.py`, `smoke_kmeans.py` — none in the L63 thread.
- `launchFiles/launch_build_poly_library_gpu.sh` — unused by the L63 thread; the L63 RBF fits are CPU-bound and have their own launchers under `results/LOR63/codes/`.
- `.benchmarks/`, `.pytest_cache/` — runtime caches, not source.
- `results/intro/`, `results/LOR96_high_int/` — out of scope.

## Files referenced by the L63 thread that I could not find or could not place

- `R-l63_step1d_risk-45510.out`, SLURM logs `R-...-45638.out`, `R-...-45641.out`, `R-...-46293.out`, `R-...-46349.out` — referenced by `2026-06-08-phase-0-l63-rbf-only-risk-surface.md` and `2026-06-09-l63-clustered-rbf-stages-A-B-E-F.md` as the SLURM stdout for the runs that produced the headline numbers. They are not under `results/LOR63/` and do not appear anywhere under `/mnt/scratch/users/sbrw610/CHORD2/` (the user-side scratch SLURM dump is elsewhere). Not copied because they are not in the source tree.
- `docs/research_plan_quantized_sindy_rbf.pdf` — cited at the top of `2026-06-07-phase-0-scope.md`. Not present under `docs/`. Not copied.

Everything else the thread's notes and journals point at landed at the path the source uses.
