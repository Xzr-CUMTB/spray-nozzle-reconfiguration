# Spray nozzle reconfiguration

Research code accompanying **Two-nozzle reconfiguration for mine spray coverage recovery after conveyor displacement**.

The model tracks droplets through a prescribed airflow and equipment geometry, measures free-air voxel coverage, and searches for installation changes to two of six operating nozzles. The release contains the calculation code, fixed inputs, archived endpoint evaluations, and scripts for reproducing the numerical comparisons. Figure-generation code is excluded.

## Installation

The release was checked with Python 3.9.20, NumPy 1.26.4, SciPy 1.13.1, and Numba 0.60.0 on Windows. A CPU is sufficient; CUDA, PyTorch, and a commercial CFD solver are not required. Use Python 3.9–3.12 with the pinned dependencies.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
```

## Reproduce the reported comparisons

From the repository directory:

```bash
python reproduce.py verify
```

This checks the release checksums, geometry, installation constraints, optimizer update and checkpoint behavior, and the P0-based endpoint selection. It then reconstructs 13 result tables from the 665 archived P4 evaluations, including the paired bootstrap intervals and mechanical adjustment measures. It makes no new droplet evaluations. Output is written to a new directory under `runs/`.

Expected K2 coverage gains relative to K0 are:

| Scene | Conveyor translation (m) | Mean gain (percentage points) | Bootstrap 95% interval (percentage points) |
|---|---|---:|---:|
| V1 | (0, +0.50, 0) | 0.497759 | [0.415933, 0.577456] |
| V2 | (0, −0.50, 0) | 0.161688 | [0.109926, 0.213674] |
| V3 | (+0.50, 0, 0) | 0.452861 | [0.404082, 0.501093] |
| V4 | (−0.50, 0, 0) | 0.432436 | [0.369535, 0.488682] |

All four scenario means are below the predefined 0.50-percentage-point target. The overall decision is `SPARSE_RECONFIGURATION_NOT_CONFIRMED`.

## Check the trajectory calculations

```bash
python reproduce.py smoke
```

This runs the non-trajectory checks first, followed by 12 P0 evaluations and four P4 evaluations. It checks that cached and uncached calculations produce identical coverage masks in all four scenes and compares one baseline P4 calculation per scene with the archive. Numba compilation occurs on the first run.

Any archived endpoint can also be evaluated directly:

```bash
python reproduce.py evaluate --endpoint K0_V1 --fidelity P4 --seed 2026101401
python reproduce.py evaluate --endpoint K6_V1_ALL_O2026101201 --fidelity P4 --seed 2026101401
```

Endpoint identifiers and the corresponding 30 installation parameters are listed in `reference/layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json`.

## Repeat the complete search

```bash
python reproduce.py full --workers 4 --hours 24 --out runs/full-study
```

This performs all 120 pair-screening runs, 96 K2 searches, 32 K6 searches, and 665 terminal evaluations, then computes the result tables. It requests 22,976 P0 and 665 P4 evaluations. Runtime depends on CPU speed, process count, compilation, and cache reuse; allow several hours and sufficient disk space for individual evaluation records and component masks. The original experiment used eight worker processes. Worker count changes execution order, while the optimizer and physical seeds remain fixed.

The default wall-time allowance is 24 hours to accommodate different machines. `--hours 4` restores the original four-hour allowance. The original request limits are retained. A full rerun was not performed as part of release validation; the checks above cover archived-result reconstruction, optimizer equivalence on an analytic fixture, and selected trajectory evaluations.

For an orderly interruption between completed evaluations, resume with the same output directory:

```bash
python reproduce.py full --out runs/full-study --resume
```

The original start time and limits remain in effect on resume. A process killed during an evaluation leaves an unfinished ledger entry; the code stops on that entry instead of silently repeating it. Use a new output directory in that case.

## Files

| Path | Contents |
|---|---|
| `src/` | Event-based trajectory integration, voxel traversal, geometry, coverage evaluation, constrained particle-swarm search, and statistical analysis |
| `inputs/` | Spray parameters, numerical protocol, search settings, seeds, nominal geometry, displacement scenes, and baseline C001 layout |
| `reference/` | Archived terminal evaluations, selected layouts, screening and search summaries, and reported result tables |
| `tests/` | Geometry, optimization, checkpoint, statistical decision, and adjustment-metric checks |
| `reproduce.py` | Command-line entry point |
| `MANIFEST.json` | SHA-256 checksums of the distributed files |

See [METHODS.md](METHODS.md) for the metric, search settings, data fields, and scope of the release. All generated files stay in the selected output directory; the distributed inputs and reference tables are read-only during normal use.
