# Computational methods and data

## Model and metric

The roadway domain is `[0, 6] × [−2.65, 2.65] × [0, 3.5] m`; the target is `[0.5, 5.5] × [−2.4, 2.4] × [0.25, 3.25] m`. Twelve axis-aligned solids represent the conveyor. Each scene translates the complete equipment geometry by 0.50 m in one of four horizontal directions. Exact subtraction of the solid union gives 71.47536 m³ of target free-air volume in each equipment scene; the empty reference has 72 m³.

Coverage is the sum of free-target volumes in voxels traversed by a droplet trajectory, divided by the total target free-air volume. Multiple visits to a voxel do not increase its contribution. Partial boundary voxels and solid intersections are accounted for. Coverage is stored as a fraction; differences are multiplied by 100 to obtain percentage points.

All six nozzles remain active. The shared spray inputs are a 60° cone, 24.3 m/s initial velocity, 2.93 L/min flow rate, and the volume-weighted lognormal diameter distribution in `inputs/configs/NUMERICAL_PROTOCOL_FROZEN_COPY.json`. The prescribed airflow is 1 m/s along +x. Droplets are subject to drag and gravity and terminate at equipment interception or domain exit. The integration horizon is 12 s, with a maximum step of 0.002 s and adaptive stability factor 0.2. Evaluations with remaining active volume fraction at or above 0.005 are rejected as incomplete.

| Level | Voxel edge | Parcels per nozzle | Role |
|---|---:|---:|---|
| P0 | 0.10 m | 200 | Search and pair selection |
| P4 | 0.04 m | 800 | Independent terminal evaluation |

This is a reduced-order trajectory and geometric-coverage calculation under prescribed airflow. It does not solve the airflow field or predict dust-removal efficiency. The release supports the four-scene reconfiguration study. It does not include the earlier baseline-design search, the numerical-protocol selection study, CFD preparation files, or field-monitoring records. Their selected inputs are supplied here as fixed starting conditions.

## Installation variables and search

Each nozzle has five variables: mounting-surface code, normalized axial coordinate, normalized transverse/height coordinate, yaw, and pitch. The six five-variable blocks form a 30-element particle. Surface codes 0, 1, and 2 correspond to left wall, right wall, and roof. The decoder applies the wall-dependent yaw offset before constructing the physical spray axis.

K0 retains C001. K2 permits changes in one selected pair and preserves the other four five-variable blocks exactly. For the selected pair, normalized coordinates may change by at most 0.15, yaw by 30°, and pitch by 25°, intersected with the full bounds. Mounting-surface changes are allowed subject to feasibility. K6 permits all six nozzles to change within the full bounds. Minimum nozzle spacing is 0.30 m; equipment clearance and forbidden mounting regions are checked for every proposed layout.

The swarm uses inertia decreasing from 0.90 to 0.40, personal/global coefficients 1.55/1.75, and velocity limits of 20% of the permitted interval. Surface categories use the discrete update in `src/search.py`. Infeasible repaired proposals revert to the preceding feasible particle. The original C001 particle is included in every initial population.

| Stage | Runs | Particles | Updates after initialization | P0 evaluations |
|---|---:|---:|---:|---:|
| Pair screening | 4 scenes × 15 pairs × 2 seeds | 8 | 1 | 1,920 |
| K2 formal search | 4 scenes × 3 pairs × 8 seeds | 8 | 16 | 13,056 |
| K6 formal search | 4 scenes × 8 seeds | 10 | 24 | 8,000 |

Screening ranks pairs by the median of two P0 best scores, then the maximum best score, then pair ID. The three leading pairs are retained. For each scene and formal optimizer seed, the primary K2 endpoint is the highest P0 endpoint among those three pairs, with exact ties resolved by pair ID. P4 scores are not used to reselect primary endpoints.

## Replication and statistical analysis

The terminal set contains four K0 layouts, 96 K2 endpoints, 32 K6 endpoints, and one empty-roadway C001 reference. Each is evaluated with five shared physical seeds, giving 665 records. Search physical seeds, terminal physical seeds, optimizer seeds, and the bootstrap seed are listed in `inputs/configs/SEEDS_FROZEN.json`.

The main statistical unit is the optimizer seed. For each of eight optimizer seeds in a scene, coverage differences are first averaged over the five paired physical seeds. The percentile bootstrap resamples these eight optimizer units 10,000 times; the 2.5% and 97.5% quantiles use NumPy's linear method. The five physical samples are not treated as five additional independent optimizer runs.

The predefined support rule requires a mean gain of at least 0.50 percentage points, at least six positive optimizer-unit gains, at least six layouts with gains positive in four of five physical samples, and a strictly positive lower bootstrap limit. The protocol also defines a conjunctive fidelity diagnostic. The complete decision rules are retained in `inputs/configs/STEP5A_PROTOCOL.json` and `src/analysis.py`.

Mechanical adjustment is reported as the sum of nozzle displacement distances, the sum of angles between the before/after physical spray axes, and the number of mounting-surface changes. Axis rotation is not computed by subtracting yaw angles alone.

## Reference data

`P4_terminal_all.csv` contains 660 equipment-scene records. `empty_C001_reference.csv` contains five empty-scene records. Key fields are:

| Field | Meaning |
|---|---|
| `endpoint_id` | Layout identity shared with the endpoint JSON manifest |
| `scene`, `method`, `pair_id` | Geometry, K0/K2/K6 case, and adjustable nozzle pair |
| `optimizer_seed` | Search replicate; empty for K0 and the empty reference |
| `physical_seed` | Shared droplet-sampling replicate |
| `coverage` | Free-air voxel coverage fraction |
| `visited_volume_m3`, `free_target_volume_m3` | Numerator and denominator of coverage |
| `collision_fraction`, `active_fraction` | Represented liquid-volume fractions at termination |
| `primary` | Whether an endpoint belongs to the main comparison |
| `status` | Trajectory completion classification |
| `direct_id`, `label` | Recorded evaluation identifiers |
| `wall_s` | Historical evaluation runtime, including the original execution conditions |
| `baseline_component_hits` | Nozzle components retrieved from the exact cache |
| `new_nozzle_trajectory_calls` | Nozzle components integrated for that evaluation |

`primary_effects_by_search_seed.csv` contains the 32 optimizer-unit results. `primary_effects.csv` contains the four scenario summaries. `remaining_gap.csv` stores K6−K2 differences. `retrofit_magnitude.csv` and `retrofit_per_nozzle.csv` contain layout and individual-nozzle adjustments. `subset_screening.csv`, `K2_formal_runs.csv`, and `K6_formal_runs.csv` retain search summaries. The remaining tables contain endpoint summaries, paired comparisons, obstruction effects, selection frequencies, fidelity diagnostics, and secondary directional or sampling summaries.

The endpoint manifests retain the recorded layout vectors, primary flags, P0 scores, and historical identifiers. Hash fields inside the original input/reference records describe the historical experiment files. `MANIFEST.json` supplies the checksums for the distributed release.

## Portable execution

The event integrator, DDA traversal, spray sampling, exact volume calculations, search updates, and statistical operations retain the research implementation. Release changes concern output-directory selection, input copying, runtime initialization, a platform-specific file lock, and command-line orchestration. Local-machine path metadata and a pointer to a redundant external diameter-fit file were removed; the full fitted distribution remains inline. Historical scheduling and preparation commands are excluded.

Exact baseline-nozzle cache entries are keyed by scene checksum, fidelity, physical seed, and float64 nozzle parameters. Their masks are combined by Boolean union. No surrogate approximation is used for cached components. Generated caches and ledger files are local run products.

Validation on the release environment includes reconstruction of all 13 analysis tables, equality of bootstrap indices, all 133 endpoint feasibility checks, an analytic comparison with the original swarm update, checkpoint continuation, exact cached/uncached P0 equivalence in four scenes, and baseline P4 comparisons in four scenes. The full optimization campaign is available through `reproduce.py full` but was not rerun during packaging. Unix file locking is implemented; Linux and macOS execution have not been tested in this release.
