"""Run result checks, trajectory checks, or the complete reconfiguration study."""
import argparse
import contextlib
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parent
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[variable] = "1"
sys.dont_write_bytecode = True


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_release():
    manifest = json_read(REPO / "MANIFEST.json")
    for relative, expected in manifest["sha256"].items():
        path = REPO / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Release checksum mismatch: " + relative)
    return len(manifest["sha256"])


def prepare(args):
    suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    root = Path(args.out).resolve() if args.out else REPO / "runs" / (args.mode + "-" + suffix)
    if root == REPO or REPO in root.parents and root.relative_to(REPO).parts[0] != "runs":
        raise ValueError("Choose an output directory under runs/ or outside the repository.")
    if root.exists():
        if not args.resume or args.mode != "full":
            raise FileExistsError("Output already exists. Use a new directory, or full --resume.")
        marker = json_read(root / "run.json")
        if marker["mode"] != "full":
            raise ValueError("Only a full run can be resumed.")
        if marker["release_manifest_sha256"] != hashlib.sha256((REPO / "MANIFEST.json").read_bytes()).hexdigest():
            raise ValueError("Resume requires the same release files.")
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a missing output directory.")
        root.mkdir(parents=True)
        shutil.copytree(REPO / "inputs", root, dirs_exist_ok=True)
        for folder in ["logs", "results", "checkpoints", "component_cache"]:
            (root / folder).mkdir(exist_ok=True)
        if args.mode in ["verify", "smoke", "evaluate"]:
            shutil.copytree(REPO / "reference" / "layouts", root / "layouts", dirs_exist_ok=True)
        if args.mode == "verify":
            for name in ["P4_terminal_all.csv", "empty_C001_reference.csv"]:
                shutil.copyfile(REPO / "reference" / name, root / name)
        for folder in ["configs", "layouts", "scenes"]:
            for path in (root / folder).glob("*.json"):
                path.with_suffix(path.suffix + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest(), encoding="ascii")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json_write(root / "configs" / "STAGE_BUDGET.json", {
            "start_utc": now, "wall_limit_s": args.hours * 3600,
            "stop_new_physics_s": args.hours * 3600,
            "stop_trajectory_s": args.hours * 3600 + 120,
            "P0": 23050, "P4": 700, "all_direct": 24000,
            "formal_K2_runs": 96, "formal_K6_runs": 32, "workers": args.workers,
        })
        json_write(root / "run.json", {
            "mode": args.mode, "started_utc": now,
            "release_manifest_sha256": hashlib.sha256((REPO / "MANIFEST.json").read_bytes()).hexdigest(),
        })
    os.environ["SPRAY_RUN_DIR"] = str(root)
    sys.path.insert(0, str(REPO / "src"))
    return root


def preflight(root):
    for name in ["preflight", "gate_tests"]:
        spec = importlib.util.spec_from_file_location("release_test_" + name, REPO / "tests" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    from common import load
    from scene_model import BASELINE, engineering_feasible, in_local, scene
    from analysis import magnitude
    manifest = root / "layouts" / "ALL_TERMINAL_ENDPOINTS_FROZEN.json"
    if manifest.exists():
        rows = load("layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json")["rows"]
        assert len(rows) == 133
        import numpy as np
        for row in rows:
            particle = np.asarray(row["particle"])
            assert engineering_feasible(particle, scene(row["scene"]))
            if row["method"] == "K2":
                subset = [int(digit) - 1 for digit in row["pair_id"][1:]]
                assert in_local(particle, subset)
                assert magnitude(particle)[0]["modified_nozzles"] <= 2
            elif row["method"] in ["K0", "EMPTY"]:
                assert np.array_equal(particle, BASELINE)
        print("All 133 archived endpoint layouts satisfy their constraints.", flush=True)


def compare_csv(actual, expected):
    left, right = read_csv(actual), read_csv(expected)
    assert len(left) == len(right), expected.name
    for row, reference in zip(left, right):
        assert row.keys() == reference.keys(), expected.name
        for key in row:
            if row[key] == reference[key]:
                continue
            try:
                a, b = float(row[key]), float(reference[key])
            except ValueError:
                raise AssertionError((expected.name, key, row[key], reference[key]))
            assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), (expected.name, key, a, b)
    return len(left)


def verify(root):
    import numpy as np
    from common import load
    from analysis import main as analyze
    from scene_model import BASELINE
    endpoints = load("layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json")["rows"]
    primary = load("layouts/PRIMARY_K2_ENDPOINTS_FROZEN.json")["rows"]
    seeds = load("configs/SEEDS_FROZEN.json")
    selected = load("layouts/TOP3_PAIRS_FROZEN.json")["selected"]
    # Check that pair selection and primary endpoints use P0 scores only.
    screening = read_csv(REPO / "reference" / "subset_screening.csv")
    for scene in ["V1", "V2", "V3", "V4"]:
        ranked = sorted([row for row in screening if row["scene"] == scene],
                        key=lambda row: (-float(row["median_best"]), -float(row["best"]), row["pair_id"]))
        assert len(ranked) == 15
        assert [row["pair_id"] for row in ranked[:3]] == [row["pair_id"] for row in selected[scene]]
        for seed in seeds["formal_optimizer"]:
            candidates = [row for row in endpoints if row["scene"] == scene and row["method"] == "K2" and row["optimizer_seed"] == seed]
            assert len(candidates) == 3
            chosen = min(candidates, key=lambda row: (-row["P0_score"], row["pair_id"]))
            recorded = next(row for row in primary if row["scene"] == scene and row["optimizer_seed"] == seed)
            assert chosen["endpoint_id"] == recorded["endpoint_id"]
    raw = read_csv(root / "P4_terminal_all.csv") + read_csv(root / "empty_C001_reference.csv")
    assert len(raw) == 665
    assert len({(row["endpoint_id"], row["physical_seed"]) for row in raw}) == 665
    for endpoint in endpoints:
        samples = [row for row in raw if row["endpoint_id"] == endpoint["endpoint_id"]]
        assert sorted(int(row["physical_seed"]) for row in samples) == seeds["terminal_physical"]
        assert all(row["status"] in ["EVENT_COMPLETE", "VOLUME_TOLERANCE_COMPLETE"] for row in samples)
    with contextlib.redirect_stdout(io.StringIO()):
        analyze()
    tables = ["primary_effects", "P4_endpoint_summary", "P4_primary_comparison",
              "primary_effects_by_search_seed", "remaining_gap", "retrofit_magnitude",
              "retrofit_per_nozzle", "obstruction_loss", "subset_selection_frequency",
              "local_fidelity_diagnostics", "local_fidelity_pair_losses",
              "physical_seed_fluctuation", "directional_secondary"]
    counts = {name: compare_csv(root / (name + ".csv"), REPO / "reference" / (name + ".csv")) for name in tables}
    assert np.array_equal(np.load(root / "configs" / "bootstrap_indices.npy", allow_pickle=False),
                          np.load(REPO / "reference" / "bootstrap_indices.npy", allow_pickle=False))
    current, recorded = load("STATISTICAL_RESULTS.json"), json_read(REPO / "reference" / "STATISTICAL_RESULTS.json")
    assert current["status"] == recorded["status"]
    assert current["supported_scenes"] == recorded["supported_scenes"] == 0
    json_write(root / "verification.json", {"passed": True, "terminal_rows": 665, "tables_checked": counts,
               "primary_endpoint_selection_checked": True, "bootstrap_indices_equal": True})
    print("Recomputed 13 result tables from 665 archived P4 evaluations.")
    print("Scene  K2 - K0 (pp)  bootstrap 95% interval (pp)")
    for row in current["scenes"]:
        print(f"{row['scene']:5}  {row['mean_delta2_pp']:.9f}  [{row['bootstrap95_low_pp']:.9f}, {row['bootstrap95_high_pp']:.9f}]")
    print("Overall decision:", current["status"])


def smoke(root):
    from common import initialize_ledger, load
    from evaluator import evaluate
    from scene_model import BASELINE
    initialize_ledger()
    reference = read_csv(REPO / "reference" / "P4_terminal_all.csv")
    seeds = load("configs/SEEDS_FROZEN.json")
    checks = []
    for scene in ["V1", "V2", "V3", "V4"]:
        seed = seeds["search_physical"][scene]
        a = evaluate("smoke_" + scene + "_populate", BASELINE, scene, "P0", seed, "smoke")
        b = evaluate("smoke_" + scene + "_cached", BASELINE, scene, "P0", seed, "smoke")
        c = evaluate("smoke_" + scene + "_uncached", BASELINE, scene, "P0", seed, "smoke", cache_enabled=False)
        assert b["baseline_component_hits"] == 6 and c["baseline_component_hits"] == 0
        for key in ["visited_sha256", "coverage", "visited_volume_m3", "free_target_volume_m3", "fractions", "status"]:
            assert a[key] == b[key] == c[key], (scene, key)
        seed = seeds["terminal_physical"][0]
        result = evaluate("smoke_" + scene + "_P4", BASELINE, scene, "P4", seed, "smoke")
        recorded = next(row for row in reference if row["endpoint_id"] == "K0_" + scene and int(row["physical_seed"]) == seed)
        error = abs(result["coverage"] - float(recorded["coverage"]))
        assert error < 1e-12, (scene, error)
        checks.append({"scene": scene, "cache_matches_uncached": True, "P4_absolute_error": error})
        print(scene, "P0 cache equivalence and P4 reference match passed.", flush=True)
    json_write(root / "smoke_checks.json", {"passed": True, "P0_calls": 12, "P4_calls": 4, "checks": checks})


def evaluate_one(args, root):
    from common import initialize_ledger, load
    from evaluator import evaluate
    rows = load("layouts/ALL_TERMINAL_ENDPOINTS_FROZEN.json")["rows"]
    matches = [row for row in rows if row["endpoint_id"] == args.endpoint]
    if len(matches) != 1:
        raise ValueError("Unknown endpoint: " + args.endpoint)
    row = matches[0]
    initialize_ledger()
    result = evaluate("selected_endpoint", row["particle"], row["scene"], args.fidelity, args.seed, "evaluation")
    print(json.dumps({key: result[key] for key in ["scene", "fidelity", "seed", "coverage", "status"]}, indent=2))


def full(root, resume):
    from common import initialize_ledger, load, freeze, check_frozen
    import workflow
    import terminal
    from analysis import main as analyze

    def freeze_once(path, value):
        if (root / path).exists():
            check_frozen(path)
            if load(path) != value:
                raise ValueError("Resumed job definition changed: " + path)
            return
        return freeze(path, value)

    workflow.freeze = freeze_once
    if not resume:
        preflight(root)
    initialize_ledger()
    if not (root / "logs" / "StageA_complete.json").exists():
        workflow.stageA()
    if not (root / "logs" / "formal_complete.json").exists():
        workflow.formal()
    terminal.main()
    analyze()
    print("Full search, terminal evaluation, and analysis completed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "smoke", "evaluate", "full"])
    parser.add_argument("--out", help="New output directory (default: a timestamped directory under runs/).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--resume", action="store_true", help="Resume an existing full run; requires --out.")
    parser.add_argument("--endpoint", default="K0_V1")
    parser.add_argument("--fidelity", choices=["P0", "P4"], default="P4")
    parser.add_argument("--seed", type=int, default=2026101401)
    args = parser.parse_args()
    if args.workers < 1 or args.hours <= 0 or (args.resume and not args.out):
        parser.error("Use positive workers/hours and provide --out when resuming.")
    if sys.flags.optimize:
        parser.error("Run without -O; scientific assertions must remain enabled.")
    print("Release files verified:", check_release(), flush=True)
    root = prepare(args)
    print("Output directory:", root, flush=True)
    if args.mode in ["verify", "smoke"]:
        preflight(root)
    if args.mode == "verify":
        verify(root)
    elif args.mode == "smoke":
        smoke(root)
    elif args.mode == "evaluate":
        evaluate_one(args, root)
    else:
        full(root, args.resume)


if __name__ == "__main__":
    main()
