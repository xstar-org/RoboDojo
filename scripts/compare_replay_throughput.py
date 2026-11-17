"""Gate fixed-action replay throughput; this does not certify policy scores or RGB parity.

Inputs are completed replay directories containing benchmark.json,
runtime-profile.json and FINISHED, as produced by the N56 replay protocol.
Use a 1-env baseline from the same Isaac Sim version as the candidate.
"""
import argparse
import json
import math
from pathlib import Path


def load_run(directory):
    directory = Path(directory)
    if not (directory / "FINISHED").is_file() or (directory / "FAILED.json").exists():
        raise ValueError(f"Incomplete or failed replay: {directory}")
    data = json.loads((directory / "benchmark.json").read_text())
    profile = json.loads((directory / "runtime-profile.json").read_text())
    for name in ("steps", "num_envs"):
        if type(data[name]) is not int or data[name] < 1:
            raise ValueError(f"Invalid {name} in {directory}")
    wall = data["episode_wall_s"]
    mean = data["step_total_s"]["mean"]
    if not all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in (wall, mean)):
        raise ValueError(f"Invalid replay timing in {directory}")
    if not math.isclose(wall, mean * data["steps"], rel_tol=1e-6):
        raise ValueError(f"Replay wall time disagrees with mean timing in {directory}")
    if profile["settings"] != data["render_profile"]:
        raise ValueError(f"Renderer profile changed in {directory}")
    if not profile.get("versions", {}).get("isaacsim"):
        raise ValueError(f"Missing Isaac Sim version in {directory}")
    return data, profile


def compare(reference, candidate, min_speedup):
    if not math.isfinite(min_speedup) or min_speedup <= 1:
        raise ValueError("min_speedup must be finite and greater than 1")
    baseline, base_profile = load_run(reference)
    trial, trial_profile = load_run(candidate)
    if baseline["num_envs"] != 1:
        raise ValueError("Reference must be the same-runtime 1-env replay")
    for name in ("steps", "tape_sha256", "render_profile"):
        if baseline[name] != trial[name]:
            raise ValueError(f"Incomparable replay field: {name}")
    if base_profile["versions"] != trial_profile["versions"]:
        raise ValueError("Runtime versions differ; run a new same-runtime reference")
    base_rate = baseline["num_envs"] * baseline["steps"] / baseline["episode_wall_s"]
    trial_rate = trial["num_envs"] * trial["steps"] / trial["episode_wall_s"]
    speedup = trial_rate / base_rate
    return {
        "reference_env_steps_per_second": base_rate,
        "candidate_env_steps_per_second": trial_rate,
        "speedup": speedup,
        "minimum_speedup": min_speedup,
        "passed": speedup >= min_speedup,
        "scope": "Fixed-action replay only; RGB parity and closed-loop evaluation require separate acceptance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    args = parser.parse_args()
    try:
        result = compare(args.reference, args.candidate, args.min_speedup)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Cannot qualify throughput: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
