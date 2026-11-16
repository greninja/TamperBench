"""Print a side-by-side comparison of benchmark result JSON files.

Usage:
    uv run python scripts/fsdp_benchmark/compare.py results/fsdp_benchmark/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def peak_gib(result: dict[str, Any]) -> str:
    """Format peak allocated memory per GPU/rank."""
    return ", ".join(f"{k}={v['allocated_gib']:.1f}" for k, v in result["peak_memory_gib"].items())


def main() -> None:
    """Load result files and print a markdown table (paste-able into the GitHub issue)."""
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    results = [json.loads(p.read_text()) for p in paths]
    baseline = next((r for r in results if r["mode"] == "device_map"), None)

    columns = [
        "run",
        "GPUs",
        "per-device bs × grad accum × procs",
        "s/step",
        "samples/s",
        "speedup",
        "last-10 loss",
        "peak mem GiB",
    ]
    print("| " + " | ".join(columns) + " |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        speedup = baseline["mean_step_seconds"] / r["mean_step_seconds"] if baseline else float("nan")
        print(
            f"| {r['mode']} ({r['precision']}) | {r['num_gpus_visible']}× {r['gpu_names'][0]} "
            f"| {r['per_device_batch_size']} × {r['gradient_accumulation_steps']} × {r['num_processes']} "
            f"| {r['mean_step_seconds']:.2f} | {r['samples_per_second']:.2f} | {speedup:.2f}× "
            f"| {r['mean_loss_last_10']:.4f} | {peak_gib(r)} |"
        )
    print(f"\nModel: {results[0]['model']}, global batch {results[0]['global_batch_size']}, "
          f"max_length {results[0]['max_length']}, {results[0]['max_steps']} steps "
          f"(first {results[0]['warmup_steps_excluded']} excluded from timing).")


if __name__ == "__main__":
    main()
