# FSDP vs `device_map="auto"` benchmark (issue #59)

Measures whether FSDP speeds up full-parameter fine-tuning compared with today's
`device_map="auto"` setup, before any attack code is refactored.

Both runs use the same model, data (the attack's own `load_prompt_completions_dataset`,
harmful data, `poison_ratio=1`), training settings (copied from `FullParameterFinetune.run_attack`)
and **global batch size**. Gradient accumulation is derived from the number of processes so the
global batch matches:

| mode | processes | per-device batch | grad accum | global batch |
|---|---|---|---|---|
| `device_map` | 1 (layers split over 2 GPUs) | 8 | 2 | 16 |
| `fsdp` | 2 (one per GPU) | 8 | 1 | 16 |

Precision: TamperBench today loads weights in bf16 and TRL's `SFTConfig` defaults to `bf16=True`.
Under FSDP, `bf16=True` makes accelerate upcast the weights to fp32 (more memory, different maths),
so the default FSDP run keeps bf16 weights without mixed precision. `--fsdp-mixed-precision`
measures the upcast variant separately.

## Hardware

One machine with 2× 80 GB GPUs (A100/H100), ~150 GB disk, ≥128 GB RAM.

## Commands (on the rented machine)

```bash
git clone https://github.com/greninja/TamperBench.git && cd TamperBench
git checkout greninja/fsdp-benchmark
uv sync

# 1. Smoke test with a tiny model (a few minutes; catches script errors cheaply)
uv run python scripts/fsdp_benchmark/run_benchmark.py --mode device_map --model Qwen/Qwen3-0.6B --max-steps 8 --warmup-steps 2
uv run accelerate launch --config_file scripts/fsdp_benchmark/fsdp_2gpu.yaml \
    scripts/fsdp_benchmark/run_benchmark.py --mode fsdp --model Qwen/Qwen3-0.6B --max-steps 8 --warmup-steps 2

# 2. The real runs (Qwen3-8B)
uv run python scripts/fsdp_benchmark/run_benchmark.py --mode device_map
uv run accelerate launch --config_file scripts/fsdp_benchmark/fsdp_2gpu.yaml \
    scripts/fsdp_benchmark/run_benchmark.py --mode fsdp

# 3. Optional: FSDP with bf16 mixed precision (fp32 upcast) — may not fit; that is a result too
uv run accelerate launch --config_file scripts/fsdp_benchmark/fsdp_2gpu.yaml \
    scripts/fsdp_benchmark/run_benchmark.py --mode fsdp --fsdp-mixed-precision

# 4. Compare (prints a markdown table to paste into the issue)
uv run python scripts/fsdp_benchmark/compare.py results/fsdp_benchmark/*.json
```

Results are written to `results/fsdp_benchmark/<mode>_<precision>_<model>.json`, including
per-step times, per-step loss, peak memory per GPU, GPU names and `nvidia-smi topo -m` output.

## Reading the results

- **s/step / samples/s**: the speed comparison. The first `--warmup-steps` steps are excluded.
- **last-10 loss**: a sanity check that both modes train comparably. Losses won't match exactly:
  with 2 processes each GPU draws different examples, so batches differ.
- **peak mem GiB**: allocated memory per GPU (`device_map`) or per rank (`fsdp`).
