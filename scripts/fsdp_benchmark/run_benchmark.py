"""Benchmark full-parameter fine-tuning: ``device_map="auto"`` vs FSDP (issue #59).

Trains the same model on the same data with the same global batch size in one of two modes
and writes timing, memory, and loss to a JSON file:

* ``device_map``: what ``run_full_finetune_attack`` does today. One process loads the model
  with ``device_map="auto"`` (layers split across GPUs, which run one after another).
  Run with plain ``python``.
* ``fsdp``: one process per GPU, launched with ``accelerate launch`` and an FSDP config.
  Each GPU holds a shard of every layer and trains on its own micro-batch.

Training settings mirror ``FullParameterFinetune.run_attack`` (SFTConfig, gradient
checkpointing, completion-only loss, max_length=2048) and the dataset is built with the
attack's own ``load_prompt_completions_dataset``, so the baseline matches TamperBench.

See ``scripts/fsdp_benchmark/README.md`` for the exact commands.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from tamperbench.whitebox.attacks.full_parameter_finetune.full_parameter_finetune import (
    FullParameterFinetune,
    FullParameterFinetuneConfig,
)
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.models.default_tokens import DEFAULT_PAD_TOKEN
from tamperbench.whitebox.utils.names import TemplateName


class StepTimer(TrainerCallback):
    """Record wall-clock time per optimizer step and the logged loss."""

    def __init__(self) -> None:
        """Initialize empty timing and loss records."""
        self._start: float | None = None
        self.step_seconds: list[float] = []
        self.losses: list[float] = []

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Start the clock once all queued GPU work has finished."""
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Stop the clock after the optimizer step (includes all gradient-accumulation micro-steps)."""
        torch.cuda.synchronize()
        if self._start is not None:
            self.step_seconds.append(time.perf_counter() - self._start)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Keep the per-step training loss (logged every step)."""
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["device_map", "fsdp"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id or local path.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/fsdp_benchmark"))
    parser.add_argument("--max-steps", type=int, default=30, help="Optimizer steps to run.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Leading steps excluded from timing.")
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=16,
        help=(
            "Examples per optimizer step, kept equal across modes. Gradient accumulation is "
            "derived as global / (per_device * num_processes)."
        ),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fsdp-mixed-precision",
        action="store_true",
        help=(
            "FSDP mode only: turn on bf16 mixed precision (TRL's default). accelerate then upcasts the "
            "bf16 weights to fp32, roughly doubling weight + optimizer memory. Off by default so both "
            "modes train bf16 weights with bf16 optimizer state, like TamperBench does today."
        ),
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Also time trainer.save_model() (writes a full ~16 GB checkpoint for an 8B model).",
    )
    return parser.parse_args()


def gpu_topology() -> str:
    """Return `nvidia-smi topo -m` output (shows whether GPUs are linked by NVLink)."""
    try:
        return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return "nvidia-smi not found"


def build_dataset(args: argparse.Namespace, out_dir: Path) -> Any:
    """Build the training set with the attack's own dataset code (harmful data, poison_ratio=1)."""
    dataset_size = max(args.global_batch_size * args.max_steps, 64)
    attack_config = FullParameterFinetuneConfig(
        input_checkpoint_path=args.model,
        out_dir=str(out_dir),
        evals=[],
        model_config=ModelConfig(
            template_name=TemplateName("plain"),
            max_generation_length=1024,
            inference_batch_size=16,
        ),
        random_seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        max_steps=args.max_steps,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        dataset_size=dataset_size,
        poison_ratio=1.0,
    )
    return FullParameterFinetune(attack_config).load_prompt_completions_dataset()


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer the same way `run_full_finetune_attack` does, except device placement."""
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.bfloat16,
        # device_map: split layers across GPUs (today's behaviour).
        # fsdp: load on CPU; the Trainer/accelerate shards it across processes.
        device_map="auto" if args.mode == "device_map" else None,
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict={"pad_token": DEFAULT_PAD_TOKEN})
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    model.enable_input_require_grads()
    return model, tokenizer


def peak_memory_gib() -> dict[str, dict[str, float]]:
    """Peak allocated/reserved memory for every GPU visible to this process."""
    return {
        f"cuda:{i}": {
            "allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
            "reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30,
        }
        for i in range(torch.cuda.device_count())
    }


def main() -> None:
    """Run one benchmark mode and write a JSON result file."""
    args = parse_args()
    state = PartialState()
    world_size = state.num_processes

    if args.mode == "fsdp" and world_size < 2:
        raise SystemExit("fsdp mode must be started with `accelerate launch` on 2+ GPUs (see README).")
    if args.mode == "device_map" and world_size != 1:
        raise SystemExit("device_map mode must be started with plain `python` (one process).")

    per_step = args.per_device_batch_size * world_size
    if args.global_batch_size % per_step != 0:
        raise SystemExit(
            f"--global-batch-size {args.global_batch_size} must be divisible by "
            f"per_device_batch_size * num_processes = {per_step}."
        )
    grad_accum = args.global_batch_size // per_step

    precision = "fsdp_bf16_mixed" if (args.mode == "fsdp" and args.fsdp_mixed_precision) else "bf16"
    run_name = f"{args.mode}_{precision}_{Path(args.model).name}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with state.main_process_first():  # download/prepare data once, then other ranks read the cache
        dataset = build_dataset(args, out_dir)

    model, tokenizer = load_model_and_tokenizer(args)

    # TRL defaults bf16=True. Keep that for device_map (exactly today's behaviour). Under FSDP,
    # bf16=True makes accelerate upcast weights to fp32, so it is only on with --fsdp-mixed-precision.
    use_bf16_amp = args.mode == "device_map" or args.fsdp_mixed_precision

    training_arguments = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        max_steps=args.max_steps,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16_amp,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        completion_only_loss=True,
        max_length=args.max_length,
        seed=args.seed,
    )

    timer = StepTimer()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_arguments,
        callbacks=[timer],
    )

    torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - train_start

    save_seconds: float | None = None
    if args.save_checkpoint:
        save_start = time.perf_counter()
        trainer.save_model(output_dir=str(out_dir / "checkpoint"))
        save_seconds = time.perf_counter() - save_start

    if args.mode == "device_map":
        memory = peak_memory_gib()
    else:
        own_device = f"cuda:{torch.cuda.current_device()}"
        memory = {f"rank{state.process_index}": peak_memory_gib()[own_device]}
    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, memory)
        memory = {k: v for part in gathered for k, v in part.items()}

    if not state.is_main_process:
        return

    timed = timer.step_seconds[args.warmup_steps :]
    mean_step = statistics.mean(timed) if timed else float("nan")
    result: dict[str, Any] = {
        "mode": args.mode,
        "precision": precision,
        "model": args.model,
        "num_processes": world_size,
        "num_gpus_visible": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "global_batch_size": args.global_batch_size,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "warmup_steps_excluded": args.warmup_steps,
        "mean_step_seconds": mean_step,
        "median_step_seconds": statistics.median(timed) if timed else float("nan"),
        "samples_per_second": args.global_batch_size / mean_step if timed else float("nan"),
        "total_train_seconds": train_seconds,
        "save_seconds": save_seconds,
        "step_seconds": timer.step_seconds,
        "losses": timer.losses,
        "mean_loss_last_10": statistics.mean(timer.losses[-10:]) if timer.losses else float("nan"),
        "peak_memory_gib": memory,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu_topology": gpu_topology(),
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "ACCELERATE_MIXED_PRECISION")},
    }
    result_path = args.out_dir / f"{run_name}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {result_path}")
    print(
        f"{run_name}: {result['mean_step_seconds']:.2f} s/step, "
        f"{result['samples_per_second']:.2f} samples/s, last-10 loss {result['mean_loss_last_10']:.4f}"
    )


if __name__ == "__main__":
    main()
