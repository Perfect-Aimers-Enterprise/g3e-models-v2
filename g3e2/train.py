#!/usr/bin/env python3
"""
G3E-2 fine-tuning of Qwen2.5-VL — LoRA or QLoRA, your choice at runtime.

    python g3e2/train.py --method lora   --stage tiny_overfit_test
    python g3e2/train.py --method qlora  --stage tiny_overfit_test

`--method` (or `training.method` in config.yaml, if you don't want to pass
it every time) selects between:
  - lora:  full-precision (bf16) base model, LoRA adapters on top. Needs
           more GPU memory but is simpler and slightly faster per step.
  - qlora: base model loaded in 4-bit (bitsandbytes), LoRA adapters on
           top of that. Needs ~1/3 the memory of `lora` for the base
           model; use this if you're hitting OOM. Requires
           `pip install bitsandbytes` (not a default dependency — see
           requirements.txt).

Per spec section 18, this NEVER jumps straight to a multi-hour training
run. `main()` walks through every stage in order and refuses to proceed
to the next one if the current one fails:

    dataset_validation -> sample_load_test -> batch_forward_test
    -> batch_backward_test -> lora_param_check -> tiny_overfit_test
    -> short_training_run -> full_training

Run with `--stage <name>` to run only up to (and including) one stage.

`short_training_run` and `full_training` ARE implemented, via
`--trainer {manual,hf_trainer,trl_sft}` (default: manual) — see
g3e2/trainers/. All three share the exact same tested loss-masking
(training_utils.py) and model-loading (this file's staged functions)
code; they differ only in which framework drives the loop:
  - manual:     hand-rolled loop, full control, easiest to audit.
  - hf_trainer: transformers.Trainer — its logging/checkpointing for free.
  - trl_sft:    trl's SFTTrainer — bypasses trl's own multimodal
                preprocessing (unverified masking behavior) in favor of
                the same tested masking the other two use.

NONE of the three have been run against a real GPU/model in this
environment (none was available) — see each module in g3e2/trainers/ for
exactly what IS verified (the masking algorithm, offline) vs. what isn't
(the actual training run). Read FULL_TRAINING.md before your first real
run regardless of which `--trainer` you pick.

REQUIRES: torch, transformers, peft (+ bitsandbytes for --method qlora,
+ trl and datasets for --trainer trl_sft), and enough GPU memory to load
Qwen2.5-VL-3B-Instruct (~6-7GB in bf16 for `lora`, ~2-3GB in 4-bit for
`qlora` — both need more on top for activations/optimizer state). This
script cannot be smoke-tested without those installed and a real GPU —
see README.md "What you need to actually run this."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from g3e2.dataset import G3E2Dataset
from g3e2.training_utils import build_single_sample_inputs

STAGE_ORDER = [
    "dataset_validation",
    "sample_load_test",
    "batch_forward_test",
    "batch_backward_test",
    "lora_param_check",
    "tiny_overfit_test",
    "short_training_run",
    "full_training",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_dataset_validation(cfg: dict, method: str) -> None:
    print("=== STAGE: dataset_validation ===")
    for split in ("train", "val", "test"):
        jsonl_path = Path(cfg["data"][f"{split}_jsonl"])
        if not jsonl_path.exists():
            raise RuntimeError(f"{jsonl_path} does not exist — run scripts/prepare_g3e2.py first.")
        ds = G3E2Dataset(jsonl_path)
        if len(ds) == 0:
            raise RuntimeError(f"{jsonl_path} has zero samples.")
        print(f"  {split}: {len(ds)} sample(s) — OK")
    print("  PASSED\n")


def stage_sample_load_test(cfg: dict, method: str) -> None:
    print("=== STAGE: sample_load_test ===")
    ds = G3E2Dataset(cfg["data"]["train_jsonl"])
    item = ds[0]
    messages = item["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert any(block["type"] == "image" for block in messages[1]["content"])
    assert messages[2]["role"] == "assistant"
    json.loads(messages[2]["content"])
    print(f"  loaded sample 0 ({item['record']['id']}) — messages well-formed, target parses as JSON")
    print("  PASSED\n")


def _load_model_and_processor(cfg: dict, method: str):
    """
    Isolated so the earlier, cheap stages (dataset_validation,
    sample_load_test) can run and fail fast WITHOUT needing torch/
    transformers/a GPU/network access to Hugging Face at all — only the
    stages from batch_forward_test onward actually need the real model.

    `method`: "lora" or "qlora" — see this file's module docstring.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))

    processor = AutoProcessor.from_pretrained(model_cfg["base_model"])

    if method == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "--method qlora requires bitsandbytes. Install it with `pip install bitsandbytes`."
            ) from exc
        from peft import prepare_model_for_kbit_training

        qlora_cfg = cfg["training"].get("qlora", {})
        quant_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg.get("load_in_4bit", True),
            bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
        )
        print(f"  loading base model in 4-bit (QLoRA): {model_cfg['base_model']}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_cfg["base_model"], quantization_config=quant_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    elif method == "lora":
        print(f"  loading base model in {model_cfg.get('torch_dtype', 'bfloat16')} (LoRA, no quantization): "
              f"{model_cfg['base_model']}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_cfg["base_model"], torch_dtype=dtype, device_map="auto"
        )
    else:
        raise ValueError(f"Unknown method {method!r} — expected 'lora' or 'qlora'.")

    # Print/verify actual module names before attaching LoRA — the
    # config's target_modules must be checked against THIS list, not
    # assumed. See config.yaml's comment on not blindly trusting "merger" etc.
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [m for m in lora_cfg["target_modules"] if m not in module_names]
    if missing:
        raise RuntimeError(
            f"config target_modules {missing} not found among this model's actual module "
            f"names. Inspect `model.named_modules()` and fix g3e2/config.yaml before proceeding."
        )

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, processor


def stage_batch_forward_test(cfg: dict, method: str):
    print("=== STAGE: batch_forward_test ===")
    model, processor = _load_model_and_processor(cfg, method)
    ds = G3E2Dataset(cfg["data"]["train_jsonl"])

    # Per-sample (batch size 1) — see training_utils.py's module docstring
    # for why this avoids the multimodal batch-padding/masking pitfalls.
    inputs = build_single_sample_inputs(processor, ds[0]["messages"], model.device)
    outputs = model(**inputs)
    print(f"  forward pass OK — logits shape: {tuple(outputs.logits.shape)}")
    print("  PASSED\n")
    return model, processor


def stage_batch_backward_test(cfg: dict, method: str):
    print("=== STAGE: batch_backward_test ===")
    model, processor = stage_batch_forward_test(cfg, method)
    ds = G3E2Dataset(cfg["data"]["train_jsonl"])

    inputs = build_single_sample_inputs(processor, ds[0]["messages"], model.device)
    outputs = model(**inputs)
    outputs.loss.backward()
    print(f"  backward pass OK — loss: {outputs.loss.item():.4f}")
    print("  PASSED\n")
    return model, processor


def stage_lora_param_check(cfg: dict, method: str):
    print("=== STAGE: lora_param_check ===")
    model, processor = stage_batch_backward_test(cfg, method)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora_trainable = [n for n in trainable if "lora_" not in n]
    if non_lora_trainable:
        raise RuntimeError(
            f"Found {len(non_lora_trainable)} trainable parameter(s) outside LoRA adapters "
            f"(e.g. {non_lora_trainable[:3]}) — the base model should be fully frozen."
        )
    print(f"  {len(trainable)} trainable parameter(s), all LoRA — base model correctly frozen")
    print("  PASSED\n")
    return model, processor


def stage_tiny_overfit_test(cfg: dict, method: str):
    print("=== STAGE: tiny_overfit_test ===")
    import torch

    model, processor = stage_lora_param_check(cfg, method)
    overfit_cfg = cfg["training"]["tiny_overfit_test"]
    ds = G3E2Dataset(cfg["data"]["train_jsonl"])
    n = min(overfit_cfg["num_samples"], len(ds))
    samples = [ds[i]["messages"] for i in range(n)]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for step in range(overfit_cfg["num_steps"]):
        # Gradient-accumulate over the n samples (per-sample processing —
        # see training_utils.py) so this is one optimizer step per outer
        # loop iteration, not n steps.
        optimizer.zero_grad()
        step_loss = 0.0
        for messages in samples:
            inputs = build_single_sample_inputs(processor, messages, model.device)
            outputs = model(**inputs)
            (outputs.loss / n).backward()
            step_loss += outputs.loss.item() / n
        optimizer.step()
        losses.append(step_loss)
        if step % 10 == 0:
            print(f"    step {step}: loss={step_loss:.4f}")

    threshold = overfit_cfg["loss_should_drop_below"]
    if losses[-1] >= threshold:
        raise RuntimeError(
            f"Tiny overfit test FAILED — final loss {losses[-1]:.4f} did not drop below "
            f"{threshold}. Do not proceed to full training; something is wrong with the "
            "data pipeline, model setup, or LoRA config. See spec section 18."
        )
    print(f"  final loss {losses[-1]:.4f} < {threshold} — model can learn from this data")
    print("  PASSED\n")


def stage_short_training_run(cfg: dict, method: str, trainer: str):
    print(f"=== STAGE: short_training_run (trainer={trainer}) ===")
    model, processor = stage_lora_param_check(cfg, method)

    train_ds = G3E2Dataset(cfg["data"]["train_jsonl"])
    val_ds = G3E2Dataset(cfg["data"]["val_jsonl"])

    short_cfg = cfg["training"].get("short_training_run", {})
    n_train = min(short_cfg.get("num_train_samples", 100), len(train_ds))
    n_val = min(short_cfg.get("num_val_samples", 20), len(val_ds))

    run_fn = _load_trainer_run_fn(trainer)
    output_dir = Path(cfg["training"]["output_dir"]) / "short_run"
    run_fn(
        model, processor,
        train_samples=[train_ds[i]["messages"] for i in range(n_train)],
        val_samples=[val_ds[i]["messages"] for i in range(n_val)],
        num_epochs=1,
        cfg=cfg,
        output_dir=output_dir,
    )
    print("  PASSED — inspect the loss curve and one saved checkpoint by hand before full_training\n")


def stage_full_training(cfg: dict, method: str, trainer: str):
    print(f"=== STAGE: full_training (trainer={trainer}) ===")
    model, processor = stage_lora_param_check(cfg, method)

    train_ds = G3E2Dataset(cfg["data"]["train_jsonl"])
    val_ds = G3E2Dataset(cfg["data"]["val_jsonl"])

    versioning = cfg["versioning"]
    output_dir = Path(cfg["training"]["output_dir"]) / f"{versioning['model_name']}-{versioning['version']}"

    run_fn = _load_trainer_run_fn(trainer)
    final_dir = run_fn(
        model, processor,
        train_samples=[train_ds[i]["messages"] for i in range(len(train_ds))],
        val_samples=[val_ds[i]["messages"] for i in range(len(val_ds))],
        num_epochs=cfg["training"]["num_train_epochs"],
        cfg=cfg,
        output_dir=output_dir,
    )
    print(f"  PASSED — final adapter saved to {final_dir}\n")


def _load_trainer_run_fn(trainer: str):
    """Lazy import — so choosing e.g. `manual` never requires trl/datasets
    to be installed, and choosing `trl_sft` never requires anything the
    other two don't also need."""
    if trainer == "manual":
        from g3e2.trainers import manual
        return manual.run
    if trainer == "hf_trainer":
        from g3e2.trainers import hf_trainer
        return hf_trainer.run
    if trainer == "trl_sft":
        from g3e2.trainers import trl_sft
        return trl_sft.run
    raise ValueError(f"Unknown trainer {trainer!r} — expected 'manual', 'hf_trainer', or 'trl_sft'.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--stage", default="tiny_overfit_test", choices=STAGE_ORDER,
                         help="Run stages up to and including this one, then stop.")
    parser.add_argument("--method", default=None, choices=["lora", "qlora"],
                         help="Overrides training.method in config.yaml if given.")
    parser.add_argument("--trainer", default=None, choices=["manual", "hf_trainer", "trl_sft"],
                         help="Overrides training.trainer in config.yaml if given. Only matters for "
                              "short_training_run/full_training — see g3e2/trainers/.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method = args.method or cfg.get("training", {}).get("method", "lora")
    trainer = args.trainer or cfg.get("training", {}).get("trainer", "manual")
    print(f"Method: {method}  |  Trainer: {trainer}\n")

    stages_to_run = STAGE_ORDER[: STAGE_ORDER.index(args.stage) + 1]

    stage_fns = {
        "dataset_validation": lambda: stage_dataset_validation(cfg, method),
        "sample_load_test": lambda: stage_sample_load_test(cfg, method),
        "batch_forward_test": lambda: stage_batch_forward_test(cfg, method),
        "batch_backward_test": lambda: stage_batch_backward_test(cfg, method),
        "lora_param_check": lambda: stage_lora_param_check(cfg, method),
        "tiny_overfit_test": lambda: stage_tiny_overfit_test(cfg, method),
        "short_training_run": lambda: stage_short_training_run(cfg, method, trainer),
        "full_training": lambda: stage_full_training(cfg, method, trainer),
    }

    for stage in stages_to_run:
        stage_fns[stage]()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
