"""
transformers.Trainer-based training loop.

Uses per-device batch size 1 (see training_utils.py's module docstring
for why) — Trainer's own `gradient_accumulation_steps` setting produces
the effective batch size instead of actually padding/batching multiple
multimodal sequences together, which would reopen the same
masking-across-a-batch risk the manual loop and the earlier smoke-test
bug fix both went out of their way to avoid.

The custom `_TorchDataset`/`_collate_fn` pair exists ONLY to plug
`build_single_sample_inputs` into Trainer's Dataset/collator interface —
they don't do anything training_utils.py doesn't already do.

KNOWN GOTCHA (already handled below, documented so it isn't "fixed" back
out later): `TrainingArguments(remove_unused_columns=False)` is required
for any multimodal Trainer usage — Trainer's default behavior inspects
the model's forward signature and drops any dataset column it doesn't
recognize as a named argument, which silently strips `pixel_values` /
`image_grid_thw` (Qwen2.5-VL's actual multimodal input tensors) if left
on its default.

NOT VERIFIED AGAINST A REAL RUN — no GPU or Qwen weights were available
while building this. Run g3e2/train.py's staged smoke tests
(dataset_validation through tiny_overfit_test) successfully first; this
module reuses the exact same model-loading and masking code those already
exercise, but Trainer's own batching/checkpointing/logging machinery
around it has not been executed here.
"""
from __future__ import annotations

from pathlib import Path

from g3e2.training_utils import build_single_sample_inputs
from g3e2.trainers.common import evaluate_state_accuracy, save_checkpoint


class _TorchDataset:
    """
    Deliberately NOT a real torch.utils.data.Dataset subclass at module
    load time (no `import torch` at the top of this file) — so importing
    this module doesn't require torch to be installed until `run()` is
    actually called. Trainer only needs `__len__`/`__getitem__`, which
    this provides regardless of the formal base class.
    """

    def __init__(self, samples: list, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Built on CPU; Trainer moves batches to the model's device itself.
        inputs = build_single_sample_inputs(self.processor, self.samples[idx], device="cpu")
        return {k: v.squeeze(0) for k, v in inputs.items()}


def _collate_fn(features: list) -> dict:
    """
    Requires per_device_train_batch_size=1 — re-adds the batch dimension
    `_TorchDataset.__getitem__` removed, rather than padding/stacking
    multiple samples (which would need per-sample-aware label masking
    across the padded region — exactly the risk this whole per-sample
    design avoids). See module docstring.
    """
    if len(features) != 1:
        raise RuntimeError(
            "g3e2/trainers/hf_trainer.py requires per_device_train_batch_size=1 "
            f"— got a batch of {len(features)}. See module docstring."
        )
    return {k: v.unsqueeze(0) for k, v in features[0].items()}


def run(
    model,
    processor,
    train_samples: list,
    val_samples: list,
    num_epochs: int,
    cfg: dict,
    output_dir,
) -> Path:
    from transformers import Trainer, TrainingArguments

    train_cfg = cfg["training"]
    if train_cfg["per_device_train_batch_size"] != 1:
        raise RuntimeError(
            "g3e2/trainers/hf_trainer.py requires per_device_train_batch_size=1 in config.yaml "
            "— use gradient_accumulation_steps for effective batch size. See module docstring."
        )

    train_dataset = _TorchDataset(train_samples, processor)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        bf16=train_cfg.get("bf16", True),
        max_grad_norm=1.0,
        remove_unused_columns=False,  # REQUIRED for multimodal — see module docstring
        report_to=[],  # wire up "wandb"/"tensorboard" here if your deployment wants that
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=_collate_fn,
    )
    trainer.train()

    # Trainer's own periodic checkpoints land under output_dir/checkpoint-N
    # (its default behavior, not something this module adds) — the final,
    # versioned adapter save below is separate and is what g3e2/predict.py
    # / g3e_inference expect to consume.
    if val_samples:
        result = evaluate_state_accuracy(model, processor, val_samples)
        print(
            f"  [hf_trainer] final eval: state accuracy = {result['accuracy']:.2%} "
            f"per_class={result['per_class_accuracy']}"
        )

    final_dir = save_checkpoint(model, Path(output_dir) / "final", cfg)
    print(f"  [hf_trainer] final checkpoint saved to {final_dir}")
    return final_dir
