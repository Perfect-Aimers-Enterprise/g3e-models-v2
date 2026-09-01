"""
Hand-rolled training loop — full manual control, easiest to verify
against training_utils.py's already-tested masking logic since there's no
framework code in between. This is FULL_TRAINING.md's specification
turned into real code.

Design choices (see training_utils.py's module docstring for the deeper
reasoning): per-SAMPLE processing (never a padded batch), gradient
accumulation for effective batch size, gradient clipping, and a linear
warmup schedule.
"""
from __future__ import annotations

import random
from pathlib import Path

from g3e2.training_utils import build_single_sample_inputs
from g3e2.trainers.common import evaluate_state_accuracy, save_checkpoint


def run(
    model,
    processor,
    train_samples: list,
    val_samples: list,
    num_epochs: int,
    cfg: dict,
    output_dir,
    seed: int = 42,
) -> Path:
    import torch
    from transformers import get_linear_schedule_with_warmup

    train_cfg = cfg["training"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(train_samples) // grad_accum)
    total_steps = steps_per_epoch * num_epochs

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_cfg["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    rng = random.Random(seed)
    global_step = 0

    for epoch in range(num_epochs):
        epoch_samples = list(train_samples)
        rng.shuffle(epoch_samples)  # seeded — reproducible across re-runs with the same seed

        optimizer.zero_grad()
        for i, messages in enumerate(epoch_samples):
            inputs = build_single_sample_inputs(processor, messages, model.device)
            loss = model(**inputs).loss / grad_accum
            loss.backward()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % train_cfg["logging_steps"] == 0:
                    print(f"  [manual] epoch {epoch} step {global_step}: loss={loss.item() * grad_accum:.4f}")

                if val_samples and global_step % train_cfg["eval_steps"] == 0:
                    result = evaluate_state_accuracy(model, processor, val_samples)
                    print(
                        f"    eval: state accuracy = {result['accuracy']:.2%} "
                        f"({result['correct']}/{result['total']}) per_class={result['per_class_accuracy']}"
                    )

                if global_step % train_cfg["save_steps"] == 0:
                    save_checkpoint(model, Path(output_dir) / f"step_{global_step}", cfg)

    final_dir = save_checkpoint(model, Path(output_dir) / "final", cfg)
    print(f"  [manual] final checkpoint saved to {final_dir}")
    return final_dir
