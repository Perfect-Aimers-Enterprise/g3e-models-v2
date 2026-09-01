"""
trl SFTTrainer-based training loop.

IMPORTANT — why this bypasses trl's own multimodal preprocessing: trl's
built-in vision-language handling (as of the versions available when this
was written) is not guaranteed to mask the loss the same way
training_utils.py's tests verify (system prompt + detections list
ignored, only the assistant's JSON counted) — some reference
implementations mask only padding tokens, which would silently
reintroduce the exact "trained on the whole sequence" bug this repo
already found and fixed once in an earlier version of tiny_overfit_test.

To avoid inheriting that risk, the "dataset" trl sees here is just a
column of INDICES — a custom `data_collator` looks the real sample up by
index and builds it via `training_utils.build_single_sample_inputs`, the
exact same already-tested masking used by every other stage in this repo.
This gets trl's Trainer machinery (logging, checkpointing, callbacks)
without inheriting any of its own VLM-specific preprocessing assumptions.

VERSION RISK: `SFTConfig`'s exact field names (e.g. `dataset_kwargs`,
`skip_prepare_dataset`) have changed across trl releases. The names used
below are correct for a recent-as-of-writing trl version — check
`SFTConfig`'s actual signature in your installed version
(`python -c "from trl import SFTConfig; help(SFTConfig)"`) before trusting
this file verbatim; a mismatched field name will raise a clear
`TypeError` at construction time rather than silently doing the wrong
thing, so this fails safe even if it's out of date.

NOT VERIFIED AGAINST A REAL RUN — no GPU, Qwen weights, or trl install
were available while building this.

Requires: pip install trl datasets
"""
from __future__ import annotations

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
) -> Path:
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise RuntimeError("g3e2/trainers/trl_sft.py requires trl: pip install trl") from exc
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("g3e2/trainers/trl_sft.py requires datasets: pip install datasets") from exc

    train_cfg = cfg["training"]
    if train_cfg["per_device_train_batch_size"] != 1:
        raise RuntimeError(
            "g3e2/trainers/trl_sft.py requires per_device_train_batch_size=1 in config.yaml "
            "— use gradient_accumulation_steps for effective batch size. See module docstring."
        )

    # Index-only dataset — see module docstring for why the real sample
    # data isn't handed to trl's own preprocessing directly.
    hf_dataset = Dataset.from_list([{"idx": i} for i in range(len(train_samples))])

    def _collate_fn(features: list) -> dict:
        if len(features) != 1:
            raise RuntimeError(
                "g3e2/trainers/trl_sft.py requires per_device_train_batch_size=1 "
                f"— got a batch of {len(features)}. See module docstring."
            )
        idx = features[0]["idx"]
        return build_single_sample_inputs(processor, train_samples[idx], device="cpu")

    sft_config = SFTConfig(
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
        remove_unused_columns=False,
        report_to=[],
        # Tells trl not to run its own chat-template/tokenization pipeline
        # on `hf_dataset` — _collate_fn already produces final model
        # inputs. VERIFY this field name against your installed trl
        # version (see module docstring) before relying on it.
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=hf_dataset,
        data_collator=_collate_fn,
    )
    trainer.train()

    if val_samples:
        result = evaluate_state_accuracy(model, processor, val_samples)
        print(
            f"  [trl_sft] final eval: state accuracy = {result['accuracy']:.2%} "
            f"per_class={result['per_class_accuracy']}"
        )

    final_dir = save_checkpoint(model, Path(output_dir) / "final", cfg)
    print(f"  [trl_sft] final checkpoint saved to {final_dir}")
    return final_dir
