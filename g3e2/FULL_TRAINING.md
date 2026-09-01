# `short_training_run` and `full_training`

Both stages are implemented, with three interchangeable training loop
backends — pick with `--trainer`:

```bash
python g3e2/train.py --stage short_training_run --trainer manual
python g3e2/train.py --stage short_training_run --trainer hf_trainer
python g3e2/train.py --stage short_training_run --trainer trl_sft
```

`--trainer` defaults to `training.trainer` in `config.yaml` (`manual`) if
not passed. All three:

- Reuse the exact same tested loss-masking (`training_utils.py:build_single_sample_inputs`)
  and model-loading (`train.py`'s staged functions) code.
- Process one sample at a time (never a padded batch), using gradient
  accumulation for effective batch size — see `training_utils.py`'s
  module docstring for why this is the one design choice all three share
  and none of them deviate from.
- Run the same evaluation (`trainers/common.py:evaluate_state_accuracy`
  — exact-match accuracy on `state`, plus a per-class breakdown so a good
  aggregate score can't hide a model that's specifically bad at
  `potential_threat`) and the same checkpoint saving/versioning
  (`trainers/common.py:save_checkpoint` — refuses to overwrite an
  existing non-empty checkpoint directory; bump `versioning.version` in
  `config.yaml` for a new run).

## What's actually verified vs. what isn't

**Verified, offline, without a GPU:**

- The loss-masking algorithm itself — `tests/test_training_utils.py`,
  4 tests, using a fake tensor/processor.
- The accuracy-aggregation math — `tests/test_trainers_common.py`,
  7 tests, including the specific scenario this metric exists to catch
  (96% aggregate accuracy while a rare class scores 0%).
- The checkpoint-immutability guard — same file, filesystem-only tests.

**NOT verified — no GPU, Qwen2.5-VL weights, or (for `trl_sft`) trl
install were available while building this:**

- That any of the three loops actually runs end-to-end without error.
- That `hf_trainer`'s and `trl_sft`'s framework-provided batching/
  checkpointing behaves as documented for THIS specific model/version
  combination.
- `trl_sft` in particular: `SFTConfig`'s field names
  (`dataset_kwargs={"skip_prepare_dataset": True}`) are correct for a
  recent trl version as of writing, but trl's API has moved before —
  check `python -c "from trl import SFTConfig; help(SFTConfig)"` against
  your installed version before your first real run. A wrong field name
  will raise `TypeError` at construction (fails safe, not silently wrong)
  but wastes a debugging cycle you can skip by checking first.

**Before your first real run of any of the three:** run
`--stage tiny_overfit_test` (not `short_training_run`) successfully first
— it exercises the identical model-loading and masking code path with a
much faster feedback loop.

## Choosing between the three

|                                   | `manual`                               | `hf_trainer`                             | `trl_sft`                                                                                     |
| --------------------------------- | -------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| Extra dependency                  | none                                   | none (transformers is already required)  | `trl`, `datasets`                                                                             |
| Logging/checkpointing             | you write it (already done, see below) | `transformers.Trainer`'s built-in        | `trl.SFTTrainer`'s built-in                                                                   |
| Easiest to audit line-by-line     | yes                                    | no (framework internals)                 | no (framework internals, plus trl's own VLM assumptions bypassed — see `trainers/trl_sft.py`) |
| `wandb`/`tensorboard` integration | you'd add it yourself                  | `report_to=[...]` in `TrainingArguments` | `report_to=[...]` in `SFTConfig`                                                              |

Default recommendation: start with `manual` for your first real run — it
has the fewest moving parts to debug if something's wrong, and its
`trainers/manual.py` is short enough to read in full in a few minutes.
Switch to `hf_trainer` once things work and you want `Trainer`'s more
polished checkpoint-resume and logging for a long unattended run.

## What each one does, in one paragraph

**`manual`** (`trainers/manual.py`): a straightforward Python `for` loop
— shuffle each epoch (seeded), forward + backward per sample, gradient
clip + optimizer step + LR scheduler step every `gradient_accumulation_steps`
samples, log/eval/checkpoint on the configured intervals, save a final
versioned checkpoint at the end.

**`hf_trainer`** (`trainers/hf_trainer.py`): wraps the same per-sample
inputs in a minimal `torch.utils.data.Dataset`-shaped object and a
`data_collator` that requires batch size 1 (asserted, not just assumed),
then hands both to `transformers.Trainer`. `remove_unused_columns=False`
is required and set — Trainer's default column-filtering would otherwise
silently strip `pixel_values`/`image_grid_thw` before they reach the
model, since it only recognizes columns matching the model's forward
signature by name.

**`trl_sft`** (`trainers/trl_sft.py`): the dataset trl sees is just a
column of row indices — a custom `data_collator` looks up the real sample
by index and builds it via the same tested `build_single_sample_inputs`,
deliberately bypassing trl's own multimodal chat-template/tokenization
pipeline (`dataset_kwargs={"skip_prepare_dataset": True}`), because that
pipeline's masking behavior for vision-language models isn't guaranteed
to match what this repo already tested and fixed a real bug around.

## Extending evaluation

`evaluate_state_accuracy` only checks `state`. Extending it to also check
`severity`/`recommended_action` (multi-field exact match, or a weighted
score) is a small change to `trainers/common.py`'s
`_aggregate_state_accuracy` and its caller — do this before relying on
`full_training`'s output for anything beyond "does it produce the right
`state` most of the time."

<!-- python g3e2/train.py --method lora --trainer manual -->

<!-- python g3e2/train.py --method lora --trainer hf_trainer -->
