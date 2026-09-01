"""
Shared evaluation + checkpoint saving used by all three training loop
implementations (manual, transformers.Trainer, trl SFTTrainer) — so
results and saved checkpoints are directly comparable no matter which
loop produced them.

The state-comparison/aggregation logic (`_aggregate_state_accuracy`) is
split out from the model-calling loop (`evaluate_state_accuracy`)
specifically so it's testable without torch — see
tests/test_trainers_common.py. The model-calling half genuinely needs a
real model + GPU and can't be verified in an environment without one.
"""
from __future__ import annotations

import json
from pathlib import Path


def _aggregate_state_accuracy(predicted_states: list, expected_states: list) -> dict:
    """
    Pure aggregation, no model/tensor involvement — `predicted_states[i]`
    may be `None` (the model's output didn't parse as JSON at all, which
    counts as wrong, not excluded from the denominator).

    Returns per-class accuracy alongside the overall number on purpose: an
    aggregate accuracy can look fine while the model is specifically bad
    at rare classes like `potential_threat` — the exact failure mode
    class balancing (shared/balancing.py) exists to reduce but can't fully
    eliminate. Always check both before deciding a checkpoint is good.
    """
    if len(predicted_states) != len(expected_states):
        raise ValueError("predicted_states and expected_states must be the same length")

    correct = 0
    total = len(expected_states)
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for predicted, expected in zip(predicted_states, expected_states):
        per_class_total[expected] = per_class_total.get(expected, 0) + 1
        if predicted == expected:
            correct += 1
            per_class_correct[expected] = per_class_correct.get(expected, 0) + 1

    accuracy = correct / total if total else 0.0
    per_class_accuracy = {
        cls: per_class_correct.get(cls, 0) / count for cls, count in per_class_total.items()
    }
    return {"accuracy": accuracy, "correct": correct, "total": total, "per_class_accuracy": per_class_accuracy}


def evaluate_state_accuracy(model, processor, val_samples: list, max_samples: int | None = None) -> dict:
    """
    Runs real generation against `val_samples` and reports exact-match
    accuracy on the `state` field (see FULL_TRAINING.md — the field that
    matters most for G3E's use case) plus per-class breakdown.

    NEEDS a real model/GPU — this function itself was not runnable in the
    environment this repo was built in (no GPU, no Qwen weights
    available). The aggregation math it delegates to
    (`_aggregate_state_accuracy`) IS verified — see
    tests/test_trainers_common.py. Treat this function's control flow as
    reviewed-but-unexecuted until you run it for real.
    """
    import torch
    from g3e2.reasoner import parse_model_output

    model.eval()
    samples = val_samples[:max_samples] if max_samples else val_samples

    predicted_states = []
    expected_states = []

    with torch.no_grad():
        for messages in samples:
            prompt_messages = messages[:2]  # system + user, no assistant
            text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            image = [b["image"] for b in prompt_messages[1]["content"] if b["type"] == "image"][0]
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            raw_text = processor.batch_decode(generated, skip_special_tokens=True)[0]

            expected = json.loads(messages[2]["content"])
            expected_states.append(expected.get("state"))

            try:
                predicted = parse_model_output(raw_text)
                predicted_states.append(predicted.get("state"))
            except ValueError:
                predicted_states.append(None)  # unparseable output -> counts as wrong

    model.train()
    return _aggregate_state_accuracy(predicted_states, expected_states)


def save_checkpoint(model, output_dir, cfg: dict) -> Path:
    """
    Saves ONLY the LoRA adapter — `model.save_pretrained()` on a PEFT
    model does this automatically; the (much larger) base model is never
    re-saved. Writes a version-info JSON alongside it per spec section 24.

    Refuses to overwrite an existing non-empty directory — versions are
    immutable, matching g3e-data-engine's release-export policy; bump
    `versioning.version` in config.yaml instead of re-running into the
    same output_dir. This check IS testable offline (pure filesystem
    logic) — see tests/test_trainers_common.py — unlike the actual
    `model.save_pretrained()` call, which needs a real PEFT model.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"{output_dir} already exists and is non-empty — versions are immutable. "
            "Bump versioning.version in g3e2/config.yaml before running again."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    version_info = {
        "model": f"{cfg['versioning']['model_name']}-{cfg['versioning']['version']}",
        "base_model": cfg["model"]["base_model"],
        "method": cfg["training"].get("method", "lora"),
        "lora_config": cfg["lora"],
        "training_config": {k: v for k, v in cfg["training"].items() if k != "stages"},
    }
    (output_dir / "g3e_version_info.json").write_text(json.dumps(version_info, indent=2))
    return output_dir
