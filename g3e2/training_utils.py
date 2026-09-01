"""
Shared per-sample input building for G3E-2 training — used by every
smoke-test stage in train.py (batch_forward_test, batch_backward_test,
tiny_overfit_test) and documented as the required building block for a
real full_training loop (see FULL_TRAINING.md).

WHY PER-SAMPLE (batch size 1), NOT A PADDED BATCH:
Correctly masking a batch of multimodal sequences (so that only the
assistant's JSON contributes to the loss, not the system prompt or the
G3E-1 detections list) requires the loss mask to line up exactly with
each sequence's own padding and image-token layout. Getting that alignment
subtly wrong is easy and would fail silently — the model would still
train and the loss would still go down, it would just be learning to
predict the wrong tokens. Processing one sample at a time sidesteps this
entirely: no padding exists, so there is nothing to get misaligned.
Multiple samples' worth of gradient still gets combined via gradient
accumulation (config.yaml's training.gradient_accumulation_steps) — this
trades a bit of throughput for a masking implementation that is much
easier to reason about, and had already gone wrong once (see below)
before this module existed.

THE BUG THIS FIXES:
An earlier version of train.py's tiny_overfit_test used
`inputs["labels"] = inputs["input_ids"].clone()` — training on the ENTIRE
sequence, including the system prompt and the detections list, not just
the assistant's answer. This masks it correctly instead: only the
assistant's JSON tokens carry a real label; everything before that is set
to -100 (the value transformers' loss functions treat as "ignore").

HOW THE MASKING WORKS:
The full conversation and the prompt-only conversation (system + user,
with add_generation_prompt=True) are both tokenized. Because the image and
every token before the assistant's turn are IDENTICAL between the two
(same system prompt, same user content, same image), the prompt-only
tokenization's length tells us exactly how many leading tokens in the full
sequence to mask. `build_single_sample_inputs` asserts this prefix
actually matches byte-for-byte before trusting the split — if a future
transformers/Qwen version ever changes chat-template behavior such that
this assumption breaks, this raises loudly instead of silently training
on a wrong mask.
"""
from __future__ import annotations


def build_single_sample_inputs(processor, messages: list[dict], device) -> dict:
    """
    `messages`: the full 3-turn conversation — [system, user (with image),
    assistant] — exactly what `g3e2.dataset.record_to_messages()` and
    `G3E2Dataset.__getitem__` already produce. Returns a dict of tensors
    ready to pass as `model(**inputs)`, including a correctly masked
    `labels` tensor.
    """
    system_msg, user_msg, assistant_msg = messages[0], messages[1], messages[2]

    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(
        [system_msg, user_msg], tokenize=False, add_generation_prompt=True
    )
    image = [b["image"] for b in user_msg["content"] if b["type"] == "image"][0]

    full_inputs = processor(text=[full_text], images=[image], return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_ids = full_inputs["input_ids"]

    # Safety check described in the module docstring — if this ever
    # fails, STOP and investigate rather than silently masking the wrong
    # tokens; do not remove this assertion to "make training run."
    if not full_ids[0, :prompt_len].equal(prompt_inputs["input_ids"][0]):
        raise RuntimeError(
            "Prompt-prefix mismatch while building training labels — the tokenized prefix of "
            "the full conversation does not match the tokenized prompt-only conversation. "
            "This means the loss-masking assumption in training_utils.py no longer holds for "
            "the installed transformers/Qwen2.5-VL version. Do not train until this is fixed — "
            "a silent mismatch here would train the model on the wrong tokens."
        )

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels

    return {k: v.to(device) for k, v in full_inputs.items()}
