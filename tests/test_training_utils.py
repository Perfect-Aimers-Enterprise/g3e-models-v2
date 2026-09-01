"""
Tests the loss-masking ALGORITHM in g3e2/training_utils.py against a
lightweight fake tensor/processor — not real torch or a real Qwen model
(unavailable in this environment; see g3e2/FULL_TRAINING.md). This proves
the masking logic itself (prefix alignment, which positions get -100,
the safety assertion firing on mismatch) is correct, independent of
whether the real Qwen2.5-VL processor is available to run against.
"""
import pytest

from g3e2.training_utils import build_single_sample_inputs


class FakeTensor:
    """Just enough of torch.Tensor's API for build_single_sample_inputs to run."""
    def __init__(self, data):
        self.data = list(data)

    @property
    def shape(self):
        return (1, len(self.data))

    def clone(self):
        return FakeTensor(self.data)

    def __getitem__(self, key):
        # supports tensor[0, :n] (row+slice) and tensor[0] (row only, like torch)
        if isinstance(key, tuple):
            row, col_slice = key
            assert row == 0
            return FakeTensor(self.data[col_slice])
        assert key == 0
        return FakeTensor(self.data)

    def __setitem__(self, key, value):
        row, col_slice = key
        assert row == 0 or row == slice(None)
        indices = range(*col_slice.indices(len(self.data)))
        for i in indices:
            self.data[i] = value

    def equal(self, other):
        return self.data == other.data

    def to(self, device):
        return self


class FakeBatchDict(dict):
    def to(self, device):
        return self


class FakeProcessor:
    """
    Deterministic fake: `apply_chat_template` renders each message to a
    word-tokenized string; `__call__` "tokenizes" by splitting on spaces
    and mapping each word to an id via a shared vocabulary, so identical
    text always yields identical ids — exactly the property the real
    masking logic depends on.
    """
    def __init__(self):
        self.vocab = {}

    def _word_id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            if isinstance(m["content"], str):
                parts.append(f"{m['role']}: {m['content']}")
            else:
                text_blocks = [b["text"] for b in m["content"] if b["type"] == "text"]
                parts.append(f"{m['role']}: <image> " + " ".join(text_blocks))
        if add_generation_prompt:
            parts.append("assistant:")
        return " ".join(parts)

    def __call__(self, text, images, return_tensors="pt"):
        ids = [self._word_id(w) for w in text[0].split(" ")]
        return FakeBatchDict(input_ids=FakeTensor(ids))


def _messages(assistant_content="{\"state\": \"normal\"}"):
    from PIL import Image
    img = Image.new("RGB", (10, 10))
    return [
        {"role": "system", "content": "You are G3E-2."},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Analyze this."}]},
        {"role": "assistant", "content": assistant_content},
    ]


def test_prompt_prefix_is_masked_to_ignore_index():
    processor = FakeProcessor()
    messages = _messages()
    inputs = build_single_sample_inputs(processor, messages, device="cpu")

    prompt_text = processor.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    prompt_len = len(prompt_text.split(" "))

    labels = inputs["labels"].data
    assert labels[:prompt_len] == [-100] * prompt_len


def test_assistant_tokens_are_not_masked():
    processor = FakeProcessor()
    messages = _messages()
    inputs = build_single_sample_inputs(processor, messages, device="cpu")

    prompt_text = processor.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    prompt_len = len(prompt_text.split(" "))

    labels = inputs["labels"].data
    full_ids = inputs["input_ids"].data
    # everything after the prompt prefix must retain its real token id, not -100
    assert labels[prompt_len:] == full_ids[prompt_len:]
    assert -100 not in labels[prompt_len:]


def test_different_assistant_content_only_changes_labels_after_prompt():
    processor = FakeProcessor()
    m1 = _messages(assistant_content="{\"state\": \"normal\"}")
    m2 = _messages(assistant_content="{\"state\": \"hazard\"}")

    inputs1 = build_single_sample_inputs(processor, m1, device="cpu")
    inputs2 = build_single_sample_inputs(processor, m2, device="cpu")

    prompt_text = processor.apply_chat_template(m1[:2], tokenize=False, add_generation_prompt=True)
    prompt_len = len(prompt_text.split(" "))

    assert inputs1["labels"].data[:prompt_len] == inputs2["labels"].data[:prompt_len]


def test_mismatched_prefix_raises_instead_of_silently_masking_wrong_tokens():
    """
    If the tokenized prompt-only prefix ever doesn't match the full
    sequence's prefix (e.g. a future processor version changes chat
    template behavior), this must raise loudly, not silently train on a
    wrong mask.
    """
    class BrokenProcessor(FakeProcessor):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            base = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            if add_generation_prompt:
                return base + " EXTRA_TOKEN_THAT_BREAKS_ALIGNMENT"
            return base

    processor = BrokenProcessor()
    messages = _messages()
    with pytest.raises(RuntimeError, match="Prompt-prefix mismatch"):
        build_single_sample_inputs(processor, messages, device="cpu")
