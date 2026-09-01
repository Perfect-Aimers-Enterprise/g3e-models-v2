"""
G3E2Reasoner — the reusable "image + detections -> semantic JSON" call,
shared by `g3e2/predict.py` (standalone CLI) and `inference/pipeline.py`
(full G3E-1 -> G3E-2 -> event pipeline), so both go through identical
model-loading and output-parsing logic.

Model/adapter loading is LAZY (happens on first `.predict()` call, not in
`__init__`) specifically so a `G3E2Reasoner` instance can be constructed
and passed around (e.g. injected into `inference/pipeline.py`) in tests
and other lightweight contexts without requiring torch/transformers/peft
to even be installed, let alone a GPU.

Adapter loading — local dir OR Hugging Face Hub:
`peft.PeftModel.from_pretrained()` transparently accepts either a local
directory path or a HF Hub repo id (e.g. "Godsave22/g3e2-lora-v1") for its
`adapter` argument — it downloads from the Hub automatically if the string
isn't a local path. This class doesn't need separate code paths for the
two cases; it only needs to pass `token=` through for a private repo.
"""
from __future__ import annotations

import json

from shared.schemas import G3E2Output
from g3e2.dataset import SYSTEM_PROMPT, build_user_prompt  # single source of truth — see that module's docstring


def parse_model_output(raw_text: str) -> dict:
    """
    The model is trained to output ONLY a JSON object. Real models
    occasionally still wrap it in stray whitespace/newlines or (rarely, if
    undertrained) a code fence — this strips the common cases before
    giving up, rather than silently returning garbage to the caller.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model output was not valid JSON after cleanup: {exc}\nRaw output: {raw_text!r}"
        ) from exc


class G3E2Reasoner:
    """
    Usage — adapter hosted on HF Hub (the deployed case):

        reasoner = G3E2Reasoner(adapter="Godsave22/g3e2-lora-v1")
        output = reasoner.predict("frame.jpg", detections)

    Usage — private HF Hub repo:

        reasoner = G3E2Reasoner(adapter="Godsave22/g3e2-lora-v1", hf_token=os.environ["HF_TOKEN"])

    Usage — local checkpoint directory (before/instead of deploying to the Hub):

        reasoner = G3E2Reasoner(adapter="./checkpoints/g3e2/final")
    """

    def __init__(
        self,
        adapter: str,
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        hf_token: str | None = None,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
    ):
        self.adapter = adapter
        self.base_model_id = base_model
        self.hf_token = hf_token
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel

        dtype = getattr(torch, self.torch_dtype)
        self._processor = AutoProcessor.from_pretrained(self.base_model_id, token=self.hf_token)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_id, torch_dtype=dtype, device_map="auto", token=self.hf_token
        )
        # `self.adapter` may be a local directory OR a HF Hub repo id —
        # PeftModel resolves that transparently; `token` only matters for
        # a private Hub repo and is harmless to pass for a local path.
        self._model = PeftModel.from_pretrained(base_model, self.adapter, token=self.hf_token)
        self._model.eval()

    def build_messages(self, image_path: str, detections: list[dict]) -> list[dict]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": build_user_prompt(detections)},
                ],
            },
        ]

    def predict(self, image_path: str, detections: list[dict]) -> G3E2Output:
        import torch

        self._ensure_loaded()

        messages = self.build_messages(image_path, detections)
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = [b["image"] for b in messages[1]["content"] if b["type"] == "image"][0]
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]

        return G3E2Output(**parse_model_output(raw_text))
