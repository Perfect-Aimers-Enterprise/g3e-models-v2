"""
G3E2Reasoner — image + G3E-1 detections -> semantic JSON, via Qwen2.5-VL
+ a fine-tuned LoRA adapter.

Model resolution is local-first, Hugging Face Hub fallback — see
artifacts.py. G3E2Reasoner() with zero arguments resolves the default
adapter repo (defaults.py) and the default base model — exactly what
makes "pip install g3e-inference" then immediately calling
G3E2Reasoner().predict(...) "just work."

Lazy-loaded (model/adapter only load on the first .predict() call) so
constructing an instance never requires torch/transformers/peft to even
be importable yet — see pipeline.py and tests/test_pipeline.py, which
rely on this to test orchestration without a GPU.
"""
from __future__ import annotations

import json

from g3e_inference.schemas import G3E2Output
from g3e_inference.artifacts import resolve_adapter_dir
from g3e_inference.defaults import DEFAULT_BASE_MODEL, DEFAULT_G3E2_HF_REPO, DEFAULT_HF_TOKEN

SYSTEM_PROMPT = (
    "You are G3E-2, a visual security-reasoning system. You are given an "
    "image and a list of objects already detected in it by G3E-1 (class, "
    "confidence, bounding box in pixel [x1,y1,x2,y2] coordinates). "
    "Analyze the scene and respond with ONLY a single JSON object with "
    'exactly these keys: "state", "severity", "description", "reason", '
    '"recommended_action". Do not include any text before or after the '
    "JSON object."
)


def build_user_prompt(detections: list[dict]) -> str:
    detections_json = json.dumps(detections, indent=2)
    return (
        "Analyze the security status of this image using the detected "
        f"objects provided by G3E-1.\n\nDetected objects:\n{detections_json}"
    )


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
    Usage — zero-config (resolves the default HF repo, local cache after first run):

        reasoner = G3E2Reasoner()

    Usage — your own local adapter checkpoint takes priority if present:

        reasoner = G3E2Reasoner(adapter="./checkpoints/g3e2/final")
        # falls back to the default HF repo automatically if that directory doesn't exist

    Usage — a different / private HF repo:

        reasoner = G3E2Reasoner(hf_repo_id="YourOrg/g3e2-lora-v2", hf_token="hf_...")
    """

    def __init__(
        self,
        adapter: str | None = None,
        hf_repo_id: str = DEFAULT_G3E2_HF_REPO,
        base_model: str = DEFAULT_BASE_MODEL,
        hf_token: str | None = DEFAULT_HF_TOKEN,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
    ):
        self.adapter = adapter
        self.hf_repo_id = hf_repo_id
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

        adapter_dir = resolve_adapter_dir(self.adapter, self.hf_repo_id, self.hf_token)

        dtype = getattr(torch, self.torch_dtype)
        self._processor = AutoProcessor.from_pretrained(self.base_model_id, token=self.hf_token)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_id, torch_dtype=dtype, device_map="auto", token=self.hf_token
        )
        self._model = PeftModel.from_pretrained(base_model, adapter_dir)
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
