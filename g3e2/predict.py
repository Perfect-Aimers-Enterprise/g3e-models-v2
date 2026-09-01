#!/usr/bin/env python3
"""
Run G3E-2 inference: image + G3E-1 detections -> semantic JSON.

The adapter can be a HF Hub repo id (the deployed case) OR a local
checkpoint directory — `--adapter` accepts either transparently:

    # Deployed to Hugging Face Hub:
    python g3e2/predict.py \\
        --adapter Godsave22/g3e2-lora-v1 \\
        --image ./samples/frame_001.jpg \\
        --detections ./samples/frame_001_detections.json

    # Private Hub repo — needs a token:
    python g3e2/predict.py \\
        --adapter Godsave22/g3e2-lora-v1 --hf-token $HF_TOKEN \\
        --image ./samples/frame_001.jpg --detections ./samples/frame_001_detections.json

    # Local checkpoint, not yet pushed to the Hub:
    python g3e2/predict.py \\
        --adapter ./checkpoints/g3e2/final \\
        --image ./samples/frame_001.jpg --detections ./samples/frame_001_detections.json

Usage — no detections available (G3E-2 reasons from the image alone; this
is a degraded mode — see README.md "Running predictions" for why
detections matter):

    python g3e2/predict.py --adapter Godsave22/g3e2-lora-v1 --image ./samples/frame_001.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g3e2.reasoner import G3E2Reasoner


def load_detections(path: str | None) -> list[dict]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True, help="HF Hub repo id (e.g. Godsave22/g3e2-lora-v1) or local adapter directory")
    parser.add_argument("--hf-token", default=None, help="Needed only for a private HF Hub adapter/base model repo; falls back to $HF_TOKEN")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detections", default=None, help="Path to a G3E-1-style detections JSON file")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    detections = load_detections(args.detections)
    if not detections:
        print(
            "[warning] no --detections provided — G3E-2 is reasoning from the image alone. "
            "It was trained expecting G3E-1's detections alongside the image; results without "
            "them are out of distribution. See README.md 'Running predictions'.",
            file=sys.stderr,
        )

    reasoner = G3E2Reasoner(
        adapter=args.adapter, base_model=args.base_model, hf_token=token, max_new_tokens=args.max_new_tokens
    )
    result = reasoner.predict(args.image, detections)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
