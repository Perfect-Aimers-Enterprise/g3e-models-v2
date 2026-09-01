"""
Console scripts, registered in pyproject.toml as `g3e-predict` and
`g3e-download` once this package is installed.
"""
from __future__ import annotations

import argparse
import sys


def predict_main() -> int:
    """`g3e-predict --image frame.jpg --output-dir ./events/evt_001`"""
    parser = argparse.ArgumentParser(description="Run the full G3E-1 + G3E-2 pipeline on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--event-id", default=None)
    args = parser.parse_args()

    from g3e_inference import G3EPipeline

    pipeline = G3EPipeline()
    event = pipeline.run(args.image, args.output_dir, event_id=args.event_id)
    print(event.model_dump_json(by_alias=True, indent=2))
    return 0


def download_main() -> int:
    """
    `g3e-download` — pre-fetches both models into the local Hugging Face
    cache ahead of time (e.g. as a build/deploy step), so the first real
    prediction in your app doesn't pay the download cost. Purely an
    optimization — G3EPipeline() downloads lazily on first use anyway if
    you skip this.
    """
    parser = argparse.ArgumentParser(description="Pre-download G3E-1 and G3E-2 model artifacts.")
    parser.add_argument("--skip-g3e1", action="store_true")
    parser.add_argument("--skip-g3e2", action="store_true")
    args = parser.parse_args()

    from g3e_inference.artifacts import resolve_weights_file, resolve_adapter_dir
    from g3e_inference.defaults import (
        DEFAULT_G3E1_HF_REPO, DEFAULT_G3E1_HF_FILENAME, DEFAULT_G3E2_HF_REPO,
        DEFAULT_BASE_MODEL, DEFAULT_HF_TOKEN,
    )

    if not args.skip_g3e1:
        print(f"Downloading G3E-1 weights from {DEFAULT_G3E1_HF_REPO} ...")
        path = resolve_weights_file(None, DEFAULT_G3E1_HF_REPO, DEFAULT_G3E1_HF_FILENAME, DEFAULT_HF_TOKEN)
        print(f"  -> {path}")

    if not args.skip_g3e2:
        print(f"Downloading G3E-2 base model {DEFAULT_BASE_MODEL} ...")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        AutoProcessor.from_pretrained(DEFAULT_BASE_MODEL, token=DEFAULT_HF_TOKEN)
        Qwen2_5_VLForConditionalGeneration.from_pretrained(DEFAULT_BASE_MODEL, token=DEFAULT_HF_TOKEN)
        print("  base model cached")

        print(f"Downloading G3E-2 adapter from {DEFAULT_G3E2_HF_REPO} ...")
        adapter_dir = resolve_adapter_dir(None, DEFAULT_G3E2_HF_REPO, DEFAULT_HF_TOKEN)
        print(f"  -> {adapter_dir}")

    print("\nAll requested artifacts are cached locally. Future predictions will not need network access.")
    return 0


if __name__ == "__main__":
    sys.exit(predict_main())
