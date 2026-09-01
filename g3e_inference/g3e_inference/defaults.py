"""
Default model locations — what G3E1Detector()/G3E2Reasoner()/
G3EPipeline() resolve to when called with no arguments at all.

**These repo id placeholders MUST be updated to your actual published HF
Hub repos before this package is used for real.** They're read from
environment variables first specifically so a deploying app can override
them without editing this file or rebuilding the package — set these in
whatever environment your app runs in (a .env file, your container's
env, your deployment platform's secrets/config panel):

    G3E1_HF_REPO=YourOrg/g3e1-yolo-v1
    G3E1_HF_FILENAME=best.pt
    G3E2_HF_REPO=YourOrg/g3e2-lora-v1
    G3E2_BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct   # only override if you fine-tuned a different base
"""
from __future__ import annotations

import os

DEFAULT_BASE_MODEL = os.environ.get("G3E2_BASE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

# PLACEHOLDER — replace with your real published G3E-1 weights repo, or
# set G3E1_HF_REPO / G3E1_HF_FILENAME in the environment.
DEFAULT_G3E1_HF_REPO = os.environ.get("G3E1_HF_REPO", "Godsave22/g3e1-yolo")
DEFAULT_G3E1_HF_FILENAME = os.environ.get("G3E1_HF_FILENAME", "best.pt")

# PLACEHOLDER — replace with your real published G3E-2 LoRA adapter repo,
# or set G3E2_HF_REPO in the environment.
DEFAULT_G3E2_HF_REPO = os.environ.get("G3E2_HF_REPO", "Godsave22/g3e2-lora-v1")

# Optional — only needed for private Hub repos. Same convention as
# g3e-data-engine / g3e-models: read from env, never hardcoded.
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN")
