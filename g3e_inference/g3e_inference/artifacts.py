"""
Artifact resolution policy: local if available, Hugging Face Hub
otherwise. This is the ONE place that decision gets made — detector.py
and reasoner.py both call into here rather than each implementing their
own local/remote fallback, so the behavior is identical and predictable
for both models.

Precedence, in order:

1. An explicit local path was given AND it actually exists on disk ->
   use it directly. No network call is made, even if HF arguments were
   also provided — an explicit local path always wins.
2. Otherwise, resolve from Hugging Face Hub. `huggingface_hub` itself
   caches downloads under `~/.cache/huggingface/hub` (or `$HF_HOME` if
   set) and only hits the network if the file/repo isn't already cached
   there — so in practice, "download from HF" only actually touches the
   network on the very first call anywhere on a machine; every call after
   that resolves from that local cache automatically. This is what makes
   "install the library, first prediction downloads the model, every
   prediction after that runs fully offline" true without this module
   needing to reimplement a caching layer itself.

Two resolution shapes are needed because the two models package
differently:
- G3E-1 is a single weights file (best.pt) -> resolve_weights_file.
- G3E-2's LoRA adapter is a small directory of files (adapter_config.json,
  adapter_model.safetensors, ...) -> resolve_adapter_dir.
"""
from __future__ import annotations

from pathlib import Path


def resolve_weights_file(
    local_path: str | None,
    hf_repo_id: str,
    hf_filename: str,
    hf_token: str | None = None,
) -> str:
    """Returns a local filesystem path to a single weights file, per the
    module docstring's precedence. Never returns a path that doesn't
    actually exist on disk at the moment it returns."""
    if local_path and Path(local_path).is_file():
        return local_path

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, token=hf_token)


def resolve_adapter_dir(
    local_path: str | None,
    hf_repo_id: str,
    hf_token: str | None = None,
) -> str:
    """Returns a local directory path holding a full PEFT adapter (or any
    multi-file HF repo), per the module docstring's precedence."""
    if local_path and Path(local_path).is_dir():
        return local_path

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=hf_repo_id, token=hf_token)
