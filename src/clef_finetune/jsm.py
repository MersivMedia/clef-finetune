"""Load Cloudflare's ``joint_schema_model.py`` (the only code released with Clef).

Resolution order:
1. ``joint_schema_model.py`` inside a local release directory (if given),
2. the file from the Hugging Face repo at a *pinned* revision,
3. the vendored, byte-for-byte unmodified copy in ``clef_finetune/_vendor``
   (Apache-2.0, Copyright Cloudflare, Inc. -- see ``_vendor/NOTICE``).

We never edit Cloudflare's file. Everything training-specific (dtype boundary,
losses, LoRA) lives in this package and calls into it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

# Pinned Hugging Face revisions (commit SHAs) checked on 2026-10-02.
PINNED_REVISIONS = {
    "Cloudflare/clef": "2f3de3dd85f379784083b0814d997ab627200f0c",
    "Cloudflare/clef-flash": "17f0b0ad64efb65d273590632833508766b2aae6",
}
VENDORED_PATH = Path(__file__).parent / "_vendor" / "joint_schema_model.py"
VENDORED_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"

_CACHE: dict[str, ModuleType] = {}


def _import_file(path: Path) -> ModuleType:
    key = str(path.resolve())
    if key in _CACHE:
        return _CACHE[key]
    name = "joint_schema_model_" + hashlib.sha1(key.encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    _CACHE[key] = module
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsm(
    release_dir: str | Path | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
) -> ModuleType:
    """Return the joint_schema_model module.

    Set ``CLEF_FINETUNE_JSM=vendored`` to force the vendored copy (used in CI).
    """
    if os.environ.get("CLEF_FINETUNE_JSM") != "vendored":
        if release_dir is not None:
            candidate = Path(release_dir) / "joint_schema_model.py"
            if candidate.is_file():
                return _import_file(candidate)
        if repo_id is not None:
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id,
                    "joint_schema_model.py",
                    revision=revision or PINNED_REVISIONS.get(repo_id),
                )
                return _import_file(Path(path))
            except Exception as error:  # offline etc.
                print(f"[clef-finetune] hub load of joint_schema_model.py failed ({error}); using vendored copy")
    return _import_file(VENDORED_PATH)
