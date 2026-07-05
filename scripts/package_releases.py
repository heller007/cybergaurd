#!/usr/bin/env python3
"""Package deploy-only weights for Google Drive (no training checkpoints)."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASES_DIR = os.path.join(ROOT, "releases")


def add_path(tar: tarfile.TarFile, src: str, arcname: str) -> bool:
    if not os.path.exists(src):
        return False
    if os.path.isdir(src):
        for dirpath, _, filenames in os.walk(src):
            for name in filenames:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, src)
                tar.add(path, arcname=os.path.join(arcname, rel))
    else:
        tar.add(src, arcname=arcname)
    return True


def muril_dir() -> str:
    for path in (
        os.path.join(ROOT, "releases/muril_category_9class"),
        os.path.join(ROOT, "model/category_9class"),
    ):
        if os.path.isfile(os.path.join(path, "best_model.pt")):
            return path
    return ""


def lora_adapter_dir() -> str:
    path = os.path.join(
        ROOT, "releases/llm_lora_qwen05b_r64/checkpoints/best_adapter"
    )
    if os.path.isfile(os.path.join(path, "adapter_model.safetensors")):
        return path
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.join(RELEASES_DIR, f"cybergaurd_models_{date.today()}.tar.gz"),
    )
    args = parser.parse_args()
    os.makedirs(RELEASES_DIR, exist_ok=True)

    bundles = [
        ("muril_category_9class", muril_dir(), "MuRIL deploy bundle"),
        (
            "llm_lora_qwen05b_r64/best_adapter",
            lora_adapter_dir(),
            "LoRA adapter only (Qwen2.5-0.5B r=64)",
        ),
    ]

    manifest = {"generated": str(date.today()), "models": []}
    with tarfile.open(args.output, "w:gz") as tar:
        for arcname, path, desc in bundles:
            ok = bool(path) and add_path(tar, path, arcname)
            manifest["models"].append(
                {
                    "id": arcname,
                    "description": desc,
                    "included": ok,
                    "source": path or None,
                }
            )

    out_manifest = os.path.join(RELEASES_DIR, "MANIFEST.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created {args.output}")
    for m in manifest["models"]:
        print(f"  {m['id']}: {'ok' if m['included'] else 'MISSING'}")


if __name__ == "__main__":
    main()
