#!/usr/bin/env python3
"""
Setup script to download and cache all Hugging Face models used by the app.

✓ Reads model IDs and local paths from config/settings.py (single source of truth)
✓ Saves each model to models/saved_models/<name>/ to match ModelLoader expectations
✓ Works offline after first run (uses the saved directories)
✓ CLI:
    python setup_models.py                # download all models (default)
    python setup_models.py --only scibert_ade --force
    python setup_models.py --list
"""


import argparse
import os
import shutil
import sys
from typing import Dict, Tuple

import torch

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification,
    )
except Exception as e:
    print("❌ transformers is required. `pip install transformers`")
    raise

# Pull everything from config so names/IDs/paths stay consistent with the README & app
try:
    from config.settings import MODEL_CONFIG, MODELS_DIR, INFERENCE_CONFIG
except Exception as e:
    print("❌ Could not import config.settings. Run this from the project root.")
    raise

HF_TOKEN = (INFERENCE_CONFIG.get("hf_api_key") or
            os.getenv("HUGGINGFACEHUB_API_TOKEN") or None)


def _need_download(path: str) -> bool:
    """Decide whether a local model directory looks complete enough to skip download."""
    if not os.path.isdir(path):
        return True
    expected = {"config.json"}
    present = set(os.listdir(path))
    has_model = ("pytorch_model.bin" in present) or ("model.safetensors" in present)
    has_tok = any(t in present for t in ["tokenizer.json", "vocab.txt", "spiece.model", "merges.txt"])
    return not (expected.issubset(present) and has_model and has_tok)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _download_and_save_token_cls(hf_id: str, save_dir: str, *, num_labels: int = None, id2label: Dict[int, str] = None):
    tok = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
    if num_labels is not None and id2label is not None:
        mdl = AutoModelForTokenClassification.from_pretrained(
            hf_id, token=HF_TOKEN, num_labels=num_labels, id2label=id2label
        )
    else:
        mdl = AutoModelForTokenClassification.from_pretrained(hf_id, token=HF_TOKEN)
    # quick warmup to materialize fast tokenizer files locally
    tok("warmup", return_tensors="pt")
    tok.save_pretrained(save_dir)
    mdl.save_pretrained(save_dir)


def _download_and_save_seq_cls(hf_id: str, save_dir: str):
    tok = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
    mdl = AutoModelForSequenceClassification.from_pretrained(hf_id, token=HF_TOKEN)
    tok("warmup", return_tensors="pt")
    tok.save_pretrained(save_dir)
    mdl.save_pretrained(save_dir)


def _download_and_save_base(hf_id: str, save_dir: str):
    tok = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN)
    mdl = AutoModel.from_pretrained(hf_id, token=HF_TOKEN)
    tok("warmup", return_tensors="pt")
    tok.save_pretrained(save_dir)
    mdl.save_pretrained(save_dir)


def _specs_from_config() -> Dict[str, Dict]:
    """Build a normalized spec dict from MODEL_CONFIG."""
    return {
        "biomedical_ner": {
            "hf_id": MODEL_CONFIG["biomedical_ner"]["hf_model_id"],
            "save_dir": MODEL_CONFIG["biomedical_ner"]["model_name"],
            "kind": "token",
            "extra": {}
        },
        "clinical_bert": {
            "hf_id": MODEL_CONFIG["clinical_bert"]["hf_model_id"],
            "save_dir": MODEL_CONFIG["clinical_bert"]["model_name"],
            "kind": "base",
            "extra": {}
        },
        "scibert_ade": {
            "hf_id": MODEL_CONFIG["scibert_ade"]["hf_model_id"],
            "save_dir": MODEL_CONFIG["scibert_ade"]["model_name"],
            "kind": "token",
            "extra": {
                "num_labels": MODEL_CONFIG["scibert_ade"].get("num_labels"),
                "id2label": MODEL_CONFIG["scibert_ade"].get("id2label"),
            }
        },
        "relation_extraction": {
            "hf_id": MODEL_CONFIG["relation_extraction"]["hf_model_id"],
            "save_dir": MODEL_CONFIG["relation_extraction"]["model_name"],
            "kind": "seqcls",
            "extra": {}
        },
    }


def download_one(name: str, force: bool = False) -> Tuple[bool, str]:
    """Download a single model by logical name from config. Returns (ok, message)."""
    specs = _specs_from_config()
    if name not in specs:
        return False, f"Unknown model key: {name}"

    hf_id = specs[name]["hf_id"]
    save_dir = specs[name]["save_dir"]
    kind = specs[name]["kind"]
    extra = specs[name]["extra"]

    _ensure_dir(MODELS_DIR)
    if force and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    _ensure_dir(save_dir)

    if not force and not _need_download(save_dir):
        return True, f"✓ {name}: already present at {save_dir}"

    try:
        print(f"↓ Downloading {name} from {hf_id} → {save_dir}")
        if kind == "token":
            _download_and_save_token_cls(hf_id, save_dir, **extra)
        elif kind == "seqcls":
            _download_and_save_seq_cls(hf_id, save_dir)
        elif kind == "base":
            _download_and_save_base(hf_id, save_dir)
        else:
            return False, f"Unsupported kind '{kind}' for {name}"
        return True, f"✓ {name}: saved to {save_dir}"
    except Exception as e:
        return False, f"✗ {name}: {e}"


def download_all(force: bool = False) -> int:
    specs = _specs_from_config()
    print("=== Pharmacovigilance NLP — Model Setup ===")
    print(f"Models root: {MODELS_DIR}")
    if HF_TOKEN:
        print("Auth: using Hugging Face token from env/config")
    else:
        print("Auth: no token detected (only public models will work)")

    failures = 0
    for name in specs.keys():
        ok, msg = download_one(name, force=force)
        print(msg)
        if not ok:
            failures += 1
    print("\nSummary:")
    if failures == 0:
        print("✅ All models ready. You can now run:  streamlit run app.py")
    else:
        print(f"⚠️  {failures} model(s) failed. Re-run with --force after fixing the issue.")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description="Download and cache models for the Pharmacovigilance NLP app.")
    parser.add_argument("--only", nargs="*", help="Download only these models (space-separated): biomedical_ner clinical_bert scibert_ade relation_extraction")
    parser.add_argument("--force", action="store_true", help="Redownload even if a local copy exists.")
    parser.add_argument("--list", action="store_true", help="List model IDs and target folders, then exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    specs = _specs_from_config()

    if args.list:
        print("Available models:")
        for k, v in specs.items():
            print(f"- {k:20s}  hf_id={v['hf_id']}  →  {v['save_dir']}")
        return 0

    if args.only:
        failures = 0
        for name in args.only:
            ok, msg = download_one(name, force=args.force)
            print(msg)
            failures += 0 if ok else 1
        print("\nDone.")
        return failures

    # Default: download all
    return download_all(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
