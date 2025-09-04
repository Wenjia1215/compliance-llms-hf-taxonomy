#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — Strong-signal candidate collection for Compliance/Regulation LLMs.

What this script does:
- Queries Hugging Face Models using ONLY high-precision compliance/regulation keywords:
  framework names (GDPR, HIPAA, SOC 2, ISO 27001, PCI DSS, SOX, FedRAMP, CMMC,
  NIST SP 800-53/207, NIST CSF, CIS Controls, CCPA/CPRA) + a few intent words
  ("compliance", "audit", "attestation", "control mapping", "evidence", "policy").
- Uses HfApi.list_models (works across hub versions) and iterates over the returned iterable.
- Captures metadata + README text and writes rows incrementally to CSV.
- Checkpoints processed repo ids (resume-safe).
- Prints huggingface_hub version at start for reproducibility.

This script DOES NOT filter out noise yet (no CV/OCR exclusion, no family merge).
It only builds a strong-signal universe we’ll later gate and dedupe in subsequent steps.

Usage examples:
  export HF_TOKEN=hf_xxx                # optional but recommended for rate limits
  python s1_candidatesCollector.py --out candidates_step1_frameworks.csv --sleep 0.2
  # Ultra-conservative (framework-only) queries:
  python s1_candidatesCollector.py --out candidates_step1_frameworks.csv --sleep 0.2 \
    --queries "GDPR" "HIPAA" "SOC 2" "ISO 27001" "PCI DSS" "SOX" "FedRAMP" "CMMC" \
              "NIST SP 800-53" "NIST SP 800-207" "NIST CSF" "CIS Controls" "CIS Controls V8" "CCPA" "CPRA"
"""

import os
import sys
import time
import csv
import json
import argparse
from datetime import datetime

# ----------------------------
# Imports & version print
# ----------------------------
try:
    from huggingface_hub import HfApi, __version__ as HFHUB_VER
except Exception as e:
    print("ERROR: huggingface_hub not installed. Install with: pip install -U huggingface_hub")
    raise

print(f"[Init] huggingface_hub={HFHUB_VER}")

# ----------------------------
# High-precision query seeds
# ----------------------------

FRAMEWORK_QUERIES = [
    # Normalize common variants; HF search is string-based
    "GDPR", "HIPAA", "SOC 2", "SOC2", "ISO 27001", "ISO/IEC 27001",
    "PCI DSS", "PCI-DSS", "SOX", "Sarbanes-Oxley", "FedRAMP",
    "CMMC", "CMMC 2.0",
    "NIST SP 800-53", "NIST 800-53", "SP 800-53", "800-53",
    "NIST SP 800-207", "NIST 800-207", "SP 800-207", "Zero Trust 800-207",
    "NIST CSF", "CSF 2.0", "CSF 1.1",
    "CIS Controls", "CIS Controls V8",
    "CCPA", "CPRA",
]

INTENT_QUERIES = [
    # keep these tight (high precision)
    "compliance", "audit", "attestation", "control mapping", "evidence", "policy"
]


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_get_model_card(api: HfApi, repo_id: str) -> str:
    """
    Try to fetch README text. If not available, return empty string.
    """
    try:
        card = api.get_model_card(repo_id)
        if card is not None and hasattr(card, "text") and card.text:
            return card.text
    except Exception:
        pass
    return ""


def model_row_from_info(api: HfApi, m, query: str) -> dict:
    """
    Convert ModelInfo to our row. Be defensive about attribute names across hub versions.
    """
    repo_id = getattr(m, "id", None) or getattr(m, "modelId", None)
    if not repo_id:
        author = getattr(m, "author", "") or getattr(m, "owner", "")
        name = getattr(m, "name", "")
        repo_id = f"{author}/{name}".strip("/")

    owner, name = "", ""
    if repo_id and "/" in repo_id:
        owner, name = repo_id.split("/", 1)

    tags = getattr(m, "tags", None)
    pipeline_tag = getattr(m, "pipeline_tag", None)
    downloads = getattr(m, "downloads", None)
    likes = getattr(m, "likes", None)
    created_at = getattr(m, "created_at", None)
    last_modified = getattr(m, "lastModified", None) or getattr(m, "last_modified", None)
    license_str = getattr(m, "license", None)

    # Fetch README once here for later gates (contentfulness, evidence, etc.)
    readme_text = safe_get_model_card(api, repo_id)

    row = {
        "id": repo_id,
        "owner": owner,
        "name": name,
        "pipeline_tag": pipeline_tag,
        "tags": json.dumps(tags or []),
        "downloads": downloads,
        "likes": likes,
        "created_at": str(created_at) if created_at is not None else "",
        "last_modified": str(last_modified) if last_modified is not None else "",
        "license": license_str or "",
        "hf_url": f"https://huggingface.co/{repo_id}" if repo_id else "",
        "source_query": query,
        "collection_ts_utc": now_utc_iso(),
        "readme_text": readme_text,
        "why_included": "strong-signal query hit",
    }
    return row


def write_header_if_needed(out_path: str, fieldnames: list):
    exists = os.path.exists(out_path)
    if not exists:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def load_checkpoint_ids(ckpt_path: str) -> set:
    ids = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.add(s)
    return ids


def append_checkpoint_id(ckpt_path: str, repo_id: str):
    with open(ckpt_path, "a", encoding="utf-8") as f:
        f.write(repo_id + "\n")


def make_iterator(api: HfApi, q: str):
    """
    Cross-version friendly iterator over models matching query q.
    Try the richest signature first, then fall back.
    NOTE: list_models in recent huggingface_hub returns an iterable/generator that
          handles paging internally. If your version returns a finite list only,
          consider upgrading huggingface_hub.
    """
    try:
        return api.list_models(search=q, sort=None, full=True, cardData=True)
    except TypeError:
        try:
            return api.list_models(search=q, sort=None, full=True)
        except TypeError:
            return api.list_models(search=q, sort=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV path (appends if exists)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep seconds between items (rate-limit friendly)")
    ap.add_argument("--max_per_query", type=int, default=0, help="Optional cap per query (0 = no cap)")
    ap.add_argument("--queries", type=str, nargs="*", default=[],
                    help="Override default queries (space-separated). If empty, use FRAMEWORK + INTENT lists.")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to processed-ids checkpoint file (default <out>.ckpt)")
    args = ap.parse_args()

    # Prepare queries
    if args.queries:
        queries = args.queries
    else:
        queries = FRAMEWORK_QUERIES + INTENT_QUERIES

    # HF API
    token = os.environ.get("HF_TOKEN", None)
    api = HfApi(token=token)

    # IO
    fieldnames = [
        "id","owner","name","pipeline_tag","tags","downloads","likes",
        "created_at","last_modified","license","hf_url",
        "source_query","collection_ts_utc","readme_text","why_included"
    ]
    write_header_if_needed(args.out, fieldnames)
    ckpt_path = args.checkpoint or (args.out + ".ckpt")
    seen = load_checkpoint_ids(ckpt_path)

    total_new = 0
    for q in queries:
        q_new = 0
        print(f"[{now_utc_iso()}] Query: {q}")
        iterator = make_iterator(api, q)

        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)

            for m in iterator:
                # Identify repo id robustly
                repo_id = getattr(m, "id", None) or getattr(m, "modelId", None)
                if not repo_id:
                    author = getattr(m, "author", "") or getattr(m, "owner", "")
                    name = getattr(m, "name", "")
                    repo_id = f"{author}/{name}".strip("/")

                if not repo_id:
                    continue

                if repo_id in seen:
                    continue

                row = model_row_from_info(api, m, q)
                w.writerow(row)
                append_checkpoint_id(ckpt_path, repo_id)
                seen.add(repo_id)
                q_new += 1
                total_new += 1

                if args.max_per_query and q_new >= args.max_per_query:
                    break

                if args.sleep > 0:
                    time.sleep(args.sleep)

        print(f"[{now_utc_iso()}] Query done: {q} (new rows: {q_new})")

    print(f"[{now_utc_iso()}] All done. Total new rows: {total_new}. Output: {args.out}")
    print(f"Checkpoint IDs: {ckpt_path}")


if __name__ == "__main__":
    main()
