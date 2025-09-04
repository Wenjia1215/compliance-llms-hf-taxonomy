#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1b — Repair missing README text in the Step-1 candidates CSV.

- Reads a Step-1 CSV (candidates_step1_*.csv)
- For rows where readme_text is empty/short, refetch using multiple fallbacks:
    1) HfApi.get_model_card(repo).text
    2) hf_hub_download variants: README.md / Readme.md / README.MD / README.rst / README
- Writes an updated CSV (does not touch other fields)

Usage:
  export HF_TOKEN=hf_xxx   # optional but helps with rate limits
  python s1b_repair_readmes.py --in /path/to/candidates_step1_frameworks.csv \
                               --out /path/to/candidates_step1_frameworks_repaired.csv \
                               --minlen 30 --sleep 0.2 --retries 2
"""

import os, time, argparse
import pandas as pd

from huggingface_hub import HfApi, hf_hub_download

VARIANTS = ["README.md","Readme.md","README.MD","README.rst","README"]

def safe_len(x):
    try:
        return len(x) if isinstance(x, str) else 0
    except Exception:
        return 0

def fetch_readme(api, repo_id, sleep=0.2, retries=2):
    # 1) try model card API
    for attempt in range(retries+1):
        try:
            card = api.get_model_card(repo_id)
            txt = getattr(card, "text", "") or ""
            if safe_len(txt) > 0:
                return txt
        except Exception:
            pass
        time.sleep(sleep)

    # 2) try file variants via hf_hub_download
    for fname in VARIANTS:
        for attempt in range(retries+1):
            try:
                p = hf_hub_download(repo_id, filename=fname, repo_type="model", local_dir="/tmp")
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read()
                if safe_len(data) > 0:
                    return data
            except Exception:
                pass
            time.sleep(sleep)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Step-1 candidates CSV path")
    ap.add_argument("--out", required=True, help="Output repaired CSV path")
    ap.add_argument("--minlen", type=int, default=30, help="Only repair when readme_text length < minlen")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between API/file attempts")
    ap.add_argument("--retries", type=int, default=2, help="Retries per method")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if "id" not in df.columns:
        raise SystemExit("Input CSV missing 'id' column.")

    # Ensure readme_text column exists
    if "readme_text" not in df.columns:
        df["readme_text"] = ""

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    repaired = 0
    total_targets = 0

    for i, row in df.iterrows():
        rid = row.get("id","")
        cur_len = safe_len(row.get("readme_text",""))
        if cur_len >= args.minlen:
            continue
        total_targets += 1
        txt = fetch_readme(api, rid, sleep=args.sleep, retries=args.retries)
        if safe_len(txt) > cur_len:
            df.at[i, "readme_text"] = txt
            repaired += 1
        # tiny polite pause between repos
        time.sleep(args.sleep)

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Targets needing repair (<{args.minlen} chars): {total_targets}")
    print(f"Successfully repaired: {repaired}")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
