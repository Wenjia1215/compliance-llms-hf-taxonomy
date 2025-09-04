#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Rescue base repos from quant forks and README backlinks, then merge.

Inputs:
  --kept      Path to step2_kept.csv
  --rejected  Path to step2_rejected.csv   (improves rescue coverage)
  --outdir    Output directory

What it does:
  1) Parse readme_text from kept + rejected to find HF links like:
       https://huggingface.co/<owner>/<repo>
     (ignores datasets/spaces/other paths)
  2) For each linked repo not already in your set, fetch model_info + README.
  3) Apply the SAME gating as Step 2 (contentfulness, compliance relevance, noise).
  4) Merge rescued repos with step2_kept.csv, then collapse families (prefer longer README).
  5) Write:
       - step3_rescued_bases.csv      (only the newly rescued & accepted)
       - step3_after_rescue.csv       (merged + deduped representatives)
       - step3_rescue_log.csv         (all rescue attempts with decisions)

Usage:
  python s3_rescue_base_repos.py \
      --kept /path/to/out_step2/step2_kept.csv \
      --rejected /path/to/out_step2/step2_rejected.csv \
      --outdir /path/to/out_step3
"""

import os, re, json, argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, __version__ as HFHUB_VER
except Exception:
    raise RuntimeError("Please: pip install -U huggingface_hub")

print(f"[Init] huggingface_hub={HFHUB_VER}")

# ----------------------------
# Config — Frameworks & Terms (same as Step 2)
# ----------------------------

FRAMEWORKS = [
    ("GDPR",        [r"\bgdpr\b", r"general data protection regulation", r"\bart\.\s*\d+"]),
    ("HIPAA",       [r"\bhipaa\b", r"hitech"]),
    ("SOC 2",       [r"\bsoc\s*2\b", r"\bsoc2\b", r"\baicpa\b", r"\btype\s*[ii1]+\b"]),
    ("ISO 27001",   [r"\biso/?iec\s*27001\b", r"\biso\s*27001\b"]),
    ("PCI DSS",     [r"\bpci[-\s]?dss\b"]),
    ("SOX",         [r"\bsox\b", r"sarbanes[-\s]?oxley", r"\bsox\s*404\b"]),
    ("FedRAMP",     [r"\bfedramp\b", r"\b(fedramp\s*(moderate|high))\b"]),
    ("CMMC",        [r"\bcmmc\b", r"\bcmmc\s*2\.0\b"]),
    ("NIST 800-53", [r"\bnist\s*(sp)?\s*800-53\b", r"\bsp\s*800-53\b", r"\b800-53\b"]),
    ("NIST 800-207",[r"\bnist\s*(sp)?\s*800-207\b", r"\bsp\s*800-207\b", r"zero\s*trust\s*800-207"]),
    ("NIST CSF",    [r"\bnist\s*csf\b", r"\bcsf\s*(1\.1|2\.0)?\b"]),
    ("CIS Controls",[r"\bcis\s*controls\b", r"\bcis\s*v?8\b"]),
    ("CCPA/CPRA",   [r"\bccpa\b", r"\bcpra\b"]),
]

INTENTS_STRONG = [
    r"\bcompliance\b", r"\baudit\b", r"\battestation\b",
    r"\bcontrol\s*mapping\b", r"\bevidence\b", r"\bpolicy\b"
]

WEAK_RISK = [r"\brisk\b", r"\bscoring\b", r"\bjudge\b", r"\bevaluation\b"]

NEG_NOISE = [
    r"\blayoutlm", r"\bocr\b", r"\bdonut\b", r"\byolo\b", r"\bdetr\b",
    r"\bsegmentation\b", r"\bobject\s*detection\b", r"\bdocument\s*layout\b",
    r"\binvoice\s*parser\b", r"\breceipt\s*parser\b",
    r"\btts\b", r"\basr\b", r"\bwhisper\b",
    r"\bimage\s*generation\b", r"\bdiffusion\b",
]

PLACEHOLDER_HINTS = [
    "more information needed", "tbd", "todo", "coming soon", "wip"
]

VARIANT_SUFFIX = re.compile(r"-(?:gguf|gptq|awq|q\d+(?:_\d+)?|i\d+)$", re.I)
HF_LINK_RE = re.compile(r"https?://huggingface\.co/([A-Za-z0-9][\w\-]+)/([\w\.\-]+)")

def now_utc_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ----------------------------
# Helpers (same style as Step 2)
# ----------------------------

def safe_text(x) -> str:
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return "" if x is None else str(x)

def parse_tags(raw):
    if isinstance(raw, list):
        return [safe_text(t).strip() for t in raw if safe_text(t).strip()]
    s = safe_text(raw).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [safe_text(t).strip() for t in obj if safe_text(t).strip()]
    except Exception:
        pass
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        return [p for p in parts if p]
    return [p.strip() for p in s.split(",") if p.strip()]

def norm_text(*parts) -> str:
    txt = " ".join([safe_text(p) for p in parts if p is not None]).lower()
    return re.sub(r"\s+", " ", txt)

def contains_any(patterns, text) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)

def find_framework_scopes(text):
    scopes = []
    for label, pats in FRAMEWORKS:
        if contains_any(pats, text):
            scopes.append(label)
    return sorted(set(scopes))

def risk_near_framework(text, window=30) -> bool:
    fw_pats = [p for (_, pats) in FRAMEWORKS for p in pats]
    fw_regex = re.compile("|".join(fw_pats), re.I)
    risk_regex = re.compile("|".join(WEAK_RISK), re.I)
    for m in fw_regex.finditer(text):
        left = max(0, m.start() - window)
        right = min(len(text), m.end() + window)
        if risk_regex.search(text[left:right]):
            return True
    for m in risk_regex.finditer(text):
        left = max(0, m.start() - window)
        right = min(len(text), m.end() + window)
        if fw_regex.search(text[left:right]):
            return True
    return False

def decide_keep(row, readme_min=120, window=30):
    tags = parse_tags(row.get("tags", "[]"))
    text = norm_text(
        row.get("id",""), row.get("owner",""), row.get("name",""),
        " ".join(tags), safe_text(row.get("readme_text"))
    )
    readme = safe_text(row.get("readme_text")).strip()
    if len(readme) < readme_min:
        return False, "empty_readme"
    if any(h in readme.lower() for h in PLACEHOLDER_HINTS):
        return False, "placeholder_readme"
    scopes = find_framework_scopes(text)
    strong_intent = contains_any(INTENTS_STRONG, text)
    risk_fw = risk_near_framework(text, window=window)
    if not (scopes or strong_intent or risk_fw):
        return False, "no_compliance_signal"
    has_noise = contains_any(NEG_NOISE, text)
    if has_noise and not scopes:
        return False, "cv_ocr_noise"
    return True, "ok"

def guess_base_id(owner, name):
    if not owner or not name:
        return ""
    base_name = VARIANT_SUFFIX.sub("", name)
    return f"{owner}/{base_name}"

# ----------------------------
# HF API helpers
# ----------------------------

def fetch_model(api: HfApi, repo_id: str) -> dict:
    """Return a row-like dict with metadata + README. Missing fields become empty."""
    try:
        mi = api.model_info(repo_id)
    except Exception:
        return {}
    owner, name = "", ""
    rid = getattr(mi, "id", None) or getattr(mi, "modelId", None) or repo_id
    if rid and "/" in rid:
        owner, name = rid.split("/", 1)
    else:
        owner = getattr(mi, "author", "") or getattr(mi, "owner", "")
        name = getattr(mi, "name", "") or rid

    # README
    readme = ""
    try:
        card = api.get_model_card(rid)
        if card is not None and hasattr(card, "text") and card.text:
            readme = card.text
    except Exception:
        pass

    tags = getattr(mi, "tags", None)
    pipeline_tag = getattr(mi, "pipeline_tag", None)
    downloads = getattr(mi, "downloads", None)
    likes = getattr(mi, "likes", None)
    created_at = getattr(mi, "created_at", None)
    last_modified = getattr(mi, "lastModified", None) or getattr(mi, "last_modified", None)
    license_str = getattr(mi, "license", None)
    row = {
        "id": rid,
        "owner": owner or "",
        "name": name or "",
        "pipeline_tag": pipeline_tag,
        "tags": json.dumps(tags or []),
        "downloads": downloads,
        "likes": likes,
        "created_at": str(created_at) if created_at is not None else "",
        "last_modified": str(last_modified) if last_modified is not None else "",
        "license": license_str or "",
        "hf_url": f"https://huggingface.co/{rid}",
        "source_query": "rescue_from_links",
        "collection_ts_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "readme_text": safe_text(readme),
        "why_included": "backlinked_from_quant_or_readme",
    }
    return row

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", required=True, help="Path to out_step2/step2_kept.csv")
    ap.add_argument("--rejected", default="", help="Path to out_step2/step2_rejected.csv (optional but recommended)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--readme_min", type=int, default=120, help="Min README chars (same as Step 2)")
    ap.add_argument("--window", type=int, default=30, help="Risk-framework proximity window (chars)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    kept = pd.read_csv(args.kept)
    rej = pd.read_csv(args.rejected) if args.rejected and os.path.exists(args.rejected) else pd.DataFrame()

    # Build set of all known repo ids (avoid re-fetch duplicates)
    known_ids = set(safe_text(x) for x in pd.concat([kept["id"], rej["id"]], ignore_index=True).dropna().unique())

    # Scrape HF links from readmes in both kept and rejected
    def extract_links(series):
        ids = set()
        for txt in series.fillna(""):
            for (owner, repo) in HF_LINK_RE.findall(safe_text(txt)):
                # filter non-model urls (simple heuristic: exactly two segments; avoid 'datasets', 'spaces')
                if owner.lower() in ("datasets", "spaces"):
                    continue
                ids.add(f"{owner}/{repo}")
        return ids

    link_ids = extract_links(kept["readme_text"])
    if not rej.empty:
        link_ids |= extract_links(rej["readme_text"])

    # Remove any we already have
    candidates = sorted(list(link_ids - known_ids))
    print(f"[{now_utc_iso()}] Found {len(link_ids)} unique HF links; {len(candidates)} not already in your set.")

    if not candidates:
        # Write no-op outputs but still emit merged representatives (just re-output Step 2 kept)
        kept.to_csv(outdir / "step3_after_rescue.csv", index=False, encoding="utf-8")
        pd.DataFrame(columns=["repo_id","status","decision","reason"]).to_csv(outdir / "step3_rescue_log.csv", index=False)
        pd.DataFrame(columns=kept.columns).to_csv(outdir / "step3_rescued_bases.csv", index=False)
        print("No new candidates to rescue. step3_after_rescue.csv = step2_kept.csv")
        return

    # Fetch candidates from HF API
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    rescued_rows = []
    rescue_log = []

    for rid in candidates:
        row = fetch_model(api, rid)
        if not row:
            rescue_log.append({"repo_id": rid, "status":"fetch_failed", "decision":"reject", "reason":"api_error"})
            continue
        keep, reason = decide_keep(row, readme_min=args.readme_min, window=args.window)
        decision = "keep" if keep else "reject"
        rescued_rows.append((row, decision, reason))
        rescue_log.append({"repo_id": rid, "status":"fetched", "decision":decision, "reason":reason})

    # Assemble dataframes
    if rescued_rows:
        resc_df = pd.DataFrame([r for (r, _, _) in rescued_rows])
        resc_df["decision"] = [d for (_, d, _) in rescued_rows]
        resc_df["why_rejected"] = [("" if d == "keep" else rsn) for (_, d, rsn) in rescued_rows]
    else:
        resc_df = pd.DataFrame(columns=kept.columns.tolist() + ["decision","why_rejected"])

    rescue_log_df = pd.DataFrame(rescue_log)
    rescue_log_df.to_csv(outdir / "step3_rescue_log.csv", index=False, encoding="utf-8")

    # Keep only rescued that passed gates
    resc_kept = resc_df[resc_df["decision"]=="keep"].copy()
    resc_kept.to_csv(outdir / "step3_rescued_bases.csv", index=False, encoding="utf-8")

    # Merge with step2_kept and collapse families (choose longest README as representative)
    merged = pd.concat([kept, resc_kept], ignore_index=True)

    # Build base_id_guess
    merged["base_id_guess"] = merged.apply(lambda r: guess_base_id(r.get("owner",""), r.get("name","")), axis=1)

    # Choose representative per family by longest README
    if not merged.empty:
        rep_mask = pd.Series(False, index=merged.index)
        for base, group in merged.groupby("base_id_guess", dropna=False):
            if (not isinstance(base, str)) or (base.strip() == "") or len(group) == 1:
                rep_mask.loc[group.index] = True
            else:
                readme_len = group["readme_text"].apply(lambda x: len(safe_text(x)))
                best_idx = readme_len.idxmax()
                rep_mask.loc[best_idx] = True
        merged_repr = merged[rep_mask].copy()
    else:
        merged_repr = merged.copy()

    merged_repr.to_csv(outdir / "step3_after_rescue.csv", index=False, encoding="utf-8")

    print("==== Step 3 Rescue Summary ====")
    print(f"Candidates from links:    {len(candidates)}")
    print(f"Rescued & kept:           {len(resc_kept)}")
    print(f"Merged representatives:    {len(merged_repr)}")
    print("--------------------------------")
    print(f"Outputs @ {outdir.resolve()}:")
    print(" - step3_rescued_bases.csv")
    print(" - step3_after_rescue.csv")
    print(" - step3_rescue_log.csv")

if __name__ == "__main__":
    main()
