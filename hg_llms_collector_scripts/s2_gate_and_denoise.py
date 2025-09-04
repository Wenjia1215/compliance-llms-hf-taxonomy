#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2/3 — Gate & De-noise strong-signal candidates (no extra API calls).

Input:  CSV from Step 1 (e.g., candidates_step1_frameworks.csv) with columns:
        id, owner, name, pipeline_tag, tags (json or string), downloads, likes,
        created_at, last_modified, license, hf_url,
        source_query, collection_ts_utc, readme_text, why_included

Output (written to --outdir):
  - step2_kept.csv        (clean set for taxonomy labeling; one representative per family)
  - step2_rejected.csv    (everything dropped, with why_rejected)
  - step2_family_map.csv  (family normalization map: base_id_guess -> members)

Also prints PRISMA-like counts.

Usage:
  python s2_gate_and_denoise.py --in /path/to/candidates_step1_frameworks.csv \
                                --outdir /path/to/out_step2 \
                                --readme_min 120 --window 30
"""

import os, re, json, argparse
import pandas as pd
from pathlib import Path

# ----------------------------
# Config — Frameworks & Terms
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

# Weak terms: only valid near a framework mention
WEAK_RISK = [r"\brisk\b", r"\bscoring\b", r"\bjudge\b", r"\bevaluation\b"]

# Obvious noise (we keep a flag to review overlaps; hard-drop only if no framework scope)
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

# suffixes for quant/variant names (family normalization)
VARIANT_SUFFIX = re.compile(r"-(?:gguf|gptq|awq|q\d+(?:_\d+)?|i\d+)$", re.I)

# ----------------------------
# Helpers
# ----------------------------

def safe_text(x) -> str:
    """Robustly coerce to a plain string, treating NaN/None as empty."""
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return "" if x is None else str(x)

def parse_tags(raw):
    """Parse tags field which may be JSON list, Python-ish list string, or comma string."""
    if isinstance(raw, list):
        return [safe_text(t).strip() for t in raw if safe_text(t).strip()]
    s = safe_text(raw).strip()
    if not s:
        return []
    # Try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [safe_text(t).strip() for t in obj if safe_text(t).strip()]
    except Exception:
        pass
    # Try Python-like list: "['a','b']"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        return [p for p in parts if p]
    # Fallback comma-split
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

def guess_base_id(owner, name):
    if not owner or not name:
        return ""
    base_name = VARIANT_SUFFIX.sub("", name)
    return f"{owner}/{base_name}"

# ----------------------------
# Gating logic
# ----------------------------

def decide_keep(row, readme_min=120, window=30):
    # Gather text fields
    tags = parse_tags(row.get("tags", "[]"))

    text = norm_text(
        row.get("id",""), row.get("owner",""), row.get("name",""),
        " ".join(tags), safe_text(row.get("readme_text"))
    )

    readme = safe_text(row.get("readme_text")).strip()
    readme_len = len(readme)

    # A) Contentfulness gate
    if readme_len < readme_min:
        return False, "empty_readme"
    if any(h in readme.lower() for h in PLACEHOLDER_HINTS):
        return False, "placeholder_readme"

    # B) Compliance relevance
    scopes = find_framework_scopes(text)
    strong_intent = contains_any(INTENTS_STRONG, text)
    risk_and_fw = risk_near_framework(text, window=window)

    if not (scopes or strong_intent or risk_and_fw):
        return False, "no_compliance_signal"

    # C) Noise exclusion (hard-drop only if no framework scope)
    has_noise = contains_any(NEG_NOISE, text)
    if has_noise and not scopes:
        return False, "cv_ocr_noise"

    return True, "ok"

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV from Step 1")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--readme_min", type=int, default=120, help="Min README characters")
    ap.add_argument("--window", type=int, default=30, help="Risk-framework proximity window size (chars)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # derive base_id_guess for family normalization
    df["base_id_guess"] = df.apply(lambda r: guess_base_id(r.get("owner",""), r.get("name","")), axis=1)

    # flags + decisions
    decisions = []
    regs = []
    noise_flags = []

    for _, r in df.iterrows():
        tags = parse_tags(r.get("tags", "[]"))
        text = norm_text(r.get("id",""), r.get("owner",""), r.get("name",""),
                         " ".join(tags), safe_text(r.get("readme_text")))
        scopes = find_framework_scopes(text)
        keep, reason = decide_keep(r, readme_min=args.readme_min, window=args.window)
        noise_flag = contains_any(NEG_NOISE, text)

        decisions.append((keep, reason))
        regs.append(",".join(scopes))
        noise_flags.append(bool(noise_flag))

    df["reg_scopes"] = regs
    df["cv_ocr_flag"] = noise_flags
    df["decision"] = ["keep" if k else "reject" for (k, _) in decisions]
    df["why_rejected"] = [("" if k else reason) for (k, reason) in decisions]

    kept = df[df["decision"]=="keep"].copy()
    rejected = df[df["decision"]=="reject"].copy()

    # Family map for reference (before representative selection)
    family_map = kept.groupby("base_id_guess")["id"].apply(list).reset_index()
    family_map.rename(columns={"id":"family_members"}, inplace=True)

    # Choose ONE representative per family (if >1) — pick the one with longest README
    if not kept.empty:
        rep_mask = pd.Series(False, index=kept.index)
        for base, group in kept.groupby("base_id_guess", dropna=False):
            if (not isinstance(base, str)) or (base.strip() == "") or len(group) == 1:
                rep_mask.loc[group.index] = True
            else:
                readme_len = group["readme_text"].apply(lambda x: len(safe_text(x)))
                best_idx = readme_len.idxmax()
                rep_mask.loc[best_idx] = True
        kept_repr = kept[rep_mask].copy()
    else:
        kept_repr = kept.copy()

    # Write outputs
    kept_repr.to_csv(outdir / "step2_kept.csv", index=False, encoding="utf-8")
    rejected.to_csv(outdir / "step2_rejected.csv", index=False, encoding="utf-8")
    family_map.to_csv(outdir / "step2_family_map.csv", index=False, encoding="utf-8")

    # PRISMA-like counts
    n_total = len(df)
    n_kept_raw = len(kept)
    n_kept = len(kept_repr)
    n_rej = len(rejected)
    n_cvflag = int(kept_repr["cv_ocr_flag"].sum()) if not kept_repr.empty else 0

    print("==== Step 2/3 Gate & De-noise Summary ====")
    print(f"Total candidates:            {n_total}")
    print(f"Kept before family collapse: {n_kept_raw}")
    print(f"Kept (representatives only): {n_kept}")
    print(f"Rejected:                    {n_rej}")
    print(f"Kept with CV/OCR flag:       {n_cvflag}  (review if desired)")
    print("------------------------------------------")
    if not rejected.empty and "why_rejected" in rejected.columns:
        print("Top rejection reasons:")
        print(rejected["why_rejected"].value_counts().head(10).to_string())
    print("------------------------------------------")
    print(f"Outputs written to: {outdir.resolve()}")
    print(" - step2_kept.csv")
    print(" - step2_rejected.csv")
    print(" - step2_family_map.csv")

if __name__ == "__main__":
    main()
