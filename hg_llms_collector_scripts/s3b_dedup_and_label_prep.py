#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, json, argparse, pandas as pd
from pathlib import Path

# ---------- Regex & ranking helpers ----------
# Match semantic-ish versions like v5, v5.2, v4.2.1
SEMVER_RE = re.compile(r"(?:^|[-_])v(?P<ver>\d+(?:\.\d+){0,2})(?:$|[-_])", re.I)

# Strip ALL trailing version/quant/format tokens to form a family key
FAMILY_STRIP_RE = re.compile(
    r"(?:-(?:finetuned-)?v\d+(?:\.\d+)*|-rc\d+|-alpha|-beta"
    r"|-gguf|-gptq|-awq|-q\d+(?:_\d+)?|-fp\d+|-int\d+|-demo|-dev)+$",
    re.I
)

# Compliance scope hints, interaction, lifecycle…
FRAMEWORKS = [
    ("GDPR",        [r"\bgdpr\b", r"general data protection regulation", r"\bart\.\s*\d+"]),
    ("HIPAA",       [r"\bhipaa\b", r"hitech"]),
    ("SOC 2",       [r"\bsoc\s*2\b", r"\bsoc2\b", r"\baicpa\b", r"\btype\s*[ii1]+\b"]),
    ("ISO 27001",   [r"\biso/?iec\s*27001\b", r"\biso\s*27001\b"]),
    ("PCI DSS",     [r"\bpci[-\s]?dss\b"]),
    ("SOX",         [r"\bsox\b", r"sarbanes[-\s]?oxley", r"\bsox\s*404\b"]),
    ("FedRAMP",     [r"\bfedramp\b"]),
    ("CMMC",        [r"\bcmmc\b"]),
    ("NIST 800-53", [r"\bnist\s*(sp)?\s*800-53\b", r"\bsp\s*800-53\b", r"\b800-53\b"]),
    ("NIST 800-207",[r"\bnist\s*(sp)?\s*800-207\b", r"\bsp\s*800-207\b", r"zero\s*trust\s*800-207"]),
    ("NIST CSF",    [r"\bnist\s*csf\b", r"\bcsf\s*(1\.1|2\.0)?\b"]),
    ("CIS Controls",[r"\bcis\s*controls\b", r"\bcis\s*v?8\b"]),
    ("CCPA/CPRA",   [r"\bccpa\b", r"\bcpra\b"]),
]
RAG_HINTS   = r"rag|retrieval|vector|faiss|chroma|weaviate|milvus|pinecone|opensearch|elasticsearch"
AGENT_HINTS = r"agent|tool.?use|function.?calling|workflow|autonomous|planner|executor"
LIFECYCLE_MAP = [
    ("control_mapping", r"control\s*mapp|map.*control|map.*(cis|nist|iso|soc|gdpr)"),
    ("evidence",       r"evidence|artifact|attestation"),
    ("monitoring",     r"continuous|monitor|drift|alert"),
    ("audit_report",   r"audit|report|attestation"),
    ("requirements",   r"requirement|policy\s*(mining|generation)|standardization"),
    ("remediation",    r"remediation|fix|mitigation|gap\s*analysis"),
]
REG_CITE = r"\bart\.\s*\d+|annex|iso/?iec\s*27001|sp\s*800-53|800-53|nist\s*csf|cis\s*controls|pci[-\s]?dss|hipaa|fedramp|sox|ccpa|cpra"

# ---------- small utils ----------
def safe_text(x):
    if isinstance(x, str): return x
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    return "" if x is None else str(x)

def parse_semver(name):
    s = safe_text(name)
    m = None
    for m in SEMVER_RE.finditer(s):
        pass  # last match (closest to end)
    if not m:
        return (0, 0, 0)
    parts = m.group("ver").split(".")
    parts += ["0"]*(3-len(parts))
    try:
        return tuple(int(p) for p in parts[:3])
    except Exception:
        return (0, 0, 0)

def quant_rank(name):
    """Lower is better: 0=base/no quant, then fp16/bf16, then int/quantized, then GGUF/GPTQ/AWQ."""
    s = safe_text(name).lower()
    if "fp32" in s: return 1
    if "fp16" in s or "bf16" in s: return 2
    if "int8" in s: return 3
    # generic quant (ranked roughly by precision names)
    if "q8" in s: return 3
    if "q6" in s: return 4
    if "q5" in s: return 5
    if "q4" in s: return 6
    if any(tok in s for tok in ["gguf","gptq","awq"]): return 7
    return 0  # default = base

def family_key(owner, name):
    return f"{safe_text(owner)}/{FAMILY_STRIP_RE.sub('', safe_text(name))}".strip("/")

def parse_tags(raw):
    s = safe_text(raw).strip()
    if not s: return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list): return [safe_text(t).strip() for t in obj if safe_text(t).strip()]
    except Exception:
        pass
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        return [p for p in parts if p]
    return [p.strip() for p in s.split(",") if p.strip()]

def contains_any(patts, text):
    return any(re.search(p, text, re.I) for p in patts)

def find_scopes(text):
    scopes = []
    for label, pats in FRAMEWORKS:
        if contains_any(pats, text): scopes.append(label)
    return ",".join(sorted(set(scopes)))

def guess_interaction(txt):
    if re.search(AGENT_HINTS, txt, re.I): return "agent"
    if re.search(RAG_HINTS, txt, re.I):   return "rag"
    return "standard"

def guess_lifecycle(txt):
    for label, pat in LIFECYCLE_MAP:
        if re.search(pat, txt, re.I): return label
    return "unknown"

def guess_evidence(txt):
    if re.search(REG_CITE, txt, re.I): return "regulatory_citation"
    if re.search(r"\bpolicy|procedure|standard\b", txt, re.I): return "policy_citation"
    return "none_or_unknown"

def guess_autonomy(interaction):
    return {"agent":"L2","rag":"L1"}.get(interaction, "L0")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="step3_after_rescue.csv")
    ap.add_argument("--outdir", required=True, help="output folder (e.g., /content/drive/MyDrive/compliance-llm-taxonomy)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if df.empty:
        raise SystemExit("Input is empty. Nothing to dedup.")

    # Family ID
    df["cluster_id"] = df.apply(lambda r: family_key(r.get("owner",""), r.get("name","")), axis=1)

    # Sort keys for representative selection
    df["_semver"] = df["name"].apply(parse_semver)
    df["_qr"] = df["name"].apply(quant_rank)
    df["_last"] = pd.to_datetime(df.get("last_modified"), errors="coerce")
    df["_dl"] = pd.to_numeric(df.get("downloads"), errors="coerce").fillna(0).astype(float)
    df["_readme_len"] = df["readme_text"].apply(lambda x: len(safe_text(x)))

    # Representative per family:
    reps = []
    for cid, g in df.groupby("cluster_id", dropna=False):
        if cid == "" or len(g) == 1:
            reps.append(g.index[0]); continue
        # Priority: newest semver ↓, best quant rank ↑(non-quant first), newest last_modified ↓, downloads ↓, readme len ↓
        g_sorted = g.sort_values(
            by=["_semver","_qr","_last","_dl","_readme_len"],
            ascending=[False, True, False, False, False]
        )
        reps.append(g_sorted.index[0])

    final = df.loc[sorted(set(reps))].copy()

    # ----- Prefill taxonomy columns for label sheet -----
    tags_txt = final.get("tags","").apply(parse_tags).apply(lambda lst: " ".join(lst))
    corpus = (
        final.get("id","").astype(str) + " " +
        final.get("owner","").astype(str) + " " +
        final.get("name","").astype(str) + " " +
        tags_txt + " " +
        final.get("readme_text","").astype(str).fillna("")
    )
    final["reg_scopes"] = corpus.apply(find_scopes)
    final["interaction_mode_prefill"] = corpus.apply(guess_interaction)
    final["lifecycle_primary_prefill"] = corpus.apply(guess_lifecycle)
    final["evidence_traceability_prefill"] = corpus.apply(guess_evidence)
    final["autonomy_level_prefill"] = final["interaction_mode_prefill"].apply(guess_autonomy)

    for col in ["deployment", "safety_profile", "evaluation_notes"]:
        if col not in final.columns:
            final[col] = ""

    # ----- Save -----
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    final_out = outdir / "step3_final_dedup.csv"
    label_out = outdir / "step4_label_sheet.csv"

    # Save the representative list (clean inventory)
    final.to_csv(final_out, index=False, encoding="utf-8")
    # Label sheet == same data + prefills (you can edit in Sheets/Excel)
    final.to_csv(label_out, index=False, encoding="utf-8")

    print("==== Dedup & Prefill Summary ====")
    print(f"Input rows:      {len(df)}")
    print(f"Families:        {df['cluster_id'].nunique()}")
    print(f"Kept (unique):   {len(final)}")
    print(f"Saved inventory: {final_out}")
    print(f"Label sheet:     {label_out}")

if __name__ == "__main__":
    main()
