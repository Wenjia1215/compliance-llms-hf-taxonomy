#!/usr/bin/env python3
"""
HF Compliance LLM Harvester → Triager → Catalog (English-only)
Resilient version with network retries & per-query resume via de-dupe.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import yaml
from huggingface_hub import HfApi, list_models
from huggingface_hub.hf_api import ModelInfo
from rapidfuzz import fuzz
from tqdm import tqdm


CORE_TERMS = [
    "compliance", "regulatory", "regulation", "governance", "GRC",
    "audit", "auditor", "attestation", "control", "control mapping",
    "policy", "policy checker", "policy classification", "policy violation",
    "risk", "risk assessment", "conformity", "privacy compliance",
]

FRAMEWORKS = [
    "HIPAA", "GDPR", "CCPA", "CPRA", "SOC 2", "SOC2", "ISO 27001", "ISO27001",
    "PCI DSS", "PCI-DSS", "SOX", "Sarbanes-Oxley",
    "NIST 800-53", "NIST SP 800-53", "NIST CSF", "FedRAMP", "FISMA", "CMMC",
    "GLBA", "FERPA", "COPPA", "LGPD", "PIPEDA", "PDPA", "MiFID", "FINRA", "FCA",
]

TASK_TERMS = [
    "compliance assistant", "regulatory assistant", "policy Q&A",
    "control mapping", "evidence extraction", "requirements traceability",
    "security questionnaire", "vendor risk questionnaire", "DPA review",
    "policy generation", "policy classification", "policy violation detection",
    "internal audit", "gap analysis",
]

SAFETY_POLICY_TERMS = [
    "guardrail", "guardrails", "LlamaGuard", "policy filter", "safety policy",
    "content safety", "safety classifier", "red-teaming policy", "safety spec",
]

NEGATIVE_TERMS = [
    "pep8", "flake8", "eslint", "pylint", "code style", "coding standards",
    "linter", "formatting", "style guide",
    "SEO", "search engine optimization", "WCAG", "accessibility",
    "PCIe", "PCI-E", "PCI express", "pcie",
    "html compliance", "css compliance",
]

PIPELINE_WHITELIST = {
    "text-generation",
    "text2text-generation",
    "text-classification",
    "document-question-answering",
    "question-answering",
    "token-classification",
}

WEIGHT_EXTS = {".safetensors", ".bin", ".gguf", ".onnx", ".pt"}


def normalize_text(s: Optional[str]) -> str:
    return (s or "").lower()


def get_last_modified(model: ModelInfo):
    return getattr(model, "last_modified", None) or getattr(model, "lastModified", None)


def has_weights(model: ModelInfo) -> bool:
    try:
        for sib in (getattr(model, "siblings", None) or []):
            name = getattr(sib, "rfilename", "") or getattr(sib, "path", "") or ""
            if any(name.endswith(ext) for ext in WEIGHT_EXTS):
                return True
        return False
    except Exception:
        return False


def load_readme(api: HfApi, repo_id: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (repo_id.replace("/", "__") + ".md")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")
    try:
        info = api.model_info(repo_id, expand=["readme"])
        readme = getattr(info, "readme", None)
        if readme is None:
            txt = ""
        else:
            txt = readme.decode("utf-8", errors="ignore") if isinstance(readme, (bytes, bytearray)) else str(readme)
        cache_file.write_text(txt, encoding="utf-8", errors="ignore")
        return txt
    except TypeError:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model", local_dir=cache_dir)
            return Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    except Exception:
        return ""


def extract_card_yaml(model: ModelInfo) -> Dict:
    data = getattr(model, "cardData", None) or {}
    if isinstance(data, dict):
        return data
    try:
        return dict(data)
    except Exception:
        return {}


def evidence_pack(model: ModelInfo, readme_text: str) -> str:
    parts = []
    parts.append(f"id: {model.id}")
    tags = getattr(model, "tags", None) or []
    if tags:
        parts.append("tags: " + ", ".join(tags[:10]))
    cd = extract_card_yaml(model)
    if "license" in cd or getattr(model, "license", None):
        parts.append(f"license: {cd.get('license') or getattr(model, 'license', None)}")
    pt = cd.get("pipeline_tag") or getattr(model, "pipeline_tag", None)
    if pt:
        parts.append(f"pipeline_tag: {pt}")
    if readme_text:
        snippet = re.sub(r"\s+", " ", readme_text)[:300]
        parts.append("readme: " + snippet)
    return " | ".join(parts)


@dataclasses.dataclass
class TriageResult:
    keep: bool
    reason: str
    tier: str
    category: str
    matched_keywords: List[str]
    negatives: List[str]


def categorize(text: str) -> str:
    t = normalize_text(text)
    if ("control mapping" in t) or ("requirements traceability" in t):
        return "Control Mapping"
    if ("evidence extraction" in t) or ("dpa review" in t) or ("document review" in t):
        return "Evidence Extraction"
    if ("security questionnaire" in t) or ("vendor risk" in t) or ("risk assessment" in t) or ("grc" in t) or ("governance" in t):
        return "Risk/GRC Assistant"
    if ("policy q&a" in t) or ("regulatory assistant" in t) or ("regulatory qa" in t):
        return "Regulatory QA"
    if any(k.lower() in t for k in SAFETY_POLICY_TERMS):
        return "Safety Policy"
    if ("legal" in t) or ("contract" in t) or ("case law" in t):
        return "Legal (non-regulatory)"
    return "Unclear"


def triage_model(model: ModelInfo, readme_text: str, min_downloads: int) -> TriageResult:
    cd = extract_card_yaml(model)
    tags = " ".join(getattr(model, "tags", None) or [])
    title = model.id.split("/")[-1].replace("-", " ")
    text_blob = " ".join([title, tags, json.dumps(cd, default=str), readme_text])

    matched = []
    negatives = []
    for k in CORE_TERMS + FRAMEWORKS + TASK_TERMS + SAFETY_POLICY_TERMS:
        if k.lower() in text_blob.lower():
            matched.append(k)
    for k in NEGATIVE_TERMS:
        if k.lower() in text_blob.lower():
            negatives.append(k)

    if negatives and not matched:
        return TriageResult(False, "Negative domain hit without compliance evidence", "-", "-", [], negatives)

    if not matched:
        return TriageResult(False, "No compliance/regulatory evidence", "-", "-", [], negatives)

    strong_framework_hit = any(fr.lower() in text_blob.lower() for fr in FRAMEWORKS)
    strong_task_hit = any(t.lower() in text_blob.lower() for t in TASK_TERMS)
    safety_only = any(s.lower() in text_blob.lower() for s in SAFETY_POLICY_TERMS) and not strong_framework_hit

    if strong_framework_hit:
        tier = "S"
    elif strong_task_hit:
        tier = "A"
    elif safety_only:
        tier = "B"
    else:
        tier = "A"

    category = categorize(text_blob)
    return TriageResult(True, "OK", tier, category, matched, negatives)


def score_model(model: ModelInfo, tri: TriageResult) -> float:
    tier_score = {"S": 1.0, "A": 0.7, "B": 0.45, "-": 0.0}.get(tri.tier, 0.0)

    cd = extract_card_yaml(model)
    completeness = 0.0
    if (cd.get("license") or getattr(model, "license", None)):
        completeness += 0.25
    if (cd.get("pipeline_tag") or getattr(model, "pipeline_tag", None)):
        completeness += 0.25
    if getattr(model, "tags", None):
        completeness += 0.25
    if has_weights(model):
        completeness += 0.25

    freshness = 0.0
    try:
        lm = get_last_modified(model)
        days = max(1, (pd.Timestamp.utcnow() - pd.Timestamp(lm)).days) if lm else 180
        freshness = 1.0 / (1.0 + (days / 90.0))
    except Exception:
        freshness = 0.3

    adoption = 0.0
    try:
        dls = getattr(model, "downloads", 0) or 0
        if dls > 0:
            adoption = min(1.0, (len(str(dls)) - 1) / 6.0)
    except Exception:
        adoption = 0.0

    return round(0.45 * tier_score + 0.2 * completeness + 0.2 * freshness + 0.15 * adoption, 4)


def generate_queries() -> List[str]:
    queries = set()
    for term in CORE_TERMS + FRAMEWORKS + TASK_TERMS + SAFETY_POLICY_TERMS:
        queries.add(term)
    for a in CORE_TERMS:
        for b in FRAMEWORKS:
            queries.add(f"{a} {b}")
    for a in CORE_TERMS:
        for b in TASK_TERMS:
            queries.add(f"{a} {b}")
    return sorted(queries)


def iter_models_with_retries(search_q: str, limit: int, max_retries: int, backoff_base: float, seen: Set[str]):
    """Yield models for a query, retrying on network errors. Uses de-dup to 'resume'."""
    remaining = limit
    attempt = 0
    while remaining > 0 and attempt <= max_retries:
        try:
            iterator = list_models(
                search=search_q,
                sort="last_modified",
                direction=-1,
                cardData=True,
                full=True,
                limit=remaining,
            )
            for m in iterator:
                if m.id in seen:
                    continue
                seen.add(m.id)
                remaining = limit - len(seen)
                yield m
            break  # done for this query
        except KeyboardInterrupt:
            raise
        except Exception as e:
            attempt += 1
            wait = min(60.0, backoff_base * (2 ** (attempt - 1)))
            print(f"[WARN] Network error on '{search_q}' (attempt {attempt}/{max_retries}). Retrying in {wait:.1f}s... Error: {type(e).__name__}: {e}")
            time.sleep(wait)
            # loop continues; we'll refetch and skip already-seen ids


def harvest(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    api = HfApi()
    results: Dict[str, Dict] = {}
    excluded: List[Dict] = []

    queries = generate_queries()
    if args.query_subset:
        qs = [q.strip() for q in args.query_subset.split(",") if q.strip()]
        queries = [q for q in queries if any(s.lower() in q.lower() for s in qs)]
        if not queries:
            print("No queries matched your --query-subset filter.", file=sys.stderr)

    pbar = tqdm(total=len(queries), desc="Queries", unit="q")

    for q in queries:
        seen_ids_for_query: Set[str] = set()
        # Iterate with retries
        for model in iter_models_with_retries(
            search_q=q,
            limit=args.max_per_query,
            max_retries=args.network_retries,
            backoff_base=args.network_backoff,
            seen=seen_ids_for_query,
        ):
            if model.id in results or any(model.id == e.get("id") for e in excluded):
                continue

            cd = extract_card_yaml(model)
            pipeline_tag = (cd.get("pipeline_tag") or getattr(model, "pipeline_tag", None))
            pipeline_ok = True
            if args.enforce_pipeline_whitelist or args.pipeline_whitelist_only:
                if pipeline_tag:
                    pipeline_ok = pipeline_tag in PIPELINE_WHITELIST

            if args.pipeline_whitelist_only and not pipeline_ok:
                excluded.append({
                    "id": model.id, "reason": "Pipeline not in whitelist", "pipeline_tag": pipeline_tag,
                    "query": q
                })
                continue

            readme_txt = ""
            if args.readme_scan:
                readme_txt = load_readme(api, model.id, Path(args.readme_cache))

            tri = triage_model(model, readme_txt, args.min_downloads)

            if not tri.keep:
                excluded.append({
                    "id": model.id,
                    "reason": tri.reason,
                    "negatives": ";".join(tri.negatives),
                    "downloads": getattr(model, "downloads", 0) or 0,
                    "likes": getattr(model, "likes", 0) or 0,
                    "last_modified": str(get_last_modified(model)),
                    "query": q,
                })
                continue

            if args.min_downloads and ((getattr(model, "downloads", 0) or 0) < args.min_downloads):
                excluded.append({
                    "id": model.id, "reason": f"Downloads<{args.min_downloads}", "downloads": getattr(model, "downloads", 0) or 0,
                    "likes": getattr(model, "likes", 0) or 0, "last_modified": str(get_last_modified(model)), "query": q
                })
                continue

            record = {
                "id": model.id,
                "author": getattr(model, "author", None),
                "private": bool(getattr(model, "private", False)),
                "downloads": getattr(model, "downloads", 0) or 0,
                "likes": getattr(model, "likes", 0) or 0,
                "last_modified": str(get_last_modified(model)),
                "pipeline_tag": cd.get("pipeline_tag") or getattr(model, "pipeline_tag", None),
                "tags": ", ".join(getattr(model, "tags", None) or []),
                "license": cd.get("license") or getattr(model, "license", None),
                "languages": ", ".join(cd.get("language", []) if isinstance(cd.get("language"), list) else ([cd.get("language")] if cd.get("language") else [])),
                "datasets": ", ".join(cd.get("datasets", []) if isinstance(cd.get("datasets"), list) else ([cd.get("datasets")] if cd.get("datasets") else [])),
                "library": cd.get("library_name") or getattr(model, "library_name", None),
                "base_model": cd.get("base_model"),
                "has_weights": has_weights(model),
                "tier": tri.tier,
                "category": tri.category,
                "matched_keywords": ";".join(tri.matched_keywords[:20]),
                "negatives": ";".join(tri.negatives[:20]),
                "evidence": evidence_pack(model, readme_txt if args.embed_readme_snippet else ""),
                "query_hit": q,
            }
            record["score"] = score_model(model, tri)
            results[model.id] = record

        pbar.update(1)
        if args.sleep > 0:
            time.sleep(args.sleep)

    pbar.close()

    df = pd.DataFrame(list(results.values())).sort_values(by=["score", "downloads", "likes"], ascending=[False, False, False])
    df_ex = pd.DataFrame(excluded)

    out = Path(args.out).resolve()
    df.to_csv(out, index=False)
    df_ex.to_csv(out.parent / f"excluded_{out.name}", index=False)

    with open(out.parent / "run_log.jsonl", "w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps({"id": row["id"], "query_hit": row["query_hit"], "score": row["score"]}) + "\n")

    print(f"\nSaved: {out}")
    print(f"Excluded: {out.parent / ('excluded_' + Path(args.out).name)}")
    print("Provenance log: run_log.jsonl")
    return df, df_ex


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest and triage compliance-related models from Hugging Face Hub (English-only keywords).")
    p.add_argument("--max-per-query", type=int, default=300, help="Max models per query to fetch (default: 300)")
    p.add_argument("--out", type=str, default="results.csv", help="Output CSV for kept models (default: results.csv)")
    p.add_argument("--min-downloads", type=int, default=0, help="Exclude models with downloads below this threshold (default: 0)")
    p.add_argument("--readme-scan", action="store_true", help="Download and scan README for ambiguous repos (slower, cached)")
    p.add_argument("--readme-cache", type=str, default="README_cache", help="Dir to cache READMEs when --readme-scan is set")
    p.add_argument("--embed-readme-snippet", action="store_true", help="Embed first 300 chars of README in evidence field")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between queries (be polite)")
    p.add_argument("--enforce-pipeline-whitelist", action="store_true", help="Lower confidence for models outside PIPELINE_WHITELIST")
    p.add_argument("--pipeline-whitelist-only", action="store_true", help="Hard exclude models outside PIPELINE_WHITELIST")
    p.add_argument("--query-subset", type=str, default="", help="Comma-separated substrings to select a subset of auto-generated queries (debugging)")
    p.add_argument("--network-retries", type=int, default=5, help="Max network retries per query (default: 5)")
    p.add_argument("--network-backoff", type=float, default=2.0, help="Base seconds for exponential backoff between retries (default: 2.0)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    print("Starting harvest with arguments:", args)
    df, df_ex = harvest(args)
    print(f"\nDone. Kept: {len(df)}  |  Excluded: {len(df_ex)}")


if __name__ == "__main__":
    main()
