#!/usr/bin/env python3
"""
HF Compliance-Oriented LLM Collector (with token support)
--------------------------------------------------------

What this does
- Queries Hugging Face models with multiple search strings (configurable)
- Authenticates with a read-only HF token (CLI flag or env var)
- Optionally pulls README (with token) and extracts a "purpose" field
- Produces results.csv and excluded.csv with taxonomy-friendly columns
- Adds quick-click links to each model's Hub page and JSON API endpoint
- Aggregates which queries matched each model (query_hits)

Install
  pip install --upgrade huggingface_hub pandas pyyaml rapidfuzz tqdm

Run (preferred: set env var instead of pasting token)
  export HUGGING_FACE_HUB_TOKEN=hf_xxx
  python hf_llm_collector_with_token.py \
      --readme-scan --embed-readme-snippet --sleep 0.25 \
      --out results.csv --excluded-out excluded.csv

Or pass token explicitly (avoid shell history when possible):
  python hf_llm_collector_with_token.py --token hf_xxx --readme-scan

Security
- Do NOT hardcode tokens in code or CSVs.
- This script never writes the token to disk and never appends it to URLs.
"""

import os
import re
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz

from huggingface_hub import HfApi, list_models, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


# -----------------------------
# Keyword sets (edit as needed)
# -----------------------------

CORE_TERMS = [
    "compliance", "security compliance", "governance", "risk", "audit", "attestation",
    "control", "controls", "control mapping", "evidence", "policy", "policy mining",
    "assurance", "trustworthiness", "line-of-defense"
]

FRAMEWORKS = [
    "CIS Controls V8", "CIS Controls", "NIST SP 800-53", "NIST 800-53", "NIST SP 800-207",
    "NIST CSF", "ISO 27001", "SOC 2", "PCI DSS", "HIPAA", "FedRAMP", "CMMC", "GDPR",
    "CCPA", "CPRA", "SOX"
]

TASK_TERMS = [
    "classification", "information extraction", "mapping", "retrieval", "question answering",
    "evaluation", "judge", "scoring", "reranking", "generation", "data labeling",
    "evidence extraction", "verification", "alignment"
]

SAFETY_POLICY_TERMS = [
    "policy", "safety policy", "red teaming", "hallucination", "refusal", "RLHF",
    "safety eval", "toxicity", "bias"
]

NEGATIVE_TERMS = [
    # helps push out irrelevant domains
    "stable diffusion", "image generation", "segmentation", "tts", "text-to-speech",
    "audio", "asr", "whisper", "speech recognition", "music", "diffusion"
]


SEARCH_QUERIES_DEFAULT = [
    # Feel free to add many more; duplicates are de-duped per run
    "compliance",
    "security compliance",
    "control mapping",
    "policy mining",
    "SOC 2",
    "GDPR",
    "HIPAA",
    "PCI DSS",
    "ISO 27001",
    "NIST 800-53",
    "CIS Controls",
    "FedRAMP",
    "CMMC",
]


# -----------------------------
# Utilities
# -----------------------------

def sleep_polite(seconds: float):
    if seconds and seconds > 0:
        time.sleep(seconds)


def to_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def safe_getattr(obj, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def guess_param_size(model) -> Optional[str]:
    txt = " ".join([safe_getattr(model, 'id', ''), " ".join(to_list(safe_getattr(model, 'tags', [])))]).lower()
    m = re.search(r"\b(\d+)\s*(b|bn)\b", txt)
    return f"{m.group(1)}B" if m else None


def guess_domain(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(k in t for k in ["hipaa", "phi", "medical", "clinical", "biomed", "healthcare"]):
        return "Healthcare"
    if any(k in t for k in ["finra", "sec ", "mifid", "sox", "bank", "finance", "trading"]):
        return "Finance"
    if any(k in t for k in ["gdpr", "ccpa", "cpra", "privacy", "dpa"]):
        return "Privacy"
    return None


def normalize_text_for_match(model, card: Dict, readme: str) -> str:
    blob_parts = [
        safe_getattr(model, "id", ""),
        " ".join(to_list(safe_getattr(model, "tags", []))),
        str(safe_getattr(model, "pipeline_tag", "") or ""),
        str(card or ""),
        str(readme or ""),
    ]
    blob = " ".join(blob_parts).lower()
    blob = re.sub(r"\s+", " ", blob)
    return blob


def fuzzy_contains(needle: str, haystack: str, threshold: int = 90) -> bool:
    needle_l = needle.lower()
    if needle_l in haystack:
        return True
    return fuzz.partial_ratio(needle_l, haystack) >= threshold


def match_keywords(blob: str) -> Tuple[List[str], List[str], List[str]]:
    pos = []
    frw = []
    task = []

    for k in CORE_TERMS:
        if fuzzy_contains(k, blob):
            pos.append(k)
    for k in FRAMEWORKS:
        if fuzzy_contains(k, blob):
            frw.append(k)
    for k in TASK_TERMS + SAFETY_POLICY_TERMS:
        if fuzzy_contains(k, blob):
            task.append(k)
    return pos, frw, task


def has_negative(blob: str) -> bool:
    for k in NEGATIVE_TERMS:
        if k.lower() in blob:
            return True
    return False


def extract_purpose(card: Dict, readme: str, category_fallback: str) -> Tuple[str, str]:
    # 1) structured cardData fields
    candidates = ["intended_use", "intended-use", "intended uses",
                  "task", "task_name", "task_categories", "tasks"]
    for key in candidates:
        val = None
        try:
            val = card.get(key)
        except Exception:
            val = None
        if val:
            if isinstance(val, list):
                val = ", ".join(map(str, val))
            val = str(val).strip()
            if val:
                return val, f"cardData.{key}"

    # 2) README section "Intended use"
    if readme:
        m = re.search(r"(?:^|\n)#{1,3}\s*intended\s*use[s]?\s*\n(.+?)(?:\n#{1,3}\s|\Z)",
                      readme, flags=re.I | re.S)
        if m:
            body = re.sub(r"\s+", " ", m.group(1)).strip()
            first_sentence = re.split(r"(?<=[.?!])\s", body, maxsplit=1)[0][:240]
            return first_sentence, "readme[intended-use]"

    return category_fallback, "fallback"


# -----------------------------
# README helpers
# -----------------------------

def load_readme(api: HfApi, repo_id: str, cache_dir: Path,
                token: Optional[str] = None) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (repo_id.replace("/", "__") + ".md")
    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    # Try model_info(expand=['readme']) first
    try:
        info = api.model_info(repo_id, expand=["readme"], token=token)
        readme = getattr(info, "readme", None)
        txt = ""
        if readme is not None:
            if isinstance(readme, (bytes, bytearray)):
                txt = readme.decode("utf-8", errors="ignore")
            else:
                txt = str(readme)
        cache_file.write_text(txt, encoding="utf-8", errors="ignore")
        return txt
    except TypeError:
        # older hubs or signature mismatch: fall back to README.md
        pass
    except HfHubHTTPError:
        pass
    except Exception:
        pass

    # Fallback to downloading README.md directly
    try:
        p = hf_hub_download(repo_id=repo_id, filename="README.md",
                            repo_type="model", local_dir=str(cache_dir),
                            token=token)
        return Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# -----------------------------
# Model iteration with retries
# -----------------------------

def iter_models_with_retries(search_q: str,
                             limit: int,
                             max_retries: int,
                             backoff_base: float,
                             seen: Set[str],
                             token: Optional[str] = None):
    remaining = limit
    attempts = 0
    while remaining > 0 and attempts <= max_retries:
        try:
            iterator = list_models(
                search=search_q,
                sort="last_modified",
                direction=-1,
                cardData=True,
                full=True,
                limit=remaining,
                token=(token or None),
            )
            for m in iterator:
                mid = safe_getattr(m, "id", None)
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                remaining = limit - len(seen)
                yield m
            break  # finished this query
        except HfHubHTTPError as e:
            attempts += 1
            sleep_polite(backoff_base * (2 ** attempts))
        except Exception:
            attempts += 1
            sleep_polite(backoff_base * (2 ** attempts))


# -----------------------------
# Main harvest
# -----------------------------

@dataclass
class Args:
    token: Optional[str]
    out: str
    excluded_out: str
    max_per_query: int
    readme_scan: bool
    readme_cache: str
    embed_readme_snippet: bool
    sleep: float
    network_retries: int
    network_backoff: float
    queries: List[str]


def parse_args() -> Args:
    ap = argparse.ArgumentParser("HF Compliance LLM collector (token-aware)")
    ap.add_argument("--token", type=str, default=os.getenv("HUGGING_FACE_HUB_TOKEN", ""),
                    help="HF token value (or set env HUGGING_FACE_HUB_TOKEN).")
    ap.add_argument("--out", type=str, default="results.csv", help="Results CSV path.")
    ap.add_argument("--excluded-out", type=str, default="excluded.csv",
                    help="Excluded models CSV path.")
    ap.add_argument("--max-per-query", type=int, default=200,
                    help="Max results per search query.")
    ap.add_argument("--readme-scan", action="store_true",
                    help="Download/read README to enrich fields like purpose.")
    ap.add_argument("--readme-cache", type=str, default=".hf_readmes",
                    help="Cache directory for READMEs.")
    ap.add_argument("--embed-readme-snippet", action="store_true",
                    help="Embed a short README snippet into results.csv")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Polite sleep between models (seconds).")
    ap.add_argument("--network-retries", type=int, default=3,
                    help="Retries on hub errors per query.")
    ap.add_argument("--network-backoff", type=float, default=0.75,
                    help="Exponential backoff base seconds.")
    ap.add_argument("--extra-query", action="append", default=[],
                    help="Add extra search query (can repeat).")

    a = ap.parse_args()

    queries = list(dict.fromkeys(SEARCH_QUERIES_DEFAULT + list(a.extra_query)))  # de-dupe
    return Args(
        token=a.token or None,
        out=a.out,
        excluded_out=a.excluded_out,
        max_per_query=a.max_per_query,
        readme_scan=a.readme_scan,
        readme_cache=a.readme_cache,
        embed_readme_snippet=a.embed_readme_snippet,
        sleep=a.sleep,
        network_retries=a.network_retries,
        network_backoff=a.network_backoff,
        queries=queries,
    )


def harvest(args: Args):
    token = args.token or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
    api = HfApi(token=token)

    results: Dict[str, Dict] = {}
    excluded: Dict[str, Dict] = {}
    query_hits_map: Dict[str, Set[str]] = {}

    for q in args.queries:
        seen_for_q: Set[str] = set()
        print(f"[query] {q}")
        for model in tqdm(iter_models_with_retries(
                search_q=q,
                limit=args.max_per_query,
                max_retries=args.network_retries,
                backoff_base=args.network_backoff,
                seen=seen_for_q,
                token=token),
                total=args.max_per_query, leave=False):

            mid = safe_getattr(model, "id", None)
            if not mid:
                continue

            query_hits_map.setdefault(mid, set()).add(q)

            card: Dict = safe_getattr(model, "cardData", {}) or {}
            tags = to_list(safe_getattr(model, "tags", []))
            pipeline_tag = safe_getattr(model, "pipeline_tag", "") or ""

            readme_txt = ""
            if args.readme_scan:
                readme_txt = load_readme(api, mid, Path(args.readme_cache), token)

            blob = normalize_text_for_match(model, card, readme_txt)
            pos, frw, task = match_keywords(blob)
            neg = has_negative(blob)

            score = len(pos) + len(frw) + (len(task) // 2)  # task half-weight
            if neg:
                score -= 2

            # derive purpose
            category_fb = pipeline_tag or "model"
            purpose, purpose_src = extract_purpose(card, readme_txt, category_fb)

            if args.embed_readme_snippet and readme_txt:
                snippet = re.sub(r"\s+", " ", readme_txt).strip()[:350]
            else:
                snippet = ""

            owner = mid.split("/")[0] if "/" in mid else ""
            record = {
                "id": mid,
                "owner": owner,
                "name": mid.split("/")[-1],
                "pipeline_tag": pipeline_tag,
                "tags": ", ".join(tags),
                "task_categories": ", ".join(to_list(card.get("task_categories"))) if isinstance(card, dict) else "",
                "downloads": safe_getattr(model, "downloads", None),
                "likes": safe_getattr(model, "likes", None),
                "created_at": str(safe_getattr(model, "created_at", "")),
                "last_modified": str(safe_getattr(model, "last_modified", "")),
                "params_guess": guess_param_size(model),
                "hf_url": f"https://huggingface.co/{mid}",
                "api_url": f"https://huggingface.co/api/models/{mid}",
                "purpose": purpose,
                "purpose_source": purpose_src,
                "domain_hint": guess_domain(readme_txt + " " + " ".join(tags)),
                "matched_keywords": "|".join(sorted(set(pos + frw + task))),
                "query_hits": "|".join(sorted(query_hits_map.get(mid, set()))),
                "score": score,
                "readme_snippet": snippet,
            }

            # Decide bucket
            if score > 0 and not neg:
                results[mid] = record
            else:
                excluded[mid] = record

            sleep_polite(args.sleep)

    # Write CSVs
    res_df = pd.DataFrame(list(results.values()))
    exc_df = pd.DataFrame(list(excluded.values()))

    # Sort results by score desc then likes/last_modified if available
    if not res_df.empty:
        sort_cols = [c for c in ["score", "likes", "last_modified"] if c in res_df.columns]
        res_df = res_df.sort_values(by=sort_cols, ascending=[False, False, False])
    if not exc_df.empty:
        exc_df = exc_df.sort_values(by=["score"] if "score" in exc_df.columns else [], ascending=[True])

    res_df.to_csv(args.out, index=False, encoding="utf-8")
    exc_df.to_csv(args.excluded_out, index=False, encoding="utf-8")

    print(f"[done] wrote {args.out} ({len(res_df)} rows)")
    print(f"[done] wrote {args.excluded_out} ({len(exc_df)} rows)")


if __name__ == "__main__":
    args = parse_args()
    harvest(args)
