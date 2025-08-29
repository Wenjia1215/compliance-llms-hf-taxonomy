#!/usr/bin/env python3
# v3.1: crawl without ModelFilter (compatible with newer huggingface_hub)

import os, re, time, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterable

import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from huggingface_hub import HfApi, list_models, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

CORE_TERMS = ["compliance","security compliance","governance","risk","audit","attestation",
              "control","controls","control mapping","evidence","policy","policy mining",
              "assurance","trustworthiness","line-of-defense"]
FRAMEWORKS  = ["CIS Controls V8","CIS Controls","NIST SP 800-53","NIST 800-53","NIST SP 800-207",
               "NIST CSF","ISO 27001","SOC 2","PCI DSS","HIPAA","FedRAMP","CMMC","GDPR","CCPA","CPRA","SOX"]
TASK_TERMS  = ["classification","information extraction","mapping","retrieval","question answering",
               "evaluation","judge","scoring","reranking","generation","data labeling","evidence extraction",
               "verification","alignment"]
SAFETY_POLICY_TERMS = ["policy","safety policy","red teaming","hallucination","refusal","RLHF","safety eval","toxicity","bias"]
NEGATIVE_TERMS = ["stable diffusion","image generation","segmentation","tts","text-to-speech","audio","asr","whisper","speech recognition","music","diffusion"]
SEARCH_QUERIES_DEFAULT = ["compliance","security compliance","control mapping","policy mining","SOC 2","GDPR","HIPAA","PCI DSS","ISO 27001","NIST 800-53","CIS Controls","FedRAMP","CMMC"]

def sleep_polite(s: float): 
    if s and s>0: time.sleep(s)
def to_list(x): 
    return [] if x is None else ([str(i) for i in x] if isinstance(x,list) else [str(x)])
def safe_getattr(obj,n,default=None):
    try: return getattr(obj,n,default)
    except Exception: return default
def mask_token(t):
    if not t: return "<none>"
    t=str(t); return t[:6]+"..."+t[-4:] if len(t)>=12 else "<set>"
def guess_param_size(model):
    txt=" ".join([safe_getattr(model,'id','')," ".join(to_list(safe_getattr(model,'tags',[])))]).lower()
    m=re.search(r"\b(\d+)\s*(b|bn)\b",txt)
    return f"{m.group(1)}B" if m else None
def guess_domain(text:str):
    t=(text or "").lower()
    if any(k in t for k in ["hipaa","phi","medical","clinical","biomed","healthcare"]): return "Healthcare"
    if any(k in t for k in ["finra","sec ","mifid","sox","bank","finance","trading"]):  return "Finance"
    if any(k in t for k in ["gdpr","ccpa","cpra","privacy","dpa"]):                    return "Privacy"
    return None
def normalize_blob_text(parts: List[str]):
    raw=" ".join(p for p in parts if p)
    blob=re.sub(r"\s+"," ",raw.lower())
    blob_dash_spaced=re.sub(r"[-_/]+"," ",blob)
    blob_slug=re.sub(r"[ \t\r\n\-_\/]+","",blob)
    return blob, blob_dash_spaced, blob_slug
def keyword_variants(k:str):
    k=k.lower(); variants={k}
    if " " in k:
        variants.add(k.replace(" ","-"))
        variants.add(k.replace(" ","_"))
        variants.add(k.replace(" ",""))
        variants.add(re.sub(r"[ \-_]+"," ",k))
    return list(variants)
def fuzzy_contains_any(needles, haystacks, threshold=90):
    for n in needles:
        for h in haystacks:
            if n in h or fuzz.partial_ratio(n,h) >= threshold:
                return True
    return False
def match_keywords(blob, blob_dash_spaced, blob_slug):
    pos, frw, task = [], [], []
    hay=[blob, blob_dash_spaced, blob_slug]
    for k in CORE_TERMS:
        if fuzzy_contains_any(keyword_variants(k), hay): pos.append(k)
    for k in FRAMEWORKS:
        if fuzzy_contains_any(keyword_variants(k), hay): frw.append(k)
    for k in TASK_TERMS + SAFETY_POLICY_TERMS:
        if fuzzy_contains_any(keyword_variants(k), hay): task.append(k)
    return pos, frw, task
def has_negative(blob, blob_dash_spaced):
    hay=blob+" "+blob_dash_spaced
    for k in NEGATIVE_TERMS:
        if k.lower() in hay: return True
    return False
def extract_purpose(card, readme, category_fb):
    for key in ["intended_use","intended-use","intended uses","task","task_name","task_categories","tasks"]:
        try: val=card.get(key)
        except Exception: val=None
        if val:
            if isinstance(val,list): val=", ".join(map(str,val))
            val=str(val).strip()
            if val: return val, f"cardData.{key}"
    if readme:
        m=re.search(r"(?:^|\n)#{1,3}\s*intended\s*use[s]?\s*\n(.+?)(?:\n#{1,3}\s|\Z)",readme,flags=re.I|re.S)
        if m:
            body=re.sub(r"\s+"," ",m.group(1)).strip()
            first=re.split(r"(?<=[.?!])\s",body,maxsplit=1)[0][:240]
            return first,"readme[intended-use]"
    return category_fb,"fallback"
def load_readme(api, repo_id, cache_dir:Path, token):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file=cache_dir / (repo_id.replace("/","__") + ".md")
    if cache_file.exists():
        try: return cache_file.read_text(encoding="utf-8",errors="ignore")
        except Exception: pass
    try:
        info=api.model_info(repo_id, expand=["readme"], token=token)
        readme=getattr(info,"readme",None)
        txt = readme.decode("utf-8","ignore") if isinstance(readme,(bytes,bytearray)) else (str(readme) if readme is not None else "")
        cache_file.write_text(txt, encoding="utf-8",errors="ignore")
        return txt
    except Exception: pass
    try:
        p=hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model", local_dir=str(cache_dir), token=token)
        return Path(p).read_text(encoding="utf-8",errors="ignore")
    except Exception:
        return ""

def iter_models_search(queries, max_per_query, retries, backoff, token):
    seen_global=set()
    for q in queries:
        seen=set(); print(f"[query] {q}")
        attempts=0; remaining=max_per_query
        while remaining>0 and attempts<=retries:
            try:
                it=list_models(search=q, sort="last_modified", direction=-1, cardData=True, full=True, limit=remaining, token=(token or None))
                for m in it:
                    mid=safe_getattr(m,"id",None)
                    if not mid or mid in seen or mid in seen_global: continue
                    seen.add(mid); seen_global.add(mid)
                    yield m, q
                break
            except HfHubHTTPError:
                attempts+=1; sleep_polite(backoff*(2**attempts))
            except Exception:
                attempts+=1; sleep_polite(backoff*(2**attempts))

def iter_models_crawl(api: HfApi, library: Optional[str], pipeline_tags: List[str],
                      token: Optional[str], max_total: Optional[int]) -> Iterable:
    # No ModelFilter: call list_models for each pipeline_tag and de-duplicate
    seen: Set[str] = set()
    tags = pipeline_tags or [None]
    total = 0
    for tag in tags:
        kwargs = dict(cardData=True, full=True, token=token)
        if library: kwargs["library"] = library
        if tag:     kwargs["pipeline_tag"] = tag
        for m in api.list_models(**kwargs):
            mid = safe_getattr(m, "id", None)
            if not mid or mid in seen: 
                continue
            seen.add(mid)
            yield m, None
            total += 1
            if max_total and total >= max_total:
                return

@dataclass
class Args:
    mode: str
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
    no_query_expansion: bool
    queries: List[str]
    library: Optional[str]
    pipeline_tags: List[str]
    max_total: Optional[int]

def expand_query_variants(q):
    q=q.strip()
    if not q: return []
    s={q}
    if " " in q:
        s.add(q.replace(" ","-")); s.add(q.replace(" ","_")); s.add(q.replace(" ",""))
    return list(s)

def parse_args():
    ap=argparse.ArgumentParser("HF Compliance LLM collector v3.1 (search or crawl, no ModelFilter)")
    ap.add_argument("--mode", choices=["search","crawl"], default="crawl")
    ap.add_argument("--token", type=str, default=os.getenv("HUGGING_FACE_HUB_TOKEN",""))
    ap.add_argument("--out", type=str, default="results.csv")
    ap.add_argument("--excluded-out", type=str, default="excluded.csv")
    ap.add_argument("--max-per-query", type=int, default=200)
    ap.add_argument("--readme-scan", action="store_true")
    ap.add_argument("--readme-cache", type=str, default=".hf_readmes")
    ap.add_argument("--embed-readme-snippet", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--network-retries", type=int, default=3)
    ap.add_argument("--network-backoff", type=float, default=0.75)
    ap.add_argument("--extra-query", action="append", default=[])
    ap.add_argument("--no-query-expansion", action="store_true")
    ap.add_argument("--library", type=str, default="transformers")
    ap.add_argument("--pipeline-tags", type=str, default="text-generation,text2text-generation,question-answering")
    ap.add_argument("--max-total", type=int, default=None)
    a=ap.parse_args()

    base=SEARCH_QUERIES_DEFAULT + list(a.extra_query)
    if a.no_query_expansion:
        queries=list(dict.fromkeys(base))
    else:
        exp=set()
        for q in base: exp.update(expand_query_variants(q))
        queries=list(dict.fromkeys(exp))

    ptags=[p.strip() for p in (a.pipeline_tags or "").split(",") if p.strip()]

    return Args(a.mode, a.token or None, a.out, a.excluded_out, a.max_per_query, a.readme_scan,
                a.readme_cache, a.embed_readme_snippet, a.sleep, a.network_retries, a.network_backoff,
                a.no_query_expansion, queries, (a.library or None), ptags, a.max_total)

def main():
    args=parse_args()
    token=args.token or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
    api=HfApi(token=token)
    print(f"[auth] token: {mask_token(token)}")
    try:
        who=api.whoami(token=token) if token else None
        print(f"[auth] authenticated as: {(who.get('name') or who.get('user')) if who else 'anonymous'}")
    except Exception as e:
        print(f"[auth] whoami failed: {e}")

    results, excluded = {}, {}
    query_hits_map = {}

    iterator = iter_models_search(args.queries, args.max_per_query, args.network_retries, args.network_backoff, token) if args.mode=="search" \
               else iter_models_crawl(api, args.library, args.pipeline_tags, token, args.max_total)

    for pair in tqdm(iterator, total=None):
        try: model, q = pair
        except Exception: model, q = pair, None

        mid=safe_getattr(model,"id",None)
        if not mid: continue
        if q: query_hits_map.setdefault(mid,set()).add(q)

        card=safe_getattr(model,"cardData",{}) or {}
        tags=to_list(safe_getattr(model,"tags",[]))
        pipeline_tag=safe_getattr(model,"pipeline_tag","") or ""

        readme_txt = load_readme(api, mid, Path(args.readme_cache), token) if args.readme_scan else ""

        blob, blob_dash, blob_slug = normalize_blob_text([safe_getattr(model,"id",""), " ".join(tags), str(pipeline_tag), str(card or ""), str(readme_txt or "")])
        pos, frw, task = match_keywords(blob, blob_dash, blob_slug)
        neg = has_negative(blob, blob_dash)

        score = len(pos) + len(frw) + (len(task)//2)
        if neg: score -= 2

        purpose, purpose_src = extract_purpose(card, readme_txt, pipeline_tag or "model")
        snippet = (re.sub(r"\s+"," ",readme_txt).strip()[:350]) if (args.embed_readme_snippet and readme_txt) else ""

        owner = mid.split("/")[0] if "/" in mid else ""
        rec = {
            "id": mid, "owner": owner, "name": mid.split("/")[-1],
            "pipeline_tag": pipeline_tag, "tags": ", ".join(tags),
            "task_categories": ", ".join(to_list(card.get("task_categories"))) if isinstance(card,dict) else "",
            "downloads": safe_getattr(model,"downloads",None),
            "likes": safe_getattr(model,"likes",None),
            "created_at": str(safe_getattr(model,"created_at","")),
            "last_modified": str(safe_getattr(model,"last_modified","")),
            "params_guess": guess_param_size(model),
            "hf_url": f"https://huggingface.co/{mid}", "api_url": f"https://huggingface.co/api/models/{mid}",
            "purpose": purpose, "purpose_source": purpose_src,
            "domain_hint": guess_domain(readme_txt + " " + " ".join(tags)),
            "matched_keywords": "|".join(sorted(set(pos+frw+task))),
            "query_hits": "|".join(sorted(query_hits_map.get(mid,set()))),
            "score": score, "readme_snippet": snippet,
        }

        if score>0 and not neg: results[mid]=rec
        else: excluded[mid]=rec

        sleep_polite(args.sleep)

    res_df=pd.DataFrame(list(results.values()))
    exc_df=pd.DataFrame(list(excluded.values()))
    if not res_df.empty:
        cols=[c for c in ["score","likes","last_modified"] if c in res_df.columns]
        res_df=res_df.sort_values(by=cols, ascending=[False]*len(cols))
    if not exc_df.empty and "score" in exc_df.columns:
        exc_df=exc_df.sort_values(by=["score"], ascending=[True])

    res_df.to_csv(args.out, index=False, encoding="utf-8")
    exc_df.to_csv(args.excluded_out, index=False, encoding="utf-8")
    print(f"[done] wrote {args.out} ({len(res_df)} rows)")
    print(f"[done] wrote {args.excluded_out} ({len(exc_df)} rows)")

if __name__=="__main__":
    main()