# compliance-llms-hf-taxonomy
This repository contains scripts, data, and artifacts for the taxonomy survey of compliance-oriented Large Language Models (LLMs) on Hugging Face. It includes reproducible search scripts, curated CSV registries, and supplementary materials for the accompanying survey paper.


### How to run scripts/hf_complianceLLM_Collector.py:
<code>
  python /Users/wenjia/Downloads/hf_complianceLLM_Collector.py \
  --max-per-query 300 \
  --min-downloads 10 \
  --readme-scan --embed-readme-snippet \
  --pipeline-whitelist-only \
  --network-retries 6 --network-backoff 2 \
  --sleep 0.2 \
  --out "$HOME/Downloads/hf_results.csv"
</code>
