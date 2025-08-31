# compliance-llms-hf-taxonomy
This repository contains scripts, data, and artifacts for the taxonomy survey of compliance-oriented Large Language Models (LLMs) on Hugging Face. It includes reproducible search scripts, curated CSV registries, and supplementary materials for the accompanying survey paper.


### How to run <a href="https://github.com/Wenjia1215/compliance-llms-hf-taxonomy/blob/main/scripts/hf_complianceLLM_Collector_v1.py">hf_complianceLLM_Collector.py</a>:
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




### How to run <a href="https://github.com/Wenjia1215/compliance-llms-hf-taxonomy/blob/main/scripts/hf_llm_collector_with_token.py">hf_complianceLLM_Collector.py</a>:
<code>python hf_llm_collector_with_token.py \
  --readme-scan --embed-readme-snippet \
  --sleep 0.25 \
  --out results.csv --excluded-out excluded.csv
</code>
<img width="3548" height="1664" alt="image" src="https://github.com/user-attachments/assets/932b7873-7301-4ad0-9a7f-df29f294bc24" />


<img width="2742" height="1072" alt="image" src="https://github.com/user-attachments/assets/7f2d43de-35b9-4bce-8f4c-74e58be76188" />
