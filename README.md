# 🧠 Vectorized Memories: Elastic Hybrid Search + Jina AI

This repository contains the architecture code for my **Elastic Blogathon 2026** submission. 

It demonstrates an enterprise-grade Customer Support Agent that eliminates LLM hallucinations by using:
1. **Elasticsearch** for Hybrid Retrieval (BM25 Keyword Search + kNN Vector Search).
2. **Reciprocal Rank Fusion (RRF)** to merge retrieval scores.
3. **Jina AI's Semantic Reranker (Cross-encoder)** to flawlessly order context before feeding it to an LLM.

### 🚀 Try it yourself
1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `ELASTIC_API_KEY` and `JINA_API_KEY` as environment variables.
4. Run `python app.py`
