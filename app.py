import os
import requests
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 1. SETUP & CONFIGURATION
ELASTIC_URL = os.getenv("ELASTIC_URL", "https://your-elastic-cluster.com")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "your_elastic_api_key")
JINA_API_KEY = os.getenv("JINA_API_KEY", "your_jina_api_key")

es = Elasticsearch(ELASTIC_URL, api_key=ELASTIC_API_KEY)

# 2. HYBRID SEARCH FUNCTION
def retrieve_hybrid_context(user_query: str, query_vector: list) -> list:
    response = es.search(
        index="support_kb",
        body={
            "retriever": {
                "rrf": {
                    "retrievers":[
                        {"standard": {"query": {"multi_match": {"query": user_query, "fields":["title", "content"]}}}},
                        {"knn": {"field": "content_vector", "query_vector": query_vector, "k": 10, "num_candidates": 50}}
                    ],
                    "rank_constant": 60
                }
            },
            "size": 10,
            "_source": ["title", "content"]
        }
    )
    return [hit["_source"]["content"] for hit in response["hits"]["hits"]]

# 3. JINA AI RERANKING FUNCTION
def rerank_with_jina(query: str, documents: list) -> list:
    if not documents: return[]
    url = "https://api.jina.ai/v1/rerank"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {JINA_API_KEY}"}
    payload = {"model": "jina-reranker-v2-base-multilingual", "query": query, "documents": documents, "top_n": 3}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200: return documents[:3]
    return [documents[res["index"]] for res in response.json().get("results",[])]
