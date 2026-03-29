from rag.retriever import retrieve
from llm.prompts import build_prompt
from llm.llm_loader import generate_response


def run_rag(query, embed_model, index, texts):

    docs = retrieve(
        query,
        embed_model,
        index,
        texts
    )

    context = "\n".join(docs)

    prompt = build_prompt(
        context,
        query
    )

    answer = generate_response(prompt)

    return answer


def rewrite_query(query):
    return f"Answer clearly using only the given context: {query}"

import time
from utils.evaluation import measure_time, relevance_score

def run_rag(query, embed_model, index, texts):

    start = time.time()

    query = rewrite_query(query)

    docs, scores = retrieve(query, embed_model, index, texts)

    context = "\n".join(docs)

    prompt = build_prompt(context, query)

    answer = generate_response(prompt)

    end = time.time()

    return {
        "answer": answer,
        "docs": docs,
        "time": measure_time(start, end),
        "relevance": relevance_score(answer, docs)
    }