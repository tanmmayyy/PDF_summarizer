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
