import numpy as np


def retrieve(query, model, index, texts, k=3):

    query_vec = model.encode([query])

    distances, indices = index.search(
        np.array(query_vec), k
    )

    results = []

    for i in indices[0]:
        results.append(texts[i])

    return results


def retrieve(query, model, index, texts, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)

    results = []
    scores = []

    for idx, dist in zip(indices[0], distances[0]):
        results.append(texts[idx])
        scores.append(dist)

    return results, scores