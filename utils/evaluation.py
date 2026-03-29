import time

def measure_time(start, end):
    return end - start

def relevance_score(answer, docs):
    score = 0
    for doc in docs:
        if any(word in doc.lower() for word in answer.lower().split()):
            score += 1
    return score / len(docs)