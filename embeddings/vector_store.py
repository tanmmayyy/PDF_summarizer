import faiss
import pickle
import os


def create_faiss_index(embeddings):

    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index


def save_index(index, texts, path="data/processed"):

    os.makedirs(path, exist_ok=True)

    faiss.write_index(index, f"{path}/index.faiss")

    with open(f"{path}/texts.pkl", "wb") as f:
        pickle.dump(texts, f)


def load_index(path="data/processed"):

    index = faiss.read_index(f"{path}/index.faiss")

    with open(f"{path}/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    return index, texts
