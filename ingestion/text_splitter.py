def split_text(text, chunk_size=500, overlap=50):
    
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

    return chunks


def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks