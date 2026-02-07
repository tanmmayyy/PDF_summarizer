import ollama


def generate_response(prompt, model="phi"):
    

    response = ollama.generate(
        model=model,
        prompt=prompt
    )

    return response["response"]
