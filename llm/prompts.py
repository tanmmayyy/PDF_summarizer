SYSTEM_PROMPT = """
You are an AI assistant.
Answer only from the given context.
If answer is not found, say "Not available".
"""


def build_prompt(context, question):

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt
