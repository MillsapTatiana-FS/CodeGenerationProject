from src.model import load_model

def generate_code(generator, prompt: str) -> str:
    """
    Generate Python code from a natural language prompt.
    """
    output = generator(
        prompt,
        max_new_tokens=160,
        temperature=0.7,
        do_sample=True
    )
    return output[0]["generated_text"]


if __name__ == "__main__":
    gen = load_model()
    prompt = "Write a Python function that returns the factorial of a number."
    print(generate_code(gen, prompt))
