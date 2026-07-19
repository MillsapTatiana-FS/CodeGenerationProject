from src.model import load_model

def generate_code(generator, prompt: str) -> str:
    output = generator(
        prompt,
        max_new_tokens=120,      # enough for full functions
        temperature=0.0,
        do_sample=False
    )
    return output[0]["generated_text"].strip()



if __name__ == "__main__":
    gen = load_model()
    prompt = "Write a Python function that returns the factorial of a number."
    print(generate_code(gen, prompt))
