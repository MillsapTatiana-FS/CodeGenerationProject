from src.data import load_code_pairs
from src.model import load_model
from src.inference import generate_code
from src.eval import evaluate

FEW_SHOT_TEMPLATE = """
Below are examples of Python functions created from docstrings.

Example 1:
Docstring: "Write a function that checks if a number is prime."
Function:
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

Example 2:
Docstring: "Write a function that reverses a string."
Function:
def reverse_string(s):
    return s[::-1]

Now follow the pattern.

Docstring: "{docstring}"
Function:
"""

def run_all_experiments():
    pairs = load_code_pairs()
    gen = load_model()

    results = []

    for item in pairs:
        doc = item["docstring"]
        expected = item["expected"]

        # Zero-shot
        zero_shot_prompt = doc
        generated_zero = generate_code(gen, zero_shot_prompt)
        metrics_zero = evaluate(generated_zero, expected)

        # Few-shot
        few_shot_prompt = FEW_SHOT_TEMPLATE.format(docstring=doc)
        generated_few = generate_code(gen, few_shot_prompt)
        metrics_few = evaluate(generated_few, expected)

        # Chain-of-thought
        cot_prompt = f"{doc}\n\nThink step by step about the logic needed, then provide the final function."
        generated_cot = generate_code(gen, cot_prompt)
        metrics_cot = evaluate(generated_cot, expected)

        result_entry = {
            "docstring": doc,
            "zero_shot": metrics_zero,
            "few_shot": metrics_few,
            "cot": metrics_cot,
        }

        print(result_entry)
        results.append(result_entry)

    return results


if __name__ == "__main__":
    run_all_experiments()
