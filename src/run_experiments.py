from src.data import load_code_pairs
from src.model import load_model
from src.inference import generate_code
from src.eval import evaluate
import json
from pathlib import Path

def save_results(results, filename="experiment_results.json"):
    metrics_path = Path("results/metrics")
    metrics_path.mkdir(parents=True, exist_ok=True)

    file_path = metrics_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {file_path}")


def save_generations(docstring, zero, few, cot):
    gen_path = Path("results/generations")
    gen_path.mkdir(parents=True, exist_ok=True)

    filename = docstring[:40].replace(" ", "_") + ".json"
    file_path = gen_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({
            "docstring": docstring,
            "zero_shot": zero,
            "few_shot": few,
            "cot": cot
        }, f, indent=4)


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
    print("run_all_experiments() started")

    pairs = load_code_pairs()
    print("Loaded pairs:", len(pairs))
    
    gen = load_model()
    print("Model loaded")
    
    results = []

    for item in pairs:
        doc = item["docstring"]
        expected = item["expected"]

        print("Starting:", doc)

        # Zero-shot
        zero_shot_prompt = f"{doc}\n\nProvide ONLY Python code. No explanations."
        generated_zero = generate_code(gen, zero_shot_prompt)
        metrics_zero = evaluate(generated_zero, expected)

        # Few-shot
        few_shot_prompt = FEW_SHOT_TEMPLATE.format(docstring=doc) + "\nProvide ONLY Python code. No explanations."
        generated_few = generate_code(gen, few_shot_prompt)
        metrics_few = evaluate(generated_few, expected)

        # Chain-of-thought
        cot_prompt = f"{doc}\n\nThink step by step, then provide ONLY the final Python function."
        generated_cot = generate_code(gen, cot_prompt)
        metrics_cot = evaluate(generated_cot, expected)

        # Debug prints 
        print("\nProcessing:", doc)
        print("Zero-shot metrics:", metrics_zero)
        print("Few-shot metrics:", metrics_few)
        print("CoT metrics:", metrics_cot)

        # Save generations
        save_generations(doc, generated_zero, generated_few, generated_cot)

        # Save metrics
        result_entry = {
            "docstring": doc,
            "zero_shot": metrics_zero,
            "few_shot": metrics_few,
            "cot": metrics_cot,
        }

        results.append(result_entry)

    save_results(results)
    return results



if __name__ == "__main__":
    run_all_experiments()
