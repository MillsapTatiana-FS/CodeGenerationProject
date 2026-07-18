import json
from pathlib import Path

def load_code_pairs(path: str = "data/code_pairs.json"):
    """
    Load the docstring → expected function pairs from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON dataset.

    Returns
    -------
    list[dict]
        A list of dictionaries with keys:
        - "docstring"
        - "expected"
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic validation to catch malformed entries early
    for i, item in enumerate(data):
        if "docstring" not in item or "expected" not in item:
            raise ValueError(f"Entry {i} is missing required keys.")

    return data


if __name__ == "__main__":
    # Quick manual test
    pairs = load_code_pairs()
    print(f"Loaded {len(pairs)} code pairs.")
    print("Example entry:")
    print(pairs[0])
