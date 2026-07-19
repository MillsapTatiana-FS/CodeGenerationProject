from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def load_model(model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Load the text-generation model and tokenizer.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier.

    Returns
    -------
    transformers.Pipeline
        A text-generation pipeline ready for inference.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,              # override the default 20
        return_full_text=False,      # prevents prompt repetition
        clean_up_tokenization_spaces=False
)

    return generator


if __name__ == "__main__":
    gen = load_model()
    print("Model loaded successfully.")
