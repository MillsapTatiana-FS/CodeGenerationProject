from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
gen = pipeline("text-generation", model=model, tokenizer=tok)
resp = gen("Write a Python function that returns the factorial of a number.", 
           max_new_tokens=160, do_sample=True, temperature=0.7)
print(resp[0]["generated_text"])