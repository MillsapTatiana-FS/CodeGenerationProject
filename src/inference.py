from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
model_id = "google/gemma-2-2b-it"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
gen = pipeline("text-generation", model=model, tokenizer=tok)
resp = gen("Summarize in 3 bullets: Large Language Models enable...", 
           max_new_tokens=160, do_sample=True, temperature=0.7)
print(resp[0]["generated_text"])