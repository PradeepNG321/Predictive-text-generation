# backend/predict_transformer.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# load a small GPT-2 model (lightweight for CPU)
MODEL_NAME = "distilgpt2"  # smaller than gpt2, runs fine on laptops

print("🔄 Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# put model in eval mode (no training here)
model.eval()

def transformer_predict(text: str, k: int = 5):
    """Return top-k likely next words using a GPT-2 family model."""
    if not text.strip():
        text = "hello"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    # get probabilities for next token
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    topk = torch.topk(probs, k)
    tokens = [tokenizer.decode([int(t)]) for t in topk.indices]
    return [t.strip() for t in tokens if t.strip()]
