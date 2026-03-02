from __future__ import annotations

import os

import torch
from env import load_hf_token

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        "transformers is not installed. Install it with:\n"
        "  pip install transformers accelerate torch huggingface_hub bitsandbytes"
    ) from exc


def main() -> None:
    hf_token = load_hf_token()

    model_id = os.environ.get(
        "LLAMA_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model: {model_id} on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=None,
    ).to(device)

    prompt = (
        "Write a very short, two-sentence description of a classic French omelette."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    print("\n=== Model output ===\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

