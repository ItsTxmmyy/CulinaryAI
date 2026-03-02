from __future__ import annotations

import os

from env import load_hf_token


def main() -> None:
    token = load_hf_token()
    model_id = os.environ.get(
        "LLAMA_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    print(f"Model: {model_id}")
    print("1) Checking token + access (fast)...")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install it with:\n"
            "  pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    info = api.model_info(model_id)
    print(f"   - Access OK. Repo id: {info.id}")

    cfg_path = hf_hub_download(repo_id=model_id, filename="config.json", token=token)
    print("   - Downloaded config.json")
    print(f"   - Local path: {cfg_path}")

    print("\n2) Trying quick generation via Hugging Face hosted inference (if enabled)...")
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("   - InferenceClient not available in this huggingface_hub version.")
        return

    prompt = "Reply with exactly 6 words: 'Llama wired up and working.'"
    try:
        client = InferenceClient(model=model_id, token=token, timeout=30)
        text = client.text_generation(prompt, max_new_tokens=12, temperature=0.0)
        print("\n=== Remote generation output ===\n")
        print(text)
    except Exception as exc:
        print("   - Hosted inference not available (or not enabled for this model).")
        print(f"   - Details: {type(exc).__name__}: {exc}")
        print("\nIf you want, we can still do local generation, but it will take longer on MPS.")


if __name__ == "__main__":
    main()

