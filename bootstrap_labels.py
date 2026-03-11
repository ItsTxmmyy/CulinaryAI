from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from env import load_hf_token

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        "transformers is not installed. Install it with:\n"
        "  pip install transformers accelerate torch huggingface_hub"
    ) from exc


def load_model_and_tokenizer() -> tuple[Any, Any, str]:
    hf_token = load_hf_token()
    model_id = os.environ.get(
        "LLAMA_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=None,
    ).to(device)

    return model, tokenizer, device


def build_prompt(example: Dict[str, Any]) -> str:
    instruction: str = example["instruction"]
    input_payload: Dict[str, Any] = example["input"]

    manuscript_title = input_payload.get("manuscript_title")
    page = input_payload.get("page")
    item_id = input_payload.get("item_id")
    transcript = input_payload.get("transcript")

    header_lines = []
    if manuscript_title:
        header_lines.append(f"Manuscript title: {manuscript_title}")
    if page:
        header_lines.append(f"Page: {page}")
    if item_id:
        header_lines.append(f"Item ID: {item_id}")

    header = "\n".join(header_lines)

    return (
        f"{instruction}\n\n"
        "Here is the source material:\n"
        f"{header}\n\n"
        "Transcript:\n"
        f"{transcript}\n"
    )


def generate_label(
    model: Any,
    tokenizer: Any,
    device: str,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the part after the prompt to reduce duplication.
    return text[len(prompt) :].strip()


def bootstrap_labels(
    input_path: Path,
    output_path: Path,
    start: int = 0,
    limit: int | None = None,
) -> None:
    model, tokenizer, device = load_model_and_tokenizer()

    input_lines = input_path.read_text(encoding="utf-8").splitlines()
    total = len(input_lines)

    if limit is not None:
        end = min(start + limit, total)
    else:
        end = total

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, line in enumerate(input_lines):
            if idx < start or idx >= end:
                # Keep original example unchanged outside the target slice.
                out_f.write(line + "\n")
                continue

            example = json.loads(line)
            prompt = build_prompt(example)
            label = generate_label(model, tokenizer, device, prompt)

            example["output"] = label
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap output labels for training.jsonl using a Llama model. "
            "By default, processes all examples."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training.jsonl"),
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_bootstrapped.jsonl"),
        help="Path to output JSONL file with labels.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based) of examples to label.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to label.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for each label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    bootstrap_labels(
        input_path=args.input,
        output_path=args.output,
        start=args.start,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

