import argparse
import json
from pathlib import Path
from typing import Any, Dict


CANONICAL_INSTRUCTION = (
    'You are an expert historical recipe normalizer. '
    'Convert the given manuscript recipe transcript and its manuscript metadata into a single '
    'Schema.org Recipe JSON-LD object. Use "@context": "https://schema.org" and "@type": "Recipe". '
    'Populate at least "name", "recipeIngredient" (as a list of ingredient strings), and '
    '"recipeInstructions" (as a list of step strings). When helpful, also include "description", '
    '"recipeYield", "totalTime", and "author". Infer reasonable modernized field values from the '
    'text, but do not invent information that is not supported by the transcript. '
    'Output strictly valid JSON-LD only, with no extra commentary or explanation. '
    "The JSON must have no trailing commas and must pass the Recipe schema validation."
)


def build_input_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the input payload shown to the model for a single example.

    We expose the curated manuscript metadata plus the transcript text.
    """
    return {
        "manuscript_title": record.get("manuscript_title"),
        "page": record.get("page"),
        "item_id": record.get("item_id"),
        "transcript": record.get("transcript"),
    }


def convert_curated_to_jsonl(
    input_path: Path,
    output_path: Path,
    max_examples: int | None = None,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if max_examples is not None:
        records = records[:max_examples]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for record in records:
            example = {
                "instruction": CANONICAL_INSTRUCTION,
                "input": build_input_payload(record),
                # The output/label will be populated later by a separate
                # labeling / bootstrapping pipeline.
                "output": None,
            }
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate JSONL training examples from curated_training.json. "
            "Each line contains {instruction, input, output}."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("curated_training.json"),
        help="Path to curated input JSON (default: curated_training.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training.jsonl"),
        help="Path to JSONL output file (default: training.jsonl).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional maximum number of examples to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    convert_curated_to_jsonl(
        input_path=args.input,
        output_path=args.output,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()

