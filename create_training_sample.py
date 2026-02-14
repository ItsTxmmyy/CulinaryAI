#!/usr/bin/env python3
"""
Create a smaller training sample from extracted_transcripts.json

Selects high-quality recipe transcripts for training Meta-Llama.
Prioritizes recipes with:
- Detected titles
- Higher confidence scores
- Diverse manuscripts
- Good transcript length (not too short, not too long)

Usage:
    python create_training_sample.py [--count 100] [--output training_sample.json]
"""

import json
import argparse
import random
from pathlib import Path


def load_transcripts(input_path: str) -> list[dict]:
    """Load the full transcripts file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def score_record(record: dict) -> float:
    """
    Score a record for quality. Higher is better.
    """
    score = 0.0
    
    # Confidence score (0-10 range typically)
    score += record.get('confidence', 0) * 2
    
    # Has detected titles (good indicator of recipe structure)
    if record.get('detected_titles'):
        score += 10
    
    # Ideal length: 300-1500 chars (not too short, not too long)
    char_count = record.get('char_count', 0)
    if 300 <= char_count <= 1500:
        score += 10
    elif 200 <= char_count <= 2000:
        score += 5
    
    # Penalize very short or very long
    if char_count < 100:
        score -= 10
    if char_count > 3000:
        score -= 5
    
    return score


def select_diverse_sample(records: list[dict], count: int) -> list[dict]:
    """
    Select a diverse sample of records.
    Ensures variety across manuscripts while prioritizing quality.
    """
    # Score all records
    scored = [(record, score_record(record)) for record in records]
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Take top candidates (3x the desired count for diversity selection)
    top_candidates = scored[:count * 3]
    
    # Select diverse sample across manuscripts
    selected = []
    manuscripts_used = {}
    max_per_manuscript = max(3, count // 20)  # Limit per manuscript for diversity
    
    for record, score in top_candidates:
        if len(selected) >= count:
            break
            
        manuscript = record.get('manuscript_title', 'unknown')
        
        # Limit how many we take from each manuscript
        if manuscripts_used.get(manuscript, 0) >= max_per_manuscript:
            continue
        
        selected.append(record)
        manuscripts_used[manuscript] = manuscripts_used.get(manuscript, 0) + 1
    
    # If we still need more, fill from remaining top candidates
    if len(selected) < count:
        for record, score in top_candidates:
            if len(selected) >= count:
                break
            if record not in selected:
                selected.append(record)
    
    return selected


def print_stats(records: list[dict], label: str):
    """Print statistics about a set of records."""
    if not records:
        print(f"{label}: No records")
        return
    
    manuscripts = set(r.get('manuscript_title') for r in records)
    with_titles = sum(1 for r in records if r.get('detected_titles'))
    total_chars = sum(r.get('char_count', 0) for r in records)
    avg_confidence = sum(r.get('confidence', 0) for r in records) / len(records)
    
    print(f"\n{label}:")
    print(f"  Records: {len(records)}")
    print(f"  Unique manuscripts: {len(manuscripts)}")
    print(f"  With detected titles: {with_titles} ({100*with_titles/len(records):.1f}%)")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Avg chars/record: {total_chars // len(records)}")
    print(f"  Avg confidence: {avg_confidence:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Create training sample from transcripts')
    parser.add_argument('--input', '-i',
                        default='extracted_transcripts.json',
                        help='Input JSON file path')
    parser.add_argument('--output', '-o',
                        default='training_sample.json',
                        help='Output JSON file path')
    parser.add_argument('--count', '-c',
                        type=int,
                        default=100,
                        help='Number of records to select')
    parser.add_argument('--seed', '-s',
                        type=int,
                        default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sort', 
                        action='store_true',
                        help='Sort output by item_id (chronological order)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Loading {args.input}...")
    records = load_transcripts(args.input)
    print_stats(records, "Full dataset")
    
    print(f"\nSelecting {args.count} high-quality, diverse samples...")
    sample = select_diverse_sample(records, args.count)
    print_stats(sample, "Training sample")
    
    # Sort by item_id and page if requested
    if args.sort:
        sample.sort(key=lambda x: (
            int(x.get('item_id', '0')) if x.get('item_id', '').isdigit() else 0,
            x.get('page', ''),
            x.get('recipe_index') or 0
        ))
        print("\nSorted by item_id (chronological order)")
    
    # Save the sample
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(sample)} records to {output_path}")


if __name__ == '__main__':
    main()
