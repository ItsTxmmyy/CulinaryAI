#!/usr/bin/env python3
"""
Culinary Manuscripts CSV Parser

Extracts recipe transcripts from the Culinary Manuscripts database dump
and prepares them for training data generation.

Usage:
    python parse_culinary_csv.py [--output transcripts.json] [--analyze]
"""

import csv
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


def clean_transcript(text: str) -> str:
    """Clean and normalize transcript text.
    
    Minimal preprocessing - let the AI model handle historical
    abbreviations and variations from context.
    """
    if not text:
        return ""
    
    # Historical typography (safe substitutions only)
    text = text.replace('ſ', 's')      # Long S → modern s
    
    # OCR artifacts
    text = text.replace('¬', '-')      # Hyphen artifacts from line breaks
    
    # Strip page numbers (common at start/end of transcripts)
    text = re.sub(r'^\s*\d+\s*\n', '', text)         # Leading page number
    text = re.sub(r'\n\s*\d+\s*$', '', text)         # Trailing page number
    text = re.sub(r'^\s*Page\s+\d+\s*\n', '', text, flags=re.IGNORECASE)  # "Page 23"
    
    # Whitespace normalization (preserve paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def recipe_confidence(transcript: str) -> int:
    """
    Calculate confidence score for recipe content.
    Returns number of recipe indicators found (0 = not a recipe, higher = more confident).
    """
    if not transcript or len(transcript) < 50:
        return 0
    
    # Skip cover pages, blank pages, indexes, etc.
    skip_patterns = [
        r'^Front cover$',
        r'^Back cover$',
        r'^Blank page',
        r'^Index',
        r'^\[blank\]',
        r'^Table of Contents',
    ]
    for pattern in skip_patterns:
        if re.match(pattern, transcript, re.IGNORECASE):
            return 0
    
    # Recipe indicators - common phrases in historical recipes
    recipe_indicators = [
        r'\bTake\b',           # "Take 3 eggs..."
        r'\bBoil[e]?\b',       # Cooking method
        r'\bBake\b',
        r'\bRoast\b',
        r'\bFry\b',
        r'\bStew\b',
        r'\bMix\b',
        r'\bPound\b',          # Common in old recipes
        r'\bPut\b.*\bin\b',    # "Put it in..."
        r'\bLet it\b',         # "Let it boil..."
        r'\bTo make\b',        # "To make a cake..."
        r'\bTo dress\b',       # "To dress a dish..."
        r'\bTo preserve\b',
        r'\bReceipt\b',        # Old word for recipe
        r'\bRecipe\b',
        r'\bpound[s]?\b',      # Measurements
        r'\bounce[s]?\b',
        r'\bpint[s]?\b',
        r'\bquart[s]?\b',
        r'\bspoon\b',
        r'\begg[s]?\b',
        r'\bflour\b',
        r'\bsugar\b',
        r'\bbutter\b',
        r'\bcream\b',
    ]
    
    return sum(1 for pattern in recipe_indicators 
               if re.search(pattern, transcript, re.IGNORECASE))


def is_recipe_content(transcript: str) -> bool:
    """Check if transcript is recipe content (requires 3+ indicators)."""
    return recipe_confidence(transcript) >= 3


def split_recipes(transcript: str) -> list[str]:
    """
    Split a transcript that may contain multiple recipes.
    
    Detects recipe boundaries by looking for title patterns like:
    - "To make/pot/roast X"
    - "A receipt for X"
    - "Sauce for X"
    """
    # Pattern matches common recipe title starts
    split_pattern = r'(?=\n(?:To (?:make|pot|dress|roast|boil|bake|fry|stew|pickle|preserve|candy)|Sauce for|A receipt|An? (?:excellent|good|fine) )\b)'
    
    parts = re.split(split_pattern, transcript, flags=re.IGNORECASE)
    
    # Clean up and filter out empty/tiny fragments
    recipes = []
    for part in parts:
        part = part.strip()
        if len(part) >= 50:  # Must be substantial
            recipes.append(part)
    
    # If no split occurred, return the original as a single-item list
    return recipes if recipes else [transcript]


def extract_recipe_titles(transcript: str) -> list[str]:
    """
    Extract potential recipe titles from transcript.
    Historical recipes often start with "To [verb] [noun]" or "A [noun]"
    """
    titles = []
    
    # Pattern for "To make/dress/preserve X" style titles
    to_patterns = re.findall(
        r'^(To (?:make|dress|preserve|pot|roast|boil|bake|fry|stew|pickle|candy)\s+[\w\s\-\']+?)(?:\.|$|\n)',
        transcript,
        re.MULTILINE | re.IGNORECASE
    )
    titles.extend(to_patterns)
    
    # Pattern for "A [recipe name]" or "An [recipe name]"
    a_patterns = re.findall(
        r'^(An?\s+(?:receipt|recipe|excellent|good|fine)?\s*[\w\s\-\']+?)(?:\.|$|\n)',
        transcript,
        re.MULTILINE | re.IGNORECASE
    )
    titles.extend(a_patterns)
    
    # Clean up titles
    cleaned = []
    for title in titles:
        title = title.strip()
        # Remove trailing numbers (page references)
        title = re.sub(r'\s+\d+$', '', title)
        if len(title) > 5 and len(title) < 100:
            cleaned.append(title)
    
    return cleaned


def parse_csv(csv_path: str) -> list[dict]:
    """
    Parse the Culinary Manuscripts CSV and extract transcripts.
    
    Returns a list of records with manuscript metadata and transcript text.
    Splits pages with multiple recipes into separate records.
    """
    records = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            transcript = clean_transcript(row.get('transcript', ''))
            
            if not is_recipe_content(transcript):
                continue
            
            # Split pages that contain multiple recipes
            recipe_parts = split_recipes(transcript)
            
            for i, recipe_text in enumerate(recipe_parts):
                # Only include parts that are actually recipes
                if not is_recipe_content(recipe_text):
                    continue
                
                record = {
                    'item_id': row.get('item_id', ''),
                    'manuscript_title': row.get('main_object_title', ''),
                    'page': row.get('part_title', ''),
                    'recipe_index': i + 1 if len(recipe_parts) > 1 else None,  # Track if split
                    'transcript': recipe_text,
                    'detected_titles': extract_recipe_titles(recipe_text),
                    'char_count': len(recipe_text),
                    'confidence': recipe_confidence(recipe_text),
                }
                records.append(record)
    
    return records


def group_by_manuscript(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by manuscript for context."""
    grouped = defaultdict(list)
    for record in records:
        key = f"{record['item_id']}_{record['manuscript_title']}"
        grouped[key].append(record)
    
    # Sort pages within each manuscript
    for key in grouped:
        grouped[key].sort(key=lambda x: x['page'])
    
    return dict(grouped)


def analyze_transcripts(records: list[dict]) -> dict:
    """Generate statistics about the extracted transcripts."""
    stats = {
        'total_records': len(records),
        'total_characters': sum(r['char_count'] for r in records),
        'avg_chars_per_record': 0,
        'manuscripts': len(set(r['item_id'] for r in records)),
        'recipes_with_titles': sum(1 for r in records if r['detected_titles']),
        'unique_detected_titles': set(),
        'char_distribution': {
            '<200': 0,
            '200-500': 0,
            '500-1000': 0,
            '1000-2000': 0,
            '>2000': 0
        }
    }
    
    for record in records:
        # Character distribution
        chars = record['char_count']
        if chars < 200:
            stats['char_distribution']['<200'] += 1
        elif chars < 500:
            stats['char_distribution']['200-500'] += 1
        elif chars < 1000:
            stats['char_distribution']['500-1000'] += 1
        elif chars < 2000:
            stats['char_distribution']['1000-2000'] += 1
        else:
            stats['char_distribution']['>2000'] += 1
        
        # Collect unique titles
        for title in record['detected_titles']:
            stats['unique_detected_titles'].add(title.lower())
    
    if records:
        stats['avg_chars_per_record'] = stats['total_characters'] // len(records)
    
    stats['unique_detected_titles'] = len(stats['unique_detected_titles'])
    
    return stats


def create_training_example(instruction: str, input_text: str, output_json: str) -> dict:
    """Create a training example in the expected format."""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_json
    }


def main():
    parser = argparse.ArgumentParser(description='Parse Culinary Manuscripts CSV')
    parser.add_argument('--input', '-i', 
                        default='culinary_collection7_20251218_141432.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', 
                        default='extracted_transcripts.json',
                        help='Output JSON file path')
    parser.add_argument('--analyze', '-a', 
                        action='store_true',
                        help='Print analysis statistics')
    parser.add_argument('--sample', '-s',
                        type=int,
                        default=0,
                        help='Print N sample transcripts')
    parser.add_argument('--min-chars',
                        type=int,
                        default=100,
                        help='Minimum character count for transcripts')
    
    args = parser.parse_args()
    
    print(f"Parsing {args.input}...")
    records = parse_csv(args.input)
    
    # Filter by minimum character count
    records = [r for r in records if r['char_count'] >= args.min_chars]
    
    print(f"Found {len(records)} recipe transcripts")
    
    if args.analyze:
        stats = analyze_transcripts(records)
        print("\n📊 Analysis Statistics:")
        print(f"  Total recipe pages: {stats['total_records']}")
        print(f"  Unique manuscripts: {stats['manuscripts']}")
        print(f"  Total characters: {stats['total_characters']:,}")
        print(f"  Average chars/record: {stats['avg_chars_per_record']}")
        print(f"  Pages with detected titles: {stats['recipes_with_titles']}")
        print(f"  Unique recipe titles found: {stats['unique_detected_titles']}")
        print("\n  Character distribution:")
        for range_name, count in stats['char_distribution'].items():
            print(f"    {range_name}: {count}")
    
    if args.sample > 0:
        print(f"\n📜 Sample Transcripts (first {args.sample}):")
        for i, record in enumerate(records[:args.sample]):
            print(f"\n--- Record {i+1} ---")
            print(f"Manuscript: {record['manuscript_title']}")
            print(f"Page: {record['page']}")
            print(f"Detected titles: {record['detected_titles']}")
            print(f"Transcript preview: {record['transcript'][:500]}...")
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(records)} transcripts to {output_path}")


if __name__ == '__main__':
    main()

