#!/usr/bin/env python3
"""
Curate high-quality training data from extracted_transcripts.json

Improvements over create_training_sample.py:
1. Splits multi-recipe records into individual recipes
2. Filters out truncated/incomplete recipes
3. Re-scores and selects the best clean, complete recipes

Usage:
    python curate_training_data.py [--count 100] [--output curated_training.json]
    python curate_training_data.py --count 200 --analyze
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import Counter


def load_transcripts(input_path: str) -> list[dict]:
    """Load the full transcripts file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: Split multi-recipe records into individual recipes
# ---------------------------------------------------------------------------

# Title patterns that signal the start of a new recipe
RECIPE_TITLE_PATTERNS = [
    # "To make/dress/preserve/pot/roast/boil/bake/fry/stew/pickle/cure/candy X"
    r'To (?:make|dress|preserve|pot|roast|boil|bake|fry|stew|pickle|cure|candy|keep|dry|collar|hash|ragoo|fricassee|fricasey|fricossey|blanch|clarify|distill|brew|bottle|prepare|clean|cook|do)\s+',
    # "A/An [adjective] [dish name]"
    r'An?\s+(?:receipt|recipe|excellent|good|fine|rich|plain|common|nice|baked|boiled|fried|cheap)\s+',
    # Short capitalised title lines (e.g. "Plum Pudding", "Walnut Catchup")
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\s*$',
]

# Compile a combined pattern that matches a title line on its own line
_TITLE_LINE_RE = re.compile(
    r'^(' + '|'.join(RECIPE_TITLE_PATTERNS) + r')',
    re.MULTILINE | re.IGNORECASE,
)


def _looks_like_recipe_title(line: str) -> bool:
    """Check if a standalone line looks like a recipe title."""
    line = line.strip()
    if not line or len(line) > 120:
        return False
    # Must start with uppercase
    if not line[0].isupper():
        return False
    # Should not contain sentence-like structure (periods mid-text, long clauses)
    if '. ' in line and not line.endswith('.'):
        return False
    # Match against known title patterns
    for pat in RECIPE_TITLE_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return True
    # "Receipt for X" pattern (catches non-food receipts too)
    if re.match(r'Receipt\s+for\b', line, re.IGNORECASE):
        return True
    # Short capitalised phrase (2-6 words, mostly capitalised)
    words = line.rstrip('.:-–—').split()
    if 1 <= len(words) <= 7:
        caps = sum(1 for w in words if w[0].isupper())
        if caps >= len(words) * 0.5:
            return True
    return False


def split_multi_recipe(transcript: str) -> list[str]:
    """
    Split a transcript containing multiple recipes into individual ones.

    Detects recipe boundaries by finding title-like lines that appear
    after the first ~60 characters (so we don't split on the first title).
    """
    lines = transcript.split('\n')
    boundaries = [0]  # First recipe always starts at line 0
    char_pos = 0

    for i, line in enumerate(lines):
        if i == 0:
            char_pos += len(line) + 1
            continue
        # Only consider boundaries after the opening of the record
        if char_pos > 60 and _looks_like_recipe_title(line):
            # Check that the previous line is blank or very short (title separator)
            prev = lines[i - 1].strip() if i > 0 else ''
            if len(prev) == 0 or prev.endswith(('.', ';', ':', '-', '—', '–', '_')):
                boundaries.append(i)
        char_pos += len(line) + 1

    if len(boundaries) == 1:
        return [transcript]

    recipes = []
    for idx in range(len(boundaries)):
        start = boundaries[idx]
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        chunk = '\n'.join(lines[start:end]).strip()
        if len(chunk) >= 80:
            recipes.append(chunk)

    return recipes if recipes else [transcript]


# ---------------------------------------------------------------------------
# Step 2: Filter truncated / incomplete recipes
# ---------------------------------------------------------------------------

def is_complete_recipe(transcript: str) -> bool:
    """
    Check if a recipe appears to be complete (not cut off mid-sentence).

    A complete recipe typically ends with:
    - A period, semicolon, colon, or closing punctuation
    - An attribution line ("Mrs Roberts", "1742", etc.)
    - A dash or em-dash (common in historical manuscripts)
    - A closing parenthesis or bracket

    Rejects recipes ending with manuscript damage markers.
    """
    text = transcript.rstrip()
    if not text:
        return False

    # Reject if it ends with a damage/uncertainty marker
    # e.g. [illegible], [text missing], [?], {?], [Citronn?], Mrs [Tilh?]
    if re.search(r'\[\w*\?\]$', text) or re.search(r'\{\?\]$', text):
        return False
    if re.search(r'\[(?:illegible|text missing|torn|damaged|faded)\w*\]$', text, re.IGNORECASE):
        return False
    # Attribution with damaged name: "Mrs [Tilh?]"
    if re.search(r'\[\w*\?\]\s*$', text):
        return False

    last_char = text[-1]

    # Ends with sentence-terminating punctuation
    if last_char in '.;:!?)]\'"':
        return True

    # Ends with a dash (common manuscript ending)
    if last_char in '-–—_=':
        return True

    # Check last line — could be an attribution or short note
    last_line = text.split('\n')[-1].strip()

    # Attribution patterns: "Mrs Roberts", "1742", "Mrs Patrick"
    if re.match(r'^(?:Mrs?|Miss|Lady|Sir|Dr)\.?\s+\w+', last_line, re.IGNORECASE):
        # But reject if the attribution name itself is damaged
        if re.search(r'\[\w*\?\]', last_line):
            return False
        return True
    if re.match(r'^\d{4}', last_line):
        return True

    return False


# ---------------------------------------------------------------------------
# Step 3: Scoring and selection
# ---------------------------------------------------------------------------

def recipe_confidence(transcript: str) -> int:
    """
    Calculate confidence score for recipe content.
    Mirrors parse_culinary_csv.py logic.
    """
    if not transcript or len(transcript) < 50:
        return 0

    recipe_indicators = [
        r'\bTake\b', r'\bBoil[e]?\b', r'\bBake\b', r'\bRoast\b',
        r'\bFry\b', r'\bStew\b', r'\bMix\b', r'\bPound\b',
        r'\bPut\b.*\bin\b', r'\bLet it\b', r'\bTo make\b',
        r'\bTo dress\b', r'\bTo preserve\b', r'\bReceipt\b',
        r'\bRecipe\b', r'\bpound[s]?\b', r'\bounce[s]?\b',
        r'\bpint[s]?\b', r'\bquart[s]?\b', r'\bspoon\b',
        r'\begg[s]?\b', r'\bflour\b', r'\bsugar\b', r'\bbutter\b',
        r'\bcream\b',
    ]

    return sum(1 for p in recipe_indicators
               if re.search(p, transcript, re.IGNORECASE))


def extract_recipe_titles(transcript: str) -> list[str]:
    """Extract potential recipe titles from transcript."""
    titles = []

    to_patterns = re.findall(
        r'^(To (?:make|dress|preserve|pot|roast|boil|bake|fry|stew|pickle|candy|cure|keep|dry|cook)\s+[\w\s\-\']+?)(?:\.|$|\n)',
        transcript,
        re.MULTILINE | re.IGNORECASE,
    )
    titles.extend(to_patterns)

    a_patterns = re.findall(
        r'^(An?\s+(?:receipt|recipe|excellent|good|fine|rich|plain|common)?\s*[\w\s\-\']+?)(?:\.|$|\n)',
        transcript,
        re.MULTILINE | re.IGNORECASE,
    )
    titles.extend(a_patterns)

    # Short capitalised title at the start
    first_line = transcript.split('\n')[0].strip()
    if first_line and len(first_line) < 80 and first_line[0].isupper():
        words = first_line.rstrip('.:-').split()
        if 1 <= len(words) <= 7 and first_line not in titles:
            titles.insert(0, first_line)

    cleaned = []
    for title in titles:
        title = title.strip()
        title = re.sub(r'\s+\d+$', '', title)
        if 3 < len(title) < 100 and title not in cleaned:
            cleaned.append(title)

    return cleaned


def score_record(record: dict) -> float:
    """Score a curated record for quality. Higher is better."""
    score = 0.0

    score += record.get('confidence', 0) * 2

    if record.get('detected_titles'):
        score += 10

    char_count = record.get('char_count', 0)
    if 200 <= char_count <= 1200:
        score += 15
    elif 150 <= char_count <= 1800:
        score += 8

    if char_count < 80:
        score -= 15
    if char_count > 3000:
        score -= 5

    # Penalise [illegible] — mild penalty, not disqualifying
    illegible_count = record.get('transcript', '').lower().count('[illegible]')
    score -= illegible_count * 2

    # Penalise damage markers: [?], [torn], [faded], [text missing], etc.
    damage_count = len(re.findall(r'\[(?:\w*\?|illegible|text missing|torn|damaged|faded)\]',
                                  record.get('transcript', ''), re.IGNORECASE))
    score -= damage_count * 3

    # Bonus for records that came from a clean single-recipe page
    if not record.get('was_split'):
        score += 3

    return score


def select_diverse_sample(records: list[dict], count: int) -> list[dict]:
    """Select a diverse sample across manuscripts, prioritising quality."""
    scored = [(r, score_record(r)) for r in records]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_candidates = scored[:count * 4]

    selected = []
    manuscripts_used: dict[str, int] = {}
    max_per_manuscript = max(3, count // 15)

    for record, _score in top_candidates:
        if len(selected) >= count:
            break
        ms = record.get('manuscript_title', 'unknown')
        if manuscripts_used.get(ms, 0) >= max_per_manuscript:
            continue
        selected.append(record)
        manuscripts_used[ms] = manuscripts_used.get(ms, 0) + 1

    # Backfill if diversity limits left us short
    if len(selected) < count:
        selected_set = set(id(r) for r in selected)
        for record, _score in top_candidates:
            if len(selected) >= count:
                break
            if id(record) not in selected_set:
                selected.append(record)

    return selected


# ---------------------------------------------------------------------------
# Stats & reporting
# ---------------------------------------------------------------------------

def print_stats(records: list[dict], label: str):
    """Print statistics about a set of records."""
    if not records:
        print(f"{label}: No records")
        return

    manuscripts = set(r.get('manuscript_title') for r in records)
    with_titles = sum(1 for r in records if r.get('detected_titles'))
    total_chars = sum(r.get('char_count', 0) for r in records)
    avg_confidence = sum(r.get('confidence', 0) for r in records) / len(records)
    illegible = sum(1 for r in records if '[illegible]' in r.get('transcript', '').lower())
    chars = [r.get('char_count', 0) for r in records]

    print(f"\n{label}:")
    print(f"  Records:             {len(records)}")
    print(f"  Unique manuscripts:  {len(manuscripts)}")
    print(f"  With detected titles:{with_titles} ({100*with_titles/len(records):.1f}%)")
    print(f"  Total characters:    {total_chars:,}")
    print(f"  Avg chars/record:    {total_chars // len(records)}")
    print(f"  Char range:          {min(chars)} - {max(chars)}")
    print(f"  Avg confidence:      {avg_confidence:.1f}")
    print(f"  With [illegible]:    {illegible} ({100*illegible/len(records):.1f}%)")


def print_detailed_analysis(records: list[dict]):
    """Print detailed breakdown for review."""
    print("\n  Confidence distribution:")
    conf_dist = Counter(r.get('confidence', 0) for r in records)
    for c in sorted(conf_dist):
        print(f"    {c:>2}: {conf_dist[c]} records")

    print("\n  Manuscripts represented:")
    ms_dist = Counter(r.get('manuscript_title') for r in records)
    for ms, cnt in ms_dist.most_common():
        print(f"    [{cnt}] {ms}")


# ---------------------------------------------------------------------------
# Post-validation: catch edge cases the earlier filters missed
# ---------------------------------------------------------------------------

def _passes_post_validation(record: dict, bracket_end_re: re.Pattern) -> bool:
    """
    Final quality gate applied after selection.
    Catches edge cases that slip through the broader filters.
    """
    t = record.get('transcript', '')
    stripped = t.rstrip()
    if not stripped:
        return False

    # Reject transcripts ending with bracket annotations: [word], [word?], [moderate]
    if bracket_end_re.search(stripped):
        return False

    # Reject transcripts that start mid-sentence (bad split artifact)
    first_line = stripped.split('\n')[0].strip()
    if first_line and first_line[0].islower():
        return False

    # Reject near-empty first lines that look like page artifacts
    if first_line and re.match(r'^\[?line\s+across\]?$', first_line, re.IGNORECASE):
        return False

    # Check for "To [verb]" appearing after char 60 on its own line (missed multi-recipe)
    lines = stripped.split('\n')
    char_pos = 0
    for i, line in enumerate(lines):
        if i == 0:
            char_pos += len(line) + 1
            continue
        s = line.strip()
        if char_pos > 60 and s:
            prev = lines[i - 1].strip() if i > 0 else ''
            if (not prev or prev.endswith(('.', ';', ':', '-', '\u2014'))) and \
               re.match(r'^To (?:make|dress|preserve|pot|roast|boil|bake|fry|stew|pickle|cure|candy|keep)\b', s, re.IGNORECASE):
                return False
        char_pos += len(line) + 1

    # Check for duplicate content: transcript starts with a numbered recipe ref
    # that indicates it was extracted from a numbered list poorly
    # (e.g. "22. Jumbals" is fine, just won't have a detected title — allow it)

    return True


def _post_validate(records: list[dict], bracket_end_re: re.Pattern) -> tuple[list[dict], list[dict]]:
    """Split records into passing and rejected lists, including deduplication."""
    passed = []
    rejected = []
    seen_keys: set[str] = set()

    for r in records:
        # Deduplication by first 100 chars of transcript
        dedup_key = r.get('transcript', '')[:100].lower().strip()
        if dedup_key in seen_keys:
            title = r.get('detected_titles', ['Untitled'])[0] if r.get('detected_titles') else 'Untitled'
            print(f"  Rejected (duplicate): {title} ({r.get('manuscript_title', '?')} / {r.get('page', '?')})")
            rejected.append(r)
            continue
        seen_keys.add(dedup_key)

        if _passes_post_validation(r, bracket_end_re):
            passed.append(r)
        else:
            title = r.get('detected_titles', ['Untitled'])[0] if r.get('detected_titles') else 'Untitled'
            print(f"  Rejected: {title} ({r.get('manuscript_title', '?')} / {r.get('page', '?')})")
            rejected.append(r)
    if not rejected:
        print("  All records passed post-validation")
    else:
        print(f"  Rejected {len(rejected)} records")
    return passed, rejected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Curate high-quality training data with recipe splitting and completeness filtering'
    )
    parser.add_argument('--input', '-i',
                        default='extracted_transcripts.json',
                        help='Input JSON file path')
    parser.add_argument('--output', '-o',
                        default='curated_training.json',
                        help='Output JSON file path')
    parser.add_argument('--count', '-c',
                        type=int, default=100,
                        help='Number of records to select')
    parser.add_argument('--seed',
                        type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--min-confidence',
                        type=int, default=5,
                        help='Minimum confidence score for inclusion')
    parser.add_argument('--analyze', '-a',
                        action='store_true',
                        help='Print detailed analysis of curated data')
    parser.add_argument('--sort',
                        action='store_true',
                        help='Sort output by item_id')

    args = parser.parse_args()
    random.seed(args.seed)

    # Load
    print(f"Loading {args.input}...")
    raw_records = load_transcripts(args.input)
    print_stats(raw_records, "Source dataset")

    # Step 1: Split multi-recipe records
    print("\n--- Step 1: Splitting multi-recipe records ---")
    split_records = []
    split_count = 0
    for record in raw_records:
        parts = split_multi_recipe(record['transcript'])
        if len(parts) > 1:
            split_count += 1
        for i, part in enumerate(parts):
            new_record = {
                'item_id': record['item_id'],
                'manuscript_title': record['manuscript_title'],
                'page': record['page'],
                'recipe_index': i + 1 if len(parts) > 1 else record.get('recipe_index'),
                'transcript': part,
                'detected_titles': extract_recipe_titles(part),
                'char_count': len(part),
                'confidence': recipe_confidence(part),
                'was_split': len(parts) > 1,
            }
            split_records.append(new_record)

    print(f"  Split {split_count} multi-recipe records")
    print(f"  Total records after splitting: {len(split_records)}")

    # Step 2: Filter incomplete recipes
    print("\n--- Step 2: Filtering incomplete/truncated recipes ---")
    complete_records = [r for r in split_records if is_complete_recipe(r['transcript'])]
    removed = len(split_records) - len(complete_records)
    print(f"  Removed {removed} truncated records ({100*removed/len(split_records):.1f}%)")
    print(f"  Remaining: {len(complete_records)}")

    # Step 3: Filter by minimum confidence
    print(f"\n--- Step 3: Filtering by confidence >= {args.min_confidence} ---")
    confident_records = [r for r in complete_records
                         if r.get('confidence', 0) >= args.min_confidence]
    removed_conf = len(complete_records) - len(confident_records)
    print(f"  Removed {removed_conf} low-confidence records")
    print(f"  Remaining: {len(confident_records)}")

    # Step 3b: Filter out non-food recipes (cleaning, medicine, cosmetics mixed in)
    NON_FOOD_PATTERNS = [
        r'\bcleaning\s+(?:plate|silver|brass|copper|furniture)',
        r'\bpolish(?:ing)?\s+(?:plate|silver|brass)',
        r'\bremoving?\s+(?:stains|grease|ink)',
        r'\bwashing\s+(?:lace|silk|linen)',
    ]
    non_food_re = re.compile('|'.join(NON_FOOD_PATTERNS), re.IGNORECASE)

    def is_food_recipe(record: dict) -> bool:
        return not non_food_re.search(record.get('transcript', ''))

    food_records = [r for r in confident_records if is_food_recipe(r)]
    removed_nonfood = len(confident_records) - len(food_records)
    if removed_nonfood:
        print(f"\n--- Step 3b: Removing non-food recipes ---")
        print(f"  Removed {removed_nonfood} non-food records (cleaning, polishing, etc.)")
        print(f"  Remaining: {len(food_records)}")
    confident_records = food_records

    print_stats(confident_records, "Eligible pool")

    # Step 4: Select diverse sample
    print(f"\n--- Step 4: Selecting {args.count} diverse, high-quality records ---")
    if len(confident_records) < args.count:
        print(f"  Warning: only {len(confident_records)} eligible records "
              f"(requested {args.count}). Using all.")
        sample = confident_records
    else:
        sample = select_diverse_sample(confident_records, args.count)

    # Step 5: Post-validation — catch edge cases the filters missed
    print("\n--- Step 5: Post-validation sweep ---")
    bracket_end_re = re.compile(r'\[[\w\s]*\??\]\s*$')
    sample, rejected = _post_validate(sample, bracket_end_re)

    # Backfill rejected records with next-best from pool
    if rejected:
        selected_ids = set(id(r) for r in sample)
        scored_pool = [(r, score_record(r)) for r in confident_records
                       if id(r) not in selected_ids]
        scored_pool.sort(key=lambda x: x[1], reverse=True)
        backfill = []
        for r, _ in scored_pool:
            if len(backfill) >= len(rejected):
                break
            ok, _ = _post_validate([r], bracket_end_re)
            if ok:
                backfill.append(r)
        sample.extend(backfill)
        print(f"  Replaced with {len(backfill)} clean records from pool")

    # Remove internal field before saving
    for r in sample:
        r.pop('was_split', None)

    if args.sort:
        sample.sort(key=lambda x: (
            int(x.get('item_id', '0')) if x.get('item_id', '').isdigit() else 0,
            x.get('page', ''),
            x.get('recipe_index') or 0,
        ))

    print_stats(sample, "Curated training sample")

    if args.analyze:
        print_detailed_analysis(sample)

    # Save
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(sample)} curated records to {output_path}")


if __name__ == '__main__':
    main()
