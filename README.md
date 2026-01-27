# CulinaryAI

A tool for extracting and processing historical recipe transcripts from the Culinary Manuscripts database for AI training data generation.

## Overview

CulinaryAI parses CSV exports from culinary manuscript collections, identifies pages containing recipe content using heuristic analysis, and extracts structured data suitable for machine learning training sets.

## Features

- **Intelligent Recipe Detection**: Uses pattern matching to identify recipe content from manuscript transcripts, filtering out cover pages, indexes, and non-recipe content
- **Historical Recipe Title Extraction**: Detects recipe titles in historical formats (e.g., "To make a cake", "A receipt for pudding")
- **Manuscript Grouping**: Organizes extracted recipes by source manuscript
- **Statistical Analysis**: Generates insights about the extracted corpus
- **Schema.org Validation**: Validates recipe JSON-LD files against Schema.org Recipe standards

## Installation

### Prerequisites

- Python 3.12+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ItsTxmmyy/CulinaryAI.git
   cd CulinaryAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Parsing Culinary Manuscripts

```bash
python parse_culinary_csv.py [options]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input CSV file path | `culinary_collection7_20251218_141432.csv` |
| `-o, --output` | Output JSON file path | `extracted_transcripts.json` |
| `-a, --analyze` | Print analysis statistics | `False` |
| `-s, --sample N` | Print N sample transcripts | `0` |
| `--min-chars N` | Minimum character count for transcripts | `100` |

#### Examples

Extract transcripts with analysis:
```bash
python parse_culinary_csv.py --analyze
```

Extract with custom input/output and view samples:
```bash
python parse_culinary_csv.py -i my_data.csv -o output.json --sample 5
```

### Validating Recipe Schema

```bash
python validate_recipe_schema.py
```

Validates `.jsonld` files against the Schema.org Recipe specification.

## Output Format

The parser outputs a JSON file containing an array of recipe records:

```json
{
  "item_id": "12345",
  "manuscript_title": "Mrs. Smith's Cookbook",
  "page": "Page 23",
  "transcript": "To make a cake. Take three eggs...",
  "detected_titles": ["To make a cake"],
  "char_count": 450
}
```

## Project Structure

```
CulinaryAI/
├── parse_culinary_csv.py      # Main CSV parser for manuscript transcripts
├── validate_recipe_schema.py  # Schema.org Recipe validator
├── requirements.txt           # Python dependencies
├── extracted_transcripts.json # Output from parser (generated)
└── *.jsonld                   # Sample Schema.org recipe files
```

## License