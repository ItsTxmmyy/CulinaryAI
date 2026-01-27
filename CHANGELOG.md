# Changelog

All notable changes to the CulinaryAI project will be documented in this file.

## [Unreleased] - 2026-01-26

### Added
- **Confidence scoring**: Each recipe record now includes a `confidence` field (0-26 scale) based on the number of recipe indicators detected. Higher scores indicate more reliable training data.
- **Multi-recipe page splitting**: Pages containing multiple recipes are now split into separate records, each tracked with a `recipe_index` field.
- **Long S normalization**: Historical typography character (ſ) is now converted to modern 's'.
- **Page number stripping**: Removes standalone page numbers and "Page X" headers from transcript start/end.
- **OCR artifact cleaning**: Removes common OCR artifacts like hyphen line-break characters (¬).

### Changed
- `clean_transcript()` now performs minimal, safe preprocessing while letting the AI model learn historical abbreviations from context.
- `is_recipe_content()` now uses the new `recipe_confidence()` function internally.
- Output records include two new fields: `confidence` and `recipe_index`.
- `extracted_transcripts.json` increased from ~64K to ~87K lines due to recipe splitting.

### Technical Notes
- Minimum confidence threshold remains at 3 indicators for recipe detection.
- Consider filtering training data to `confidence >= 8` for highest quality examples.

