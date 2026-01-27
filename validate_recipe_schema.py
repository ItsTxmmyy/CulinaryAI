import json
import os
from jsonschema import validate, ValidationError

# Schema.org Recipe validator
RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "@context": {"type": "string", "pattern": "^https?://schema.org/?$"},
        "@type": {"const": "Recipe"},
        "name": {"type": "string"},
        "recipeIngredient": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        },
        "recipeInstructions": {
            "type": "array",
            "items": {"type": "object"}
        }
    },
    "required": ["@context", "@type", "name", "recipeIngredient", "recipeInstructions"]
}

def validate_recipes(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jsonld"):
            path = os.path.join(directory, filename)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Check against our schema
                validate(instance=data, schema=RECIPE_SCHEMA)
                print(f" {filename}: Valid Schema.org Recipe")
                
            except json.JSONDecodeError:
                print(f" {filename}: Invalid JSON syntax")
            except ValidationError as e:
                print(f" {filename}: Schema error - {e.message}")
            except Exception as e:
                print(f" {filename}: Unexpected error: {e}")

# Usage: Place your .jsonld files in a folder named 'recipes'
# validate_recipes('./recipes')