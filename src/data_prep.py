# src/data_prep.py

import ast
import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/RAW_recipes.csv")
OUT_PATH = Path("data/recipes_clean.json")


def safe_list_parse(x):
    """
    RAW_recipes stores 'ingredients' and 'steps' as stringified Python lists.
    This turns the string into a real list of lowercased strings.
    """
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return [str(v).strip().lower() for v in val]
        return []
    except Exception:
        return []


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Put RAW_recipes.csv into the data/ folder."
        )

    # Only load the columns we actually need
    usecols = ["name", "ingredients", "steps"]
    df = pd.read_csv(RAW_PATH, usecols=usecols)

    records = []

    for _, row in df.iterrows():
        title = str(row["name"])
        ingredients = safe_list_parse(row["ingredients"])
        steps = safe_list_parse(row["steps"])

        if not ingredients or not steps:
            continue

        instructions = " ".join(steps)

        records.append(
            {
                "title": title,
                "ingredients": ingredients,
                "instructions": instructions,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    print(f"Saved {len(records)} cleaned recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
