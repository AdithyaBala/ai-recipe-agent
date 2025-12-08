# src/data_prep.py

import argparse
import ast
import json
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions

RAW_PATH = Path("data/RAW_recipes.csv")
OUT_PATH = Path("data/recipes_clean.json")
CHROMA_DIR = Path("models/chroma")
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def build_chroma_index(recipes, batch_size: int = 500, max_records: int | None = None):
    """
    Persist recipe embeddings into a local ChromaDB store for fast semantic lookup.
    """
    if not recipes:
        print("No recipes to index; skipping Chroma build.")
        return

    if max_records:
        recipes = recipes[:max_records]
        print(f"Limiting to first {len(recipes)} recipes for indexing.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )

    # Recreate the collection to avoid duplicate entries on reruns.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embed_fn,
    )

    total = len(recipes)
    for start in range(0, total, batch_size):
        batch = recipes[start : start + batch_size]
        ids = []
        documents = []
        metadatas = []
        for idx, recipe in enumerate(batch, start=start):
            ids.append(f"recipe-{idx}")
            documents.append(" ".join(recipe["ingredients"]))
            metadatas.append(
                {
                    "title": recipe["title"],
                    # Chroma metadata values must be primitives; store list as JSON string.
                    "ingredients_json": json.dumps(
                        recipe["ingredients"], ensure_ascii=False
                    ),
                    "instructions": recipe["instructions"],
                }
            )

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        if (start // batch_size) % 20 == 0 or start + batch_size >= total:
            print(
                f"Indexed {min(start + batch_size, total)} / {total} recipes into Chroma..."
            )

    print(f"Saved {total} recipes to Chroma at {CHROMA_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit number of recipes indexed (useful for quick tests).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of recipes to embed per Chroma add batch.",
    )
    args = parser.parse_args()

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
    build_chroma_index(records, batch_size=args.batch_size, max_records=args.limit)


if __name__ == "__main__":
    main()
