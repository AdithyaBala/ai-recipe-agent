# src/retrieval.py

import json
from functools import lru_cache
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DATA_PATH = Path("data/recipes_clean.json")
CHROMA_DIR = Path("models/chroma")
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_recipes():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python -m src.data_prep` first."
        )
    recipes = json.loads(DATA_PATH.read_text())
    return recipes


@lru_cache(maxsize=1)
def _embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def _get_client():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"{CHROMA_DIR} not found. Run `python -m src.data_prep` to build the index."
        )
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def build_index():
    print("Loading recipes...")
    recipes = load_recipes()
    print(f"Loaded {len(recipes)} recipes.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()
    embed_fn = _embedding_function()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embed_fn,
    )

    ids = []
    documents = []
    metadatas = []
    for idx, recipe in enumerate(recipes):
        ids.append(f"recipe-{idx}")
        documents.append(" ".join(recipe["ingredients"]))
        metadatas.append(
            {
                "title": recipe["title"],
                "ingredients": recipe["ingredients"],
                "instructions": recipe["instructions"],
            }
        )

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Saved {len(ids)} recipes to Chroma at {CHROMA_DIR}")


def _get_collection():
    client = _get_client()
    embed_fn = _embedding_function()

    # This raises if the collection does not exist, which guides the user to run data prep.
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def retrieve(user_ingredients: List[str], top_k: int = 5):
    """
    user_ingredients: list of strings, e.g. ["chicken", "rice", "garlic"]
    returns: list of recipe dicts with a 'similarity' field
    """
    collection = _get_collection()
    query_text = " ".join([ing.lower().strip() for ing in user_ingredients])

    results = collection.query(query_texts=[query_text], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    recipes = []
    for doc, meta, dist in zip(docs, metas, distances):
        similarity = 1 - dist if dist is not None else None
        recipes.append(
            {
                "title": meta.get("title"),
                "ingredients": meta.get("ingredients", []),
                "instructions": meta.get("instructions", ""),
                "similarity": similarity,
                "match_ingredients": doc,
            }
        )
    return recipes


if __name__ == "__main__":
    # Build the index once
    build_index()

    # Example: uncomment to test
    print(retrieve(["chicken", "rice", "garlic"], top_k=3))
