# src/retrieval.py

from pathlib import Path
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/recipes_clean.json")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_recipes():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python -m src.data_prep` first."
        )
    recipes = json.loads(DATA_PATH.read_text())
    return recipes


def build_index():
    print("Loading recipes...")
    recipes = load_recipes()
    print(f"Loaded {len(recipes)} recipes.")

    docs = [" ".join(r["ingredients"]) for r in recipes]
    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(X, MODEL_DIR / "tfidf_matrix.joblib")
    joblib.dump(recipes, MODEL_DIR / "recipes.joblib")

    print("Saved TF-IDF index and recipes in models/")


def load_index():
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    X = joblib.load(MODEL_DIR / "tfidf_matrix.joblib")
    recipes = joblib.load(MODEL_DIR / "recipes.joblib")
    return vectorizer, X, recipes


def retrieve(user_ingredients, top_k: int = 5):
    """
    user_ingredients: list of strings, e.g. ["chicken", "rice", "garlic"]
    returns: list of recipe dicts with a 'similarity' field
    """
    vectorizer, X, recipes = load_index()
    query_text = " ".join([ing.lower().strip() for ing in user_ingredients])
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, X)[0]

    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        r = dict(recipes[idx])
        r["similarity"] = float(sims[idx])
        results.append(r)
    return results


if __name__ == "__main__":
    # Build the index once
    build_index()

    # Example: uncomment to test
    print(retrieve(["chicken", "rice", "garlic"], top_k=3))
