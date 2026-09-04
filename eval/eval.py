# eval/eval.py
"""
Retrieval + planner + critic eval harness.
Measures reliability and latency across eval/test_queries.json.

Usage:
    python -m eval.eval
    python -m eval.eval --test-file path/to/other.json --top-k 5 --output eval/my_results.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.agent import _critic, _plan
from src.retrieval import retrieve

# ---------------------------------------------------------------------------
# Forbidden-ingredient map — used for constraint satisfaction scoring.
# Each tag maps to a list of substrings; a match is case-insensitive contains.
# ---------------------------------------------------------------------------
_FORBIDDEN: dict[str, list[str]] = {
    "vegan": [
        "chicken", "beef", "pork", "lamb", "turkey", "bacon", "sausage",
        "fish", "salmon", "tuna", "shrimp", "prawn", "lobster", "crab",
        "cow milk", "dairy milk", "whole milk", "skim milk", "heavy cream",
        "sour cream", "cream cheese", "cheese", "butter", "yogurt", "ghee",
        "egg", "honey",
    ],
    "vegetarian": [
        "chicken", "beef", "pork", "lamb", "turkey", "bacon", "sausage",
        "fish", "salmon", "tuna", "shrimp", "prawn", "lobster", "crab",
    ],
    "dairy-free": ["milk", "cheese", "butter", "cream", "yogurt", "ghee", "whey"],
    "nut-free": [
        "peanut", "almond", "cashew", "walnut", "pecan", "hazelnut",
        "pistachio", "macadamia", "chestnut",
    ],
    "gluten-free": ["flour", " bread", "pasta", "wheat", "barley", "rye", "semolina"],
    "shellfish-free": ["shrimp", "prawn", "lobster", "crab", "scallop", "oyster", "mussel", "clam"],
    "keto": ["potato", " rice", " bread", "pasta", "sugar", "flour", "corn", "beans"],
}


def _check_constraints(recipe_ingredients: list[str], constraints: dict) -> tuple[bool, list[str]]:
    violations: list[str] = []
    dietary = constraints.get("dietary", [])
    for tag in dietary:
        forbidden = _FORBIDDEN.get(tag)
        if not forbidden:
            continue
        for ing in recipe_ingredients:
            ing_lower = ing.lower()
            for token in forbidden:
                if token in ing_lower:
                    violations.append(f"{tag}:{ing}")
                    break
    return len(violations) == 0, violations


def _run_one(query: dict, top_k: int) -> dict[str, Any]:
    ingredients = query.get("ingredients", [])
    constraints = query.get("constraints", {})

    start = time.perf_counter()
    try:
        if not ingredients:
            raise ValueError("empty ingredient list")

        planned = _plan(ingredients)
        candidates = retrieve(planned, top_k=top_k)

        if not candidates:
            return {
                "success": False,
                "constraint_pass": False,
                "violations": [],
                "error": "no candidates returned",
                "top1_title": None,
                "latency_seconds": round(time.perf_counter() - start, 3),
            }

        top1 = candidates[0]
        recipe_copy = dict(top1)
        _critic(recipe_copy, planned)

        constraint_pass, violations = _check_constraints(
            recipe_copy.get("ingredients", []), constraints
        )

        return {
            "success": True,
            "constraint_pass": constraint_pass,
            "violations": violations,
            "error": None,
            "top1_title": top1.get("title"),
            "latency_seconds": round(time.perf_counter() - start, 3),
        }

    except Exception as exc:
        return {
            "success": False,
            "constraint_pass": False,
            "violations": [],
            "error": str(exc),
            "top1_title": None,
            "latency_seconds": round(time.perf_counter() - start, 3),
        }


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _rollup(rows: list[dict]) -> dict:
    latencies = sorted(r["latency_seconds"] for r in rows)
    n = len(rows)
    return {
        "n": n,
        "success_rate": round(sum(r["success"] for r in rows) / n, 4),
        "constraint_pass_rate": round(sum(r["constraint_pass"] for r in rows) / n, 4),
        "latency_avg_s": round(statistics.mean(latencies), 3),
        "latency_median_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(_percentile(latencies, 95), 3),
    }


def run_eval(test_file: str = "eval/test_queries.json", top_k: int = 3, output: str = "eval/eval_results.json") -> dict:
    queries = json.loads(Path(test_file).read_text())

    results = []
    for q in queries:
        row = _run_one(q, top_k)
        row["id"] = q["id"]
        row["category"] = q["category"]
        results.append(row)
        status = "PASS" if row["success"] and row["constraint_pass"] else "FAIL"
        print(f"  [{status}] {q['id']:45s}  {row['latency_seconds']:.3f}s  {row.get('error') or ''}")

    summary = _rollup(results)

    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)
    per_category = {cat: _rollup(rows) for cat, rows in categories.items()}

    output_data = {"summary": summary, "per_category": per_category, "results": results}
    Path(output).write_text(json.dumps(output_data, indent=2))

    print(f"\n{'='*60}")
    print(f"Overall  n={summary['n']}  success={summary['success_rate']:.1%}  "
          f"constraint_pass={summary['constraint_pass_rate']:.1%}  "
          f"latency avg={summary['latency_avg_s']}s  median={summary['latency_median_s']}s  "
          f"p95={summary['latency_p95_s']}s")
    print(f"\nPer-category breakdown:")
    col = max(len(c) for c in per_category) + 2
    print(f"  {'Category':<{col}} {'n':>4}  {'success':>8}  {'constrs':>8}  {'avg_s':>7}  {'p95_s':>7}")
    print(f"  {'-'*col} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")
    for cat, stats in sorted(per_category.items()):
        print(f"  {cat:<{col}} {stats['n']:>4}  {stats['success_rate']:>8.1%}  "
              f"{stats['constraint_pass_rate']:>8.1%}  {stats['latency_avg_s']:>7.3f}  "
              f"{stats['latency_p95_s']:>7.3f}")

    print(f"\nFull results → {output}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recipe agent eval harness (retrieval + critic mode)")
    parser.add_argument("--test-file", default="eval/test_queries.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="eval/eval_results.json")
    args = parser.parse_args()
    run_eval(args.test_file, args.top_k, args.output)
