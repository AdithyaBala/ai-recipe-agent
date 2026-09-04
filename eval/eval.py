# eval/eval.py
"""
Retrieval + planner + critic eval harness.
Measures reliability, constraint satisfaction, ingredient coverage, and latency.

Modes:
    rag     — retrieve + dietary filter + critic  (default)
    no-rag  — planner + critic only, no retrieval
    ab      — run both and print a side-by-side comparison

Usage:
    python -m eval.eval
    python -m eval.eval --mode ab
    python -m eval.eval --mode no-rag --output eval/baseline_results.json
    python -m eval.eval --test-file path/to/other.json --top-k 5
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.agent import _critic, _filter_by_dietary, _plan
from src.evaluate import ingredient_coverage
from src.retrieval import retrieve

# ---------------------------------------------------------------------------
# Constraint violation checker (post-hoc scoring, independent of agent filter)
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
    "shellfish-free": [
        "shrimp", "prawn", "lobster", "crab", "scallop", "oyster", "mussel", "clam",
    ],
    "keto": ["potato", " rice", " bread", "pasta", "sugar", "flour", "corn", "beans"],
}


def _check_constraints(recipe_ingredients: list[str], constraints: dict) -> tuple[bool, list[str]]:
    violations: list[str] = []
    for tag in constraints.get("dietary", []):
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


# ---------------------------------------------------------------------------
# Per-query runner
# ---------------------------------------------------------------------------

def _run_one(query: dict, top_k: int, use_rag: bool) -> dict[str, Any]:
    ingredients = query.get("ingredients", [])
    constraints = query.get("constraints", {})

    start = time.perf_counter()
    try:
        if not ingredients:
            raise ValueError("empty ingredient list")

        planned = _plan(ingredients)

        if use_rag:
            candidates = retrieve(planned, top_k=top_k)
            dietary = constraints.get("dietary", [])
            if dietary:
                candidates = _filter_by_dietary(candidates, dietary)

            if not candidates:
                raise RuntimeError("no candidates returned")

            top1 = candidates[0]
            recipe_ings = list(top1.get("ingredients", []))
            top1_title = top1.get("title")
            # Coverage: fraction of user's requested ingredients found in retrieved recipe
            coverage = round(ingredient_coverage(ingredients, recipe_ings), 4)
        else:
            # No-RAG baseline: the "recipe" is simply the planned ingredients
            recipe_ings = list(planned)
            top1_title = None
            coverage = 1.0  # planned ings are derived directly from user ings

        recipe_copy = {"ingredients": recipe_ings}
        _critic(recipe_copy, planned)

        constraint_pass, violations = _check_constraints(
            recipe_copy.get("ingredients", []), constraints
        )

        return {
            "success": True,
            "constraint_pass": constraint_pass,
            "violations": violations,
            "coverage": coverage,
            "error": None,
            "top1_title": top1_title,
            "latency_seconds": round(time.perf_counter() - start, 3),
        }

    except Exception as exc:
        return {
            "success": False,
            "constraint_pass": False,
            "violations": [],
            "coverage": 0.0,
            "error": str(exc),
            "top1_title": None,
            "latency_seconds": round(time.perf_counter() - start, 3),
        }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

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
        "avg_coverage": round(sum(r["coverage"] for r in rows) / n, 4),
        "latency_avg_s": round(statistics.mean(latencies), 3),
        "latency_median_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(_percentile(latencies, 95), 3),
    }


# ---------------------------------------------------------------------------
# Single-mode eval
# ---------------------------------------------------------------------------

def run_eval(
    test_file: str = "eval/test_queries.json",
    top_k: int = 3,
    output: str = "eval/eval_results.json",
    use_rag: bool = True,
    verbose: bool = True,
) -> dict:
    queries = json.loads(Path(test_file).read_text())
    label = "RAG" if use_rag else "no-RAG"

    results = []
    for q in queries:
        row = _run_one(q, top_k, use_rag)
        row["id"] = q["id"]
        row["category"] = q["category"]
        results.append(row)
        if verbose:
            status = "PASS" if row["success"] and row["constraint_pass"] else "FAIL"
            print(f"  [{status}] {q['id']:45s}  {row['latency_seconds']:.3f}s  {row.get('error') or ''}")

    summary = _rollup(results)

    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)
    per_category = {cat: _rollup(rows) for cat, rows in categories.items()}

    output_data = {
        "mode": label,
        "summary": summary,
        "per_category": per_category,
        "results": results,
    }
    Path(output).write_text(json.dumps(output_data, indent=2))

    if verbose:
        _print_summary(summary, per_category, label, output)

    return output_data


def _print_summary(summary: dict, per_category: dict, label: str, output: str) -> None:
    print(f"\n{'='*65}")
    print(f"[{label}]  n={summary['n']}  "
          f"success={summary['success_rate']:.1%}  "
          f"constraint_pass={summary['constraint_pass_rate']:.1%}  "
          f"coverage={summary['avg_coverage']:.1%}  "
          f"latency median={summary['latency_median_s']}s  p95={summary['latency_p95_s']}s")
    print(f"\nPer-category ({label}):")
    col = max(len(c) for c in per_category) + 2
    print(f"  {'Category':<{col}} {'n':>4}  {'success':>8}  {'constrs':>8}  {'coverage':>9}  {'avg_s':>7}  {'p95_s':>7}")
    print(f"  {'-'*col} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}")
    for cat, stats in sorted(per_category.items()):
        print(f"  {cat:<{col}} {stats['n']:>4}  {stats['success_rate']:>8.1%}  "
              f"{stats['constraint_pass_rate']:>8.1%}  {stats['avg_coverage']:>9.1%}  "
              f"{stats['latency_avg_s']:>7.3f}  {stats['latency_p95_s']:>7.3f}")
    print(f"\nFull results → {output}")


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------

def run_ab_eval(
    test_file: str = "eval/test_queries.json",
    top_k: int = 3,
    output: str = "eval/ab_results.json",
) -> dict:
    queries = json.loads(Path(test_file).read_text())

    print("Running RAG condition...")
    rag_results, norag_results = [], []
    for q in queries:
        rag_row = _run_one(q, top_k, use_rag=True)
        rag_row["id"] = q["id"]
        rag_row["category"] = q["category"]
        rag_results.append(rag_row)

    print("Running no-RAG baseline...")
    for q in queries:
        norag_row = _run_one(q, top_k, use_rag=False)
        norag_row["id"] = q["id"]
        norag_row["category"] = q["category"]
        norag_results.append(norag_row)

    rag_summary = _rollup(rag_results)
    norag_summary = _rollup(norag_results)

    # Per-query delta table
    deltas = []
    for r, b in zip(rag_results, norag_results):
        deltas.append({
            "id": r["id"],
            "category": r["category"],
            "rag_constraint_pass": r["constraint_pass"],
            "norag_constraint_pass": b["constraint_pass"],
            "rag_coverage": r["coverage"],
            "norag_coverage": b["coverage"],
            "rag_latency_s": r["latency_seconds"],
            "norag_latency_s": b["latency_seconds"],
            "rag_top1_title": r["top1_title"],
            "rag_violations": r["violations"],
            "norag_violations": b["violations"],
        })

    output_data = {
        "rag": {"summary": rag_summary, "results": rag_results},
        "no_rag": {"summary": norag_summary, "results": norag_results},
        "per_query_delta": deltas,
    }
    Path(output).write_text(json.dumps(output_data, indent=2))

    _print_ab_comparison(rag_summary, norag_summary, deltas)
    print(f"\nFull A/B results → {output}")
    return output_data


def _print_ab_comparison(rag: dict, norag: dict, deltas: list[dict]) -> None:
    print(f"\n{'='*65}")
    print(f"{'Metric':<30} {'RAG+filter':>12} {'No-RAG':>10} {'Delta':>8}")
    print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*8}")

    metrics = [
        ("Success rate",         "success_rate",        True),
        ("Constraint pass rate", "constraint_pass_rate", True),
        ("Avg ingredient coverage", "avg_coverage",     True),
        ("Latency median (s)",   "latency_median_s",    False),
        ("Latency p95 (s)",      "latency_p95_s",       False),
    ]
    for label, key, higher_better in metrics:
        r_val = rag[key]
        b_val = norag[key]
        delta = r_val - b_val
        is_pct = isinstance(r_val, float) and key not in ("latency_median_s", "latency_p95_s", "latency_avg_s")
        fmt = "{:.1%}" if is_pct else "{:.3f}"
        delta_sign = "+" if delta > 0 else ""
        delta_str = f"{delta_sign}{fmt.format(delta)}" if is_pct else f"{delta_sign}{delta:.3f}"
        print(f"  {label:<28} {fmt.format(r_val):>12} {fmt.format(b_val):>10} {delta_str:>8}")

    # Highlight queries where RAG changed the outcome
    rag_wins = [d for d in deltas if d["rag_constraint_pass"] and not d["norag_constraint_pass"]]
    norag_wins = [d for d in deltas if d["norag_constraint_pass"] and not d["rag_constraint_pass"]]

    if rag_wins:
        print(f"\nQueries where RAG+filter flipped FAIL → PASS ({len(rag_wins)}):")
        for d in rag_wins:
            print(f"  {d['id']}  →  retrieved: {d['rag_top1_title']}")

    if norag_wins:
        print(f"\nQueries where no-RAG outperformed RAG ({len(norag_wins)}):")
        for d in norag_wins:
            print(f"  {d['id']}  violations with RAG: {d['rag_violations']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Recipe agent eval harness")
    parser.add_argument("--mode", choices=["rag", "no-rag", "ab"], default="rag",
                        help="rag: retrieval+filter (default) | no-rag: baseline | ab: side-by-side")
    parser.add_argument("--test-file", default="eval/test_queries.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default depends on mode)")
    parser.add_argument("--min-success-rate", type=float, default=None,
                        help="Fail (exit 1) if success_rate drops below this threshold (0-1)")
    parser.add_argument("--min-constraint-pass-rate", type=float, default=None,
                        help="Fail (exit 1) if constraint_pass_rate drops below this threshold (0-1)")
    args = parser.parse_args()

    if args.mode == "ab":
        output = args.output or "eval/ab_results.json"
        data = run_ab_eval(args.test_file, args.top_k, output)
        summary = data["rag"]["summary"]
    elif args.mode == "no-rag":
        output = args.output or "eval/baseline_results.json"
        data = run_eval(args.test_file, args.top_k, output, use_rag=False)
        summary = data["summary"]
    else:
        output = args.output or "eval/eval_results.json"
        data = run_eval(args.test_file, args.top_k, output, use_rag=True)
        summary = data["summary"]

    failed = False
    if args.min_success_rate is not None and summary["success_rate"] < args.min_success_rate:
        print(f"\nFAIL: success_rate {summary['success_rate']:.1%} < threshold {args.min_success_rate:.1%}")
        failed = True
    if args.min_constraint_pass_rate is not None and summary["constraint_pass_rate"] < args.min_constraint_pass_rate:
        print(f"\nFAIL: constraint_pass_rate {summary['constraint_pass_rate']:.1%} < threshold {args.min_constraint_pass_rate:.1%}")
        failed = True
    if failed:
        sys.exit(1)
