# src/eval.py
import json
import time
from src.pipeline import run_pipeline  # replace with your actual entrypoint

def check_constraints(recipe, constraints):
    """Returns (passed: bool, violations: list[str])"""
    violations = []
    if "max_time_minutes" in constraints:
        if recipe.get("time_minutes", 0) > constraints["max_time_minutes"]:
            violations.append("time_exceeded")
    if "dietary" in constraints:
        for tag in constraints["dietary"]:
            if tag not in recipe.get("dietary_tags", []):
                violations.append(f"missing_dietary_tag:{tag}")
    return len(violations) == 0, violations

def run_eval(test_file="eval/test_queries.json"):
    with open(test_file) as f:
        queries = json.load(f)

    results = []
    for q in queries:
        start = time.time()
        # swap in your actual pipeline call + agent trace
        output = run_pipeline(ingredients=q["ingredients"], constraints=q["constraints"])
        latency = time.time() - start

        passed, violations = check_constraints(output["recipe"], q["constraints"])
        results.append({
            "id": q["id"],
            "passed": passed,
            "violations": violations,
            "latency_seconds": round(latency, 2),
            "critic_caught_issue": output.get("critic_flagged", None),
        })

    reliability = sum(r["passed"] for r in results) / len(results)
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)

    summary = {
        "reliability": round(reliability, 4),
        "avg_latency_seconds": round(avg_latency, 2),
        "num_queries": len(results),
        "results": results,
    }

    with open("eval/eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Reliability: {reliability:.1%} | Avg latency: {avg_latency:.2f}s")
    return summary

if __name__ == "__main__":
    run_eval()