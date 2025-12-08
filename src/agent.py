# src/agent.py
# Agentic planner + critic for recipe generation

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .generator import (
    GenerationConfig,
    generate_recipe,
    normalize_ingredients_list,
    render_recipe_markdown,
)
from .retrieval import retrieve

PANTRY_STAPLES = ["salt", "pepper", "oil", "water"]


@dataclass
class AgentResult:
    status_log: List[str]
    recipe_markdown: str
    raw_recipe: Dict
    retrieved_preview: str


def _plan(user_ingredients: List[str]) -> List[str]:
    planned = normalize_ingredients_list(user_ingredients)
    if len(planned) < 3:
        for staple in PANTRY_STAPLES:
            if staple not in planned:
                planned.append(staple)
    return planned


def _critic(recipe: Dict, allowed: List[str]) -> Tuple[Dict, List[str]]:
    allowed_set = {ing.strip().lower() for ing in allowed}
    filtered = []
    removed = []
    for ing in recipe.get("ingredients", []):
        norm = ing.strip().lower()
        if norm in allowed_set:
            filtered.append(ing)
        else:
            removed.append(ing)

    if filtered:
        recipe["ingredients"] = filtered
    else:
        recipe["ingredients"] = allowed

    return recipe, removed


def _format_status(log: List[str]) -> str:
    return "\n".join(f"- {line}" for line in log)


def _format_retrieved(retrieved: Optional[Dict]) -> str:
    if not retrieved:
        return "_No similar recipes found in dataset._"
    ing_list = "\n".join(f"- {ing}" for ing in retrieved.get("ingredients", []))
    return (
        f"**{retrieved.get('title','(no title)')}**\n\n"
        f"**Ingredients (dataset):**\n{ing_list}"
    )


def run_agent(
    ingredient_text: str,
    top_k: int = 3,
    use_retrieval: bool = True,
    model_provider: Optional[str] = None,
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
) -> AgentResult:
    status = []
    status.append("Agent analyzing ingredients...")

    user_ingredients = [
        part.strip() for part in ingredient_text.split(",") if part.strip()
    ]
    if not user_ingredients:
        raise ValueError("Please enter at least one ingredient, separated by commas.")

    planned_ingredients = _plan(user_ingredients)
    status.append(
        f"Planner finalized ingredients: {', '.join(planned_ingredients)}"
    )

    retrieved_recipe = None
    retrieved_preview = "_Retrieval skipped_"
    if use_retrieval:
        status.append("Retrieving similar recipes from memory...")
        candidates = retrieve(planned_ingredients, top_k=top_k)
        retrieved_recipe = candidates[0] if candidates else None
        retrieved_preview = _format_retrieved(retrieved_recipe)

    status.append("Generating recipe with selected model...")
    defaults = GenerationConfig()
    config = GenerationConfig(
        provider=model_provider or defaults.provider,
        model_id=model_id or defaults.model_id,
        model_path=model_path,
    )
    recipe = generate_recipe(
        planned_ingredients,
        retrieved_recipe=retrieved_recipe,
        config=config,
    )

    status.append("Critic verifying ingredients against user input + staples...")
    recipe, removed = _critic(recipe, planned_ingredients)
    if removed:
        status.append(
            f"Critic removed hallucinated ingredients: {', '.join(removed)}"
        )
    else:
        status.append("Critic check passed: no extra ingredients.")

    recipe_md = render_recipe_markdown(recipe)

    return AgentResult(
        status_log=status,
        recipe_markdown=recipe_md,
        raw_recipe=recipe,
        retrieved_preview=retrieved_preview,
    )
