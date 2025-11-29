# src/evaluate.py
"""
Utility functions to evaluate generated recipes.

Includes:
- ingredient_coverage: how many user ingredients were used
- extra_ingredients: how many ingredients appeared that the user didn't list
- parse_ingredients_from_markdown: extract ingredients from a markdown-style recipe
- evaluate_recipe: convenience wrapper to get coverage & extra counts from raw text
"""

from typing import List, Set, Dict


def ingredient_coverage(
    user_ingredients: List[str],
    recipe_ingredients: List[str],
) -> float:
    """
    Compute the fraction of user ingredients that appear in the recipe.

    Args:
        user_ingredients: list of ingredients the user actually has
        recipe_ingredients: list of ingredients extracted from the recipe

    Returns:
        coverage in [0, 1]
    """
    user_set: Set[str] = {x.lower().strip() for x in user_ingredients if x.strip()}
    recipe_set: Set[str] = {x.lower().strip() for x in recipe_ingredients if x.strip()}

    if not user_set:
        return 0.0

    used = len(user_set & recipe_set)
    return used / len(user_set)


def extra_ingredients(
    user_ingredients: List[str],
    recipe_ingredients: List[str],
) -> int:
    """
    Count how many ingredients appear in the recipe that the user did NOT list.

    Args:
        user_ingredients: list of ingredients the user actually has
        recipe_ingredients: list of ingredients extracted from the recipe

    Returns:
        number of extra ingredients
    """
    user_set: Set[str] = {x.lower().strip() for x in user_ingredients if x.strip()}
    recipe_set: Set[str] = {x.lower().strip() for x in recipe_ingredients if x.strip()}
    return len(recipe_set - user_set)


def parse_ingredients_from_markdown(recipe_text: str) -> List[str]:
    """
    Very simple parser that assumes ingredients are listed as bullet points
    after a line containing 'Ingredients'.

    Works with outputs like:

        Title: Garlic Chicken Rice

        Ingredients:
        - chicken breast
        - rice
        - garlic
        - olive oil

        Steps:
        1. Do X
        2. Do Y

    Args:
        recipe_text: full generated recipe as a string

    Returns:
        list of ingredient strings (without the leading '-')
    """
    lines = recipe_text.splitlines()
    ingredients: List[str] = []
    in_block = False

    for line in lines:
        stripped = line.strip()

        # Start of ingredients section
        if "ingredients" in stripped.lower():
            in_block = True
            continue

        if in_block:
            # Bullet point line
            if stripped.startswith("-"):
                ing = stripped.lstrip("-").strip()
                if ing:
                    ingredients.append(ing)
            # Empty line or start of another section -> end of block
            elif stripped == "" or stripped.lower().startswith("steps"):
                if ingredients:
                    break

    return ingredients


def evaluate_recipe(
    user_ingredients: List[str],
    recipe_text: str,
) -> Dict[str, float]:
    """
    Convenience helper to evaluate a generated recipe from raw text.

    Args:
        user_ingredients: list of ingredients the user has
        recipe_text: raw generated recipe text (markdown-like)

    Returns:
        dict with:
            - coverage: fraction of user ingredients used
            - extra: number of extra ingredients added
            - n_user: number of user ingredients
            - n_recipe: number of ingredients parsed from recipe
    """
    recipe_ings = parse_ingredients_from_markdown(recipe_text)
    cov = ingredient_coverage(user_ingredients, recipe_ings)
    extra = extra_ingredients(user_ingredients, recipe_ings)

    return {
        "coverage": cov,
        "extra": extra,
        "n_user": len([x for x in user_ingredients if x.strip()]),
        "n_recipe": len(recipe_ings),
    }


if __name__ == "__main__":
    # Tiny sanity test
    user = ["chicken", "rice", "garlic"]

    sample_recipe = """
    Title: Garlic Chicken Rice

    Ingredients:
    - chicken breast
    - rice
    - garlic
    - olive oil
    - salt

    Steps:
    1. Cook everything in a pan.
    2. Serve hot.
    Total time: 30 minutes
    """

    parsed = parse_ingredients_from_markdown(sample_recipe)
    print("Parsed ingredients:", parsed)

    from pprint import pprint
    pprint(evaluate_recipe(user, sample_recipe))
