# src/generator.py  -- local model with heuristic fallback for better recipes

from typing import Optional, List, Dict
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# If this is too heavy, change to "google/flan-t5-small"
MODEL_NAME = "google/flan-t5-base"

print(f"Loading local text2text-generation model: {MODEL_NAME} (first time may take a while)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

text2text = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
)


# ---------- helpers for structure ----------

def _clean_ingredients(user_ingredients: List[str]) -> List[str]:
    """Normalize user ingredients and add a couple of common pantry items."""
    cleaned = []
    for ing in user_ingredients:
        ing = ing.strip().lower()
        if ing:
            cleaned.append(ing)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for ing in cleaned:
        if ing not in seen:
            seen.add(ing)
            unique.append(ing)

    # Add some common extras if not already present
    for extra in ["salt", "black pepper", "olive oil"]:
        if extra not in seen:
            unique.append(extra)

    return unique


def _make_title(ingredients: List[str]) -> str:
    """Simple heuristic title based on first 1–2 ingredients."""
    if not ingredients:
        return "Simple Home-Cooked Dish"

    main = ingredients[0].title()
    if len(ingredients) > 1:
        second = ingredients[1].title()
        return f"{main} and {second} Recipe"
    return f"{main} Recipe"


# ---------- heuristic fallback steps ----------

COMMON_VERBS = [
    "chop", "dice", "slice", "mince",
    "heat", "cook", "fry", "bake", "boil", "simmer", "saute", "stir",
    "add", "mix", "combine", "season", "serve",
]


def _fallback_steps(ingredients: List[str]) -> str:
    """Generic but sensible cooking steps using the ingredients."""
    main = ingredients[0] if ingredients else "main ingredient"
    others = [ing for ing in ingredients[1:] if ing not in ["salt", "black pepper", "olive oil"]]

    aromatics = [ing for ing in others if any(x in ing for x in ["onion", "garlic", "ginger"])]
    grains = [ing for ing in others if any(x in ing for x in ["rice", "pasta", "noodle", "quinoa"])]
    veggies = [ing for ing in others if ing not in aromatics + grains]

    steps = []

    # 1. Prep
    prep_items = aromatics + veggies
    if prep_items:
        steps.append(
            f"1. Wash and prep the ingredients. Finely chop the {', '.join(prep_items)} and cut the {main} into bite-sized pieces."
        )
    else:
        steps.append(
            f"1. Wash and prep the ingredients. Cut the {main} into bite-sized pieces and gather the remaining ingredients."
        )

    # 2. Heat oil & aromatics
    if aromatics:
        steps.append(
            "2. Heat a little olive oil in a pan over medium heat. Add the chopped aromatics and cook for a few minutes until fragrant and softened."
        )
    else:
        steps.append(
            "2. Heat a little olive oil in a pan over medium heat."
        )

    # 3. Cook the main
    steps.append(
        f"3. Add the {main} to the pan. Season with salt and black pepper and cook, stirring occasionally, until mostly cooked through."
    )

    # 4. Add grains/veggies
    if grains:
        steps.append(
            f"4. Add the {', '.join(grains)} and enough water or broth to cook them. Bring to a simmer, cover, and cook until tender, stirring occasionally."
        )
    if veggies:
        steps.append(
            f"5. Stir in the remaining vegetables ({', '.join(veggies)}). Cook for a few more minutes until tender but still bright."
        )

    # Final step
    final_step_num = len(steps) + 1 if veggies or grains else len(steps) + 1
    steps.append(
        f"{final_step_num}. Taste and adjust the seasoning with more salt and black pepper if needed. Serve warm."
    )

    # Re-number steps correctly
    renumbered = []
    for i, s in enumerate(steps, start=1):
        # remove existing "N." at start and re-add
        s = re.sub(r"^\d+\.\s*", "", s)
        renumbered.append(f"{i}. {s}")

    return "\n".join(renumbered)


# ---------- model-based steps with quality check ----------

def _generate_steps_with_model(
    ingredients: List[str],
    retrieved_recipe: Optional[Dict],
) -> str:
    """Ask the model for steps, then check if they look reasonable."""
    ingredients_str = ", ".join(ingredients)

    prompt = (
        "You are an expert home cook. Write clear, numbered cooking steps for a home recipe "
        f"using these ingredients: {ingredients_str}. "
        "Assume the cook has basic kitchen equipment and knows simple techniques.\n\n"
    )

    if retrieved_recipe is not None:
        prompt += (
            "Here is a similar existing recipe from a dataset for inspiration:\n"
            f"Title: {retrieved_recipe['title']}\n"
            f"Ingredients: {', '.join(retrieved_recipe['ingredients'])}\n"
            f"Instructions: {retrieved_recipe['instructions'][:300]}...\n\n"
        )

    prompt += (
        "Now write the steps for a NEW recipe. Output only the steps as a numbered list like:\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n"
        "Make sure there are at least 4 steps and each step describes an action, "
        "not just a list of ingredients."
    )

    raw = text2text(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        num_return_sequences=1,
    )[0]["generated_text"].strip()

    # Parse numbered lines
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    numbered = [ln for ln in lines if re.match(r"^\d+\.", ln)]
    text = "\n".join(numbered) if numbered else raw

    # Basic quality checks
    long_enough = len(text) > 80
    has_multiple_steps = len(numbered) >= 3
    has_verbs = any(v in text.lower() for v in COMMON_VERBS)

    if not (long_enough and has_multiple_steps and has_verbs):
        # Fail -> we'll let caller fall back to heuristic
        return ""

    return text


# ---------- main entry point ----------

def generate_recipe(
    user_ingredients: List[str],
    retrieved_recipe: Optional[Dict] = None,
    total_time_minutes: int = 30,
) -> str:
    """
    Build a full recipe in markdown.

    - Title: heuristic from ingredients
    - Ingredients: cleaned user ingredients + basic pantry items
    - Steps: try model; if bad, fall back to heuristic steps
    """
    ing_list = _clean_ingredients(user_ingredients)
    title = _make_title(ing_list)

    model_steps = _generate_steps_with_model(ing_list, retrieved_recipe)
    if model_steps:
        steps_text = model_steps
    else:
        steps_text = _fallback_steps(ing_list)

    ingredients_block = "\n".join(f"- {ing}" for ing in ing_list)

    recipe_md = (
        f"Title: {title}\n\n"
        f"Ingredients:\n{ingredients_block}\n\n"
        f"Steps:\n{steps_text}\n\n"
        f"Total time: {total_time_minutes} minutes\n"
    )

    return recipe_md


if __name__ == "__main__":
    example_ings = ["chicken", "rice", "onion", "garlic"]
    print(generate_recipe(example_ings))
