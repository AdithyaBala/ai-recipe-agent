# src/generator.py
# Agent-aware recipe generation with pluggable model backends

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# Defaults can be overridden via env vars to swap providers/models without code changes.
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "hf")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "MBZUAI/LaMini-Flan-T5-783M")
HF_FALLBACK_MODEL = "google/flan-t5-small"

# Cache loaded generators so we only hit disk once.
_GEN_CACHE: Dict[Tuple[str, str, Optional[str]], Callable] = {}


@dataclass
class GenerationConfig:
    provider: str = DEFAULT_PROVIDER  # hf | llama-cpp | ollama | openai | groq
    model_id: str = DEFAULT_MODEL_ID
    model_path: Optional[str] = None  # for llama-cpp local gguf files
    temperature: float = 0.5
    top_p: float = 0.9
    max_new_tokens: int = 256


# ---------- basic text utilities ----------

def _normalize(text: str) -> str:
    return text.strip().lower()


def normalize_ingredients_list(ingredients: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for ing in ingredients:
        ing_norm = _normalize(ing)
        if ing_norm and ing_norm not in seen:
            seen.add(ing_norm)
            cleaned.append(ing_norm)
    return cleaned


def _make_title(ingredients: List[str]) -> str:
    if not ingredients:
        return "Simple Home-Cooked Dish"
    main = ingredients[0].title()
    if len(ingredients) > 1:
        return f"{main} and {ingredients[1].title()} Recipe"
    return f"{main} Recipe"


COMMON_VERBS = [
    "chop",
    "dice",
    "slice",
    "mince",
    "heat",
    "cook",
    "fry",
    "bake",
    "boil",
    "simmer",
    "saute",
    "stir",
    "add",
    "mix",
    "combine",
    "season",
    "serve",
]


def _fallback_steps(ingredients: List[str]) -> str:
    """Lightweight heuristic recipe steps when the model output is unusable."""
    main = ingredients[0] if ingredients else "main ingredient"
    others = [ing for ing in ingredients[1:] if ing not in ["salt", "pepper", "oil", "water"]]

    steps = [
        f"1. Prep the ingredients. Cut the {main} into bite-sized pieces and chop any vegetables you have ({', '.join(others) or 'if any'}).",
        "2. Heat a little oil in a pan over medium heat. Add aromatics like onion or garlic if available and cook until fragrant.",
        f"3. Add the {main} to the pan, season with salt and pepper, and cook until mostly done.",
    ]

    if others:
        steps.append(
            f"4. Stir in the remaining ingredients ({', '.join(others)}). Cook until tender and well combined."
        )

    steps.append("5. Taste and adjust seasoning. Serve warm.")
    return "\n".join(steps)


# ---------- model backends ----------

def _build_hf_generator(model_id: str) -> Callable:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        return pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
        )[0]["generated_text"]

    return _generate


def _build_llama_cpp_generator(model_path: str) -> Callable:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "Install llama-cpp-python to use provider 'llama-cpp' (pip install llama-cpp-python)."
        ) from exc

    llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)

    def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        output = llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["</s>", "###"],
        )
        return output["choices"][0]["text"]

    return _generate


def _build_ollama_generator(model_id: str) -> Callable:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Install ollama package to use provider 'ollama'.") from exc

    def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        res = ollama.chat(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "top_p": top_p, "num_predict": max_new_tokens},
        )
        return res["message"]["content"]

    return _generate


def _build_openai_generator(model_id: str, api_key: Optional[str], base_url: Optional[str]) -> Callable:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai to use provider 'openai'.") from exc

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)

    def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a concise home cooking assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content

    return _generate


def _build_groq_generator(model_id: str, api_key: Optional[str]) -> Callable:
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Install groq to use provider 'groq'.") from exc

    client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a concise home cooking assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content

    return _generate


def _get_generator(config: GenerationConfig) -> Callable:
    key = (config.provider, config.model_id, config.model_path)
    if key in _GEN_CACHE:
        return _GEN_CACHE[key]

    if config.provider == "hf":
        try:
            gen = _build_hf_generator(config.model_id)
        except Exception:
            # Lightweight fallback if preferred model fails.
            gen = _build_hf_generator(HF_FALLBACK_MODEL)
    elif config.provider == "llama-cpp":
        if not config.model_path:
            raise ValueError("model_path is required for provider 'llama-cpp'.")
        gen = _build_llama_cpp_generator(config.model_path)
    elif config.provider == "ollama":
        gen = _build_ollama_generator(config.model_id)
    elif config.provider == "openai":
        gen = _build_openai_generator(config.model_id, api_key=os.getenv("OPENAI_API_KEY"), base_url=None)
    elif config.provider == "groq":
        gen = _build_groq_generator(config.model_id, api_key=os.getenv("GROQ_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

    _GEN_CACHE[key] = gen
    return gen


# ---------- prompt + parsing ----------

def _build_prompt(ingredients: List[str], retrieved_recipe: Optional[Dict]) -> str:
    ing_str = ", ".join(ingredients)
    prompt = (
        "You are an expert home cook. Create a concise recipe using ONLY these ingredients: "
        f"{ing_str}.\n"
        "Do NOT invent new ingredients. It's fine to include pantry staples already listed.\n"
        "Return a JSON object with keys: title (string), ingredients (list of strings), steps (list of strings).\n"
        "Ensure steps are ordered actions for a home cook. Keep it brief but clear."
    )
    if retrieved_recipe:
        prompt += (
            "\nHere is a similar recipe for inspiration (do not copy ingredients outside the allowed list):\n"
            f"Title: {retrieved_recipe.get('title','')}\n"
            f"Ingredients: {', '.join(retrieved_recipe.get('ingredients', []))}\n"
            f"Instructions: {retrieved_recipe.get('instructions','')[:300]}...\n"
        )
    return prompt


def _extract_json_block(text: str) -> Optional[str]:
    try:
        return json.dumps(json.loads(text))
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            json.loads(snippet)
            return snippet
        except Exception:
            return None
    return None


def _parse_recipe_text(raw: str) -> Dict:
    json_block = _extract_json_block(raw)
    if not json_block:
        return {}
    try:
        data = json.loads(json_block)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _format_steps(steps: List[str]) -> str:
    formatted = []
    for idx, step in enumerate(steps, start=1):
        s = step.strip()
        if not s:
            continue
        s = re.sub(r"^\d+[\.\)]\s*", "", s)
        formatted.append(f"{idx}. {s}")
    return "\n".join(formatted)


# ---------- public API ----------

def generate_recipe(
    ingredients: List[str],
    retrieved_recipe: Optional[Dict] = None,
    total_time_minutes: int = 30,
    config: Optional[GenerationConfig] = None,
) -> Dict:
    """
    Return a structured recipe dict: {title, ingredients, steps, raw, total_time}
    Ingredients are user-provided (already planned by the agent).
    """
    config = config or GenerationConfig()
    ing_list = normalize_ingredients_list(ingredients)
    gen_fn = _get_generator(config)
    prompt = _build_prompt(ing_list, retrieved_recipe)

    try:
        raw = gen_fn(
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    except Exception as exc:
        raw = ""
        error_note = f"[Generation failed: {exc}]"
    else:
        error_note = ""

    parsed = _parse_recipe_text(raw)
    title = parsed.get("title") if isinstance(parsed, dict) else None
    parsed_ings = parsed.get("ingredients") if isinstance(parsed, dict) else None
    parsed_steps = parsed.get("steps") if isinstance(parsed, dict) else None

    recipe_ingredients = (
        [str(i).strip() for i in parsed_ings if str(i).strip()] if parsed_ings else ing_list
    )
    steps_text = (
        _format_steps([str(s) for s in parsed_steps]) if parsed_steps else ""
    )

    if not steps_text or not any(v in steps_text.lower() for v in COMMON_VERBS):
        steps_text = _fallback_steps(ing_list)

    recipe = {
        "title": title or _make_title(ing_list),
        "ingredients": recipe_ingredients,
        "steps": steps_text,
        "raw": raw or error_note,
        "total_time": total_time_minutes,
    }
    return recipe


def render_recipe_markdown(recipe: Dict) -> str:
    ingredients_block = "\n".join(f"- {ing}" for ing in recipe.get("ingredients", []))
    steps = recipe.get("steps", "")
    title = recipe.get("title", "Recipe")
    total_time = recipe.get("total_time", 30)
    return (
        f"**{title}**\n\n"
        f"Ingredients:\n{ingredients_block}\n\n"
        f"Steps:\n{steps}\n\n"
        f"Total time: {total_time} minutes\n"
    )


if __name__ == "__main__":
    example_ings = ["chicken", "rice", "onion", "garlic"]
    recipe = generate_recipe(example_ings)
    print(render_recipe_markdown(recipe))
