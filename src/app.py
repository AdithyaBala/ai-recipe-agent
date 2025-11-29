# src/app.py

import gradio as gr

from .retrieval import retrieve
from .generator import generate_recipe   # <-- now local HF version


def run_pipeline(
    ingredient_text: str,
    top_k: int = 3,
    use_retrieval: bool = True,
):
    ingredients = [
        part.strip()
        for part in ingredient_text.split(",")
        if part.strip()
    ]
    if not ingredients:
        return "Please enter at least one ingredient, separated by commas."

    retrieved_recipe = None
    baseline_preview = ""

    if use_retrieval:
        retrieved = retrieve(ingredients, top_k=top_k)
        if retrieved:
            retrieved_recipe = retrieved[0]
            baseline_preview = (
                f"**Baseline retrieved recipe:**\n\n"
                f"**{retrieved_recipe['title']}**\n\n"
                f"**Ingredients (dataset):**\n"
                + "\n".join(f"- {ing}" for ing in retrieved_recipe["ingredients"])
                + "\n\n"
            )
        else:
            baseline_preview = "_No similar recipes found in dataset._\n\n"

    generated = generate_recipe(ingredients, retrieved_recipe)

    result_md = baseline_preview + "\n---\n\n" + "**AI-generated recipe:**\n\n" + generated
    return result_md


def main():
    iface = gr.Interface(
        fn=run_pipeline,
        inputs=[
            gr.Textbox(
                lines=2,
                label="Ingredients (comma-separated)",
                placeholder="e.g. chicken, rice, onion, garlic",
            ),
            gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=3,
                label="Top-K retrieved recipes",
            ),
            gr.Checkbox(
                value=True,
                label="Use retrieved recipe as context",
            ),
        ],
        outputs=gr.Markdown(label="Output"),
        title="Ingredient-based Recipe Generator (Local Model)",
        description="Runs completely locally using an open-source model.",
    )
    iface.launch()


if __name__ == "__main__":
    main()
