# src/app.py

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv


def _load_env():
    """
    Load environment variables from a root .env file and any *.env files inside a .env/ folder.
    This avoids needing `export` commands for keys like GROQ_API_KEY.
    """
    # Load default .env at repo root if present
    load_dotenv()

    env_dir = Path(".env")
    if env_dir.is_dir():
        for env_file in env_dir.glob("*.env"):
            load_dotenv(env_file)


_load_env()

from .agent import run_agent


PROVIDER = "groq"
MODEL_ID = "llama-3.3-70b-versatile"


def run_agent_groq(ingredients: str, k: int, use_retrieval: bool):
    try:
        result = run_agent(
            ingredient_text=ingredients,
            top_k=int(k),
            use_retrieval=use_retrieval,
            model_provider=PROVIDER,
            model_id=MODEL_ID,
        )
    except Exception as exc:
        return f"Error: {exc}", ""

    status_log = "\n".join(f"- {line}" for line in result.status_log)
    status_md = f"{status_log}\n\nRetrieval preview:\n{result.retrieved_preview}"
    return status_md, result.recipe_markdown


def main():
    with gr.Blocks(title="AI Recipe Agent") as demo:
        gr.Markdown("# 🍳 Agentic Recipe Generator (Groq Edition)")
        gr.Markdown(
            "Enter your ingredients. The Agent will plan, retrieve context, generate a recipe, "
            "and critique it to ensure no made-up ingredients. "
            f"Model: {MODEL_ID} via Groq."
        )

        with gr.Row():
            with gr.Column(scale=1):
                ingredients_input = gr.Textbox(
                    label="Ingredients (comma-separated)",
                    placeholder="e.g. chicken, rice, onion, garlic",
                    lines=2,
                )
                with gr.Row():
                    k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Retrieved Recipes (Context)",
                    )
                    retrieval_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Memory (RAG)",
                    )
                submit_btn = gr.Button("Generate Recipe", variant="primary")

            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Agent Status Log",
                    interactive=False,
                    lines=12,
                )
                recipe_output = gr.Markdown(label="Generated Recipe")

        submit_btn.click(
            fn=run_agent_groq,
            inputs=[ingredients_input, k_slider, retrieval_checkbox],
            outputs=[status_output, recipe_output],
        )

    demo.launch()


if __name__ == "__main__":
    main()
