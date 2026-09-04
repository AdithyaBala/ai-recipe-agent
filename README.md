# AI Recipe Agent

An agentic RAG pipeline that generates recipes from a list of ingredients. Given what you have on hand, the system retrieves semantically similar recipes from a 230k-recipe corpus, filters candidates against dietary constraints, generates a structured recipe with an LLM, and runs a critic pass to remove hallucinated ingredients.

## Architecture

```
User ingredients
      │
      ▼
  ┌─────────┐
  │ Planner │  normalizes + deduplicates, adds pantry staples if sparse
  └────┬────┘
       │
       ▼
  ┌───────────┐     ┌──────────────────────┐
  │ Retriever │────▶│  ChromaDB (cosine)   │  sentence-transformers/all-MiniLM-L6-v2
  └────┬──────┘     └──────────────────────┘
       │  top-k candidates
       ▼
  ┌────────────────┐
  │ Dietary Filter │  removes candidates that violate vegan/dairy-free/nut-free/etc.
  └────┬───────────┘
       │  best compliant candidate
       ▼
  ┌───────────┐
  │ Generator │  pluggable backend: Groq · OpenAI · HuggingFace · llama-cpp · Ollama
  └────┬──────┘
       │  raw recipe text
       ▼
  ┌────────┐
  │ Critic │  strips any ingredient not in the user's allowed list
  └────────┘
       │
       ▼
  Structured recipe  {title, ingredients, steps, total_time}
```

## Eval Results

Evaluated across **50 test scenarios** spanning 16 semantic categories (single constraint, multi-constraint, allergy safety, conflicting constraints, edge cases, stress test, and more).

| Metric | RAG + dietary filter | No-RAG baseline | Delta |
|---|---|---|---|
| Pipeline success rate | **98%** | 98% | — |
| Dietary constraint pass rate | **88%** | 82% | **+6pp** |
| Avg ingredient coverage | 20% | 98% | — |
| Retrieval latency (median) | **8ms** | — | — |
| Retrieval latency (p95) | **12ms** | — | — |

> Coverage is lower for RAG because retrieved recipes are full dataset recipes (10+ ingredients); the user typically provides 3–5. The constraint pass gain is what matters: RAG + dietary filtering resolves cases no-RAG cannot, including 2 allergy-safety-critical queries (peanut-free, shellfish-free).

A/B comparison between RAG and no-RAG baseline is reproducible with:
```bash
python -m eval.eval --mode ab
```

## Setup

### 1. Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, smaller download
pip install -r requirements.txt
```

### 2. Get the recipe dataset

1. Download `RAW_recipes.csv` from the [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset on Kaggle.
2. Place it at `data/RAW_recipes.csv`.

### 3. Build the vector index

```bash
python -m src.data_prep              # full dataset (~230k recipes, takes a few minutes)
python -m src.data_prep --limit 2000 # quick test run
```

### 4. Set your API key (for LLM generation)

```bash
cp .env.example .env
# edit .env and add your GROQ_API_KEY
```

Groq offers a free tier and low-latency inference. Supported providers: `groq`, `openai`, `ollama`, `llama-cpp`, `hf` (HuggingFace local).

## Usage

### Gradio app

```bash
python -m src.app
```

### Python API

```python
from src.agent import run_agent

result = run_agent(
    ingredient_text="chicken, broccoli, garlic, soy sauce",
    constraints={"dietary": ["dairy-free"], "max_time_minutes": 30},
    top_k=3,
)
print(result.recipe_markdown)
```

### CLI eval

```bash
python -m eval.eval                  # RAG + dietary filter (requires built index)
python -m eval.eval --mode no-rag    # baseline, no index required
python -m eval.eval --mode ab        # side-by-side A/B comparison
```

## Project Structure

```
src/
  data_prep.py   — cleans RAW_recipes.csv and builds the ChromaDB vector index
  retrieval.py   — semantic search over the index; returns top-k recipe candidates
  agent.py       — planner, dietary filter, critic; orchestrates the full pipeline
  generator.py   — pluggable LLM backends (Groq, OpenAI, HF, llama-cpp, Ollama)
  evaluate.py    — ingredient coverage and hallucination-rate utilities
  app.py         — Gradio UI

eval/
  eval.py            — eval harness: reliability, constraint pass rate, coverage, latency
  test_queries.json  — 50 test cases across 16 categories
  eval_results.json  — latest RAG eval output
  ab_results.json    — latest A/B comparison output

models/
  chroma/        — persisted ChromaDB vector index (not committed)
```

## CI

Every push runs the eval harness in no-RAG mode (no dataset required in CI) and fails the build if:
- Pipeline success rate drops below **95%**
- Dietary constraint pass rate drops below **78%**

See `.github/workflows/eval.yml`.

## Supported Dietary Constraints

`vegan` · `vegetarian` · `dairy-free` · `nut-free` · `gluten-free` · `shellfish-free` · `keto`

Pass any combination via the `constraints` dict or the Gradio UI (coming soon).
