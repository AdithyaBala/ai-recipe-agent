# ai-recipes2
AI Recipe Generator from Available Ingredients

# Data Instructions

This project uses the Food.com "RAW_recipes.csv" file from Kaggle.

1. Create or log in to a Kaggle account.
2. Go to the **Food.com Recipes and Interactions** dataset.
3. Download `RAW_recipes.csv`.
4. Place it at: `data/RAW_recipes.csv`
5. Run:

   ```bash
   python -m src.data_prep_foodcom
   python -m src.retrieval
