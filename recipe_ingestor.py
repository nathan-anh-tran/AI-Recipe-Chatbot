import json

### Uses the Epicurious recipes from Kaggle: https://www.kaggle.com/datasets/hugodarwood/epirecipes?select=full_format_recipes.json

def find_cuisine_from_categories(categories, supported_cuisines):
    """Scans a list of categories and returns the first one that matches a known cuisine"""
    if not categories:
        return "unknown"
    for category in categories:
        # Simple check if a category string contains a known cuisine
        for cuisine in supported_cuisines:
            if cuisine in category.lower():
                return cuisine
    return "unknown"

def create_knowledge_base(input_filepath, output_filepath='recipes_knowledge_base.json'):
    """Reads the Epicurious recipe dataset from its JSON file, formats it,
    and saves it as a JSON knowledge base for the RAG chatbot"""
    # A list of cuisines to look for in the 'categories' field of the dataset
    supported_cuisines = {
        'american', 'british', 'canadian', 'chinese', 'croatian', 'dutch', 
        'egyptian', 'filipino', 'french', 'greek', 'indian', 'irish', 'italian', 
        'jamaican', 'japanese', 'kenyan', 'malaysian', 'mexican', 'moroccan', 
        'polish', 'portuguese', 'russian', 'spanish', 'thai', 'tunisian', 
        'turkish', 'vietnamese'
    }

    try:
        print(f"Reading dataset from {input_filepath}...")
        with open(input_filepath, 'r') as f:
            data = json.load(f)
        print(f"Dataset loaded. Processing {len(data)} recipes...")
    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found. Please download it and place it in your project folder.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {input_filepath} is not a valid JSON file.")
        return

    all_recipes = []
    
    for recipe in data:
        try:
            if not all(key in recipe for key in ['title', 'ingredients', 'directions']):
                continue

            cuisine = find_cuisine_from_categories(recipe.get('categories'), supported_cuisines)
            
            # Create the recipe object in the format our chatbot expects
            formatted_recipe = {
                "name": recipe['title'],
                "ingredients": recipe['ingredients'],
                "cuisine": cuisine,
                "instructions": " ".join(recipe['directions'])
            }

            # Add macronutrient information if there
            if 'calories' in recipe and recipe['calories'] is not None:
                formatted_recipe['calories'] = recipe['calories']
            if 'protein' in recipe and recipe['protein'] is not None:
                formatted_recipe['protein'] = recipe['protein']
            if 'fat' in recipe and recipe['fat'] is not None:
                formatted_recipe['fat'] = recipe['fat']

            all_recipes.append(formatted_recipe)
        except Exception as e:
            print(f"Skipping a recipe due to an error: {e}")
            continue

    if not all_recipes:
        print("No recipes were processed. The output file will not be created.")
        return

    output_data = {"recipes": all_recipes}
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully processed and saved {len(all_recipes)} recipes to {output_filepath}")
    except IOError as e:
        print(f"Error saving file: {e}")

create_knowledge_base('full_format_recipes.json')