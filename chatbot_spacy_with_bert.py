### USE Python 3.11
import json
import random
import re
import requests
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import warnings
warnings.filterwarnings('ignore')


class Chatbot:
    def __init__(self, intents_file, confidence_threshold=0.6):
        # Load the small English model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Please run 'python3 -m spacy download en_core_web_sm'")
            return

        self.intents = self.load_json_file(intents_file)
        self.confidence_threshold = confidence_threshold

        self.context = {}

        # Load Sentence-BERT pre-trained model and store pattern embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pattern_embeddings = None
        self.pattern_tags = []

        spacy_stopwords = self.nlp.Defaults.stop_words
        custom_stopwords = {'recipe', 'recipes', 'idea', 'ideas', 'suggestion', 'suggestions',
        'make', 'cook', 'with', 'for', 'and', 'a', 'an', 'some', 'any', 'something', 'food'}
        self.stopwords = spacy_stopwords.union(custom_stopwords)

        # Creates a list of supported cuisines that TheMealDB API uses
        self.supported_cuisines = {
            'american', 'british', 'canadian', 'chinese', 'croatian', 'dutch', 
            'egyptian', 'filipino', 'french', 'greek', 'indian', 'irish', 'italian', 
            'jamaican', 'japanese', 'kenyan', 'malaysian', 'mexican', 'moroccan', 
            'polish', 'portuguese', 'russian', 'spanish', 'thai', 'tunisian', 
            'turkish', 'vietnamese'
        }
        
        self.country_to_cuisine = {
            "america": "american", 
            "usa": "american", 
            "states": "american",
            "britain": "british", 
            "uk": "british", 
            "england": "british",
            "canada": "canadian",
            "china": "chinese",
            "croatia": "croatian",
            "holland": "dutch", 
            "netherlands": "dutch",
            "egypt": "egyptian",
            "philippines": "filipino",
            "france": "french",
            "greece": "greek",
            "india": "indian",
            "ireland": "irish",
            "italy": "italian",
            "jamaica": "jamaican",
            "japan": "japanese",
            "kenya": "kenyan",
            "malaysia": "malaysian",
            "mexico": "mexican",
            "morocco": "moroccan",
            "poland": "polish",
            "portugal": "portuguese",
            "russia": "russian",
            "spain": "spanish",
            "thailand": "thai",
            "tunisia": "tunisian",
            "turkey": "turkish",
            "vietnam": "vietnamese"
        }

        self.train()


    def load_json_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {filepath} was not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: The file {filepath} is an invalid JSON.")
            return


    def preprocess_text(self, text):
        # Processes text using spaCy for lemmatization and removal of punctuation
        doc = self.nlp(text.lower())
        return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]


    def train(self):
        if not self.intents:
            print("No intents file found.")
            return
        
        # Gathers all intent data in patterns, tags
        patterns = [pattern for intent in self.intents['intents'] for pattern in intent['patterns']]
        self.pattern_tags = [intent['tag'] for intent in self.intents['intents'] for _ in intent['patterns']]
        
        self.pattern_embeddings = self.model.encode(patterns, convert_to_tensor=True)
        

    def classify_intent(self, user_input):
        """Finds intent by cosine similarity"""
        tensorized_input = self.model.encode(user_input, convert_to_tensor=True)
        
        # Get the cosine similarities for all intents in relation to the user input intent
        cosine_similarities = util.pytorch_cos_sim(tensorized_input, self.pattern_embeddings)

        # Find the highest cosine similarity and corresponding index
        highest_score, best_match_index = torch.max(cosine_similarities, dim=1)
        
        if highest_score.item() >= self.confidence_threshold:
            # If similarity is high enough, return the predicted intent
            return self.pattern_tags[best_match_index.item()]
        else:
            # Otherwise, return fallback intent
            return 'fallback'


    def extract_entities(self, user_input, intent):
        """Extracts entities like ingredients or cuisine based on stopwords and POS"""
        entities = {}
        doc = self.nlp(user_input.lower())

        if intent == 'find_recipe_by_ingredients':
            ingredients = [token.text for token in doc if token.pos_ == 'NOUN' and token.text not in self.stopwords]

            if ingredients:
                entities['ingredients'] = ingredients

        elif intent == 'search_by_cuisine':
            for token in doc:
                if token.text in self.supported_cuisines:
                    entities['cuisine'] = token.text
                    return entities
                elif token.text in self.country_to_cuisine:
                    entities['cuisine'] = self.country_to_cuisine[token.text]
                    return entities

        return entities


    def find_recipe(self, entities):
        """Finds a recipe from TheMealDB API using entities dictionary"""
        try:
            if 'ingredients' in entities and entities['ingredients']:
                # TheMealDB's free API searches by one main ingredient
                ingredient = entities['ingredients'][0]
                url = f"https://www.themealdb.com/api/json/v1/1/filter.php?i={ingredient}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                if data['meals']:
                    recipe = random.choice(data['meals'])
                    # F-strings are different in this file, so you have to use Python 3.11
                    recipe_url = f"https://www.themealdb.com/meal/{recipe['idMeal']}"
                    return f"I found a recipe with {ingredient}! How about {recipe['strMeal']}? You can see it here: {recipe_url}"
                else:
                    return f"I couldn't find any recipes with the main ingredient: {ingredient}."

            elif 'cuisine' in entities:
                cuisine = entities['cuisine']
                # TheMealDB API uses 'Area' for cuisine
                url = f"https://www.themealdb.com/api/json/v1/1/filter.php?a={cuisine}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if data['meals']:
                    recipe = random.choice(data['meals'])
                    recipe_url = f"https://www.themealdb.com/meal/{recipe['idMeal']}"
                    return f"I found a great {cuisine.capitalize()} recipe for you! Try {recipe['strMeal']}. You can see it here: {recipe_url}"
                else:
                    return f"I don't have any {cuisine} recipes right now, but I'm always learning new ones!"
            
            return None
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return "I'm having trouble connecting to my recipe book right now. Please try again later."


    def get_generic_response(self, intent):
        # Grabs a random meal for random suggestion intent
        if intent == 'get_random_suggestion':
            try:
                url = "https://www.themealdb.com/api/json/v1/1/random.php"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                if data['meals']:
                    recipe = data['meals'][0]
                    recipe_url = f"https://www.themealdb.com/meal/{recipe['idMeal']}"
                    return f"Let's roll the dice! How about this one: {recipe['strMeal']}? You can see it here: {recipe_url}"
            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}")
                return "I'm having trouble finding a random recipe right now."

        # Loops through intents in intents file
        for i in self.intents['intents']:
            # If intent matches the given intent input, return one of the random responses
            if i['tag'] == intent:
                return random.choice(i['responses'])
        return "I'm not sure how to respond to that."


    def chat(self):
        print("Welcome to the recipe chatbot! (Type 'quit' to exit)")
        while True:
            user_input = input("You: ")

            # Quits out of the chatbot if user types 'quit' and prints a goodbye message
            if user_input.lower() == 'quit':
                print(f"Bot: {self.get_generic_response('goodbye')}")
                return
            
            response = ""
            if self.context.get('awaiting_entity'):
                # Reclassify new input's intent
                new_intent = self.classify_intent(user_input)
                
                # If the new intent is conversational, break the context
                if new_intent in ['greet', 'goodbye', 'help', 'affirm', 'deny', 'fallback']:
                    self.context.clear()
                    response = self.get_generic_response(new_intent)
                else:
                    # Otherwise, assume they are providing the missing info
                    original_intent = self.context.get('previous_intent')
                    entities = self.extract_entities(user_input, original_intent)
                    
                    recipe_response = self.find_recipe(entities)
                    if recipe_response:
                        response = recipe_response
                        self.context.clear()
                    else:
                        # If it still fails, give a generic response and clear context to avoid loop
                        response = self.get_generic_response(original_intent)
                        self.context.clear()

            else:
                intent = self.classify_intent(user_input)
                entities = self.extract_entities(user_input, intent)

                # Check to see if a cuisine was actually grabbed and not something else
                if intent == 'search_by_cuisine' and not entities:
                    intent = 'fallback'

                # Check if the intents are to find a recipe by ingredients or cuisine
                elif intent in ['find_recipe_by_ingredients', 'search_by_cuisine']:
                    if entities:
                        recipe_response = self.find_recipe(entities)
                        if recipe_response:
                            response = recipe_response
                        else:
                            response = self.get_generic_response(intent)
                    else:
                        # If no entities are found, ask a clarifying question and set context
                        response = self.get_generic_response(intent)
                        self.context['awaiting_entity'] = 'ingredients'
                        self.context['previous_intent'] = intent
                else:
                    # For all other intents get a generic response.
                    self.context.clear()
                    response = self.get_generic_response(intent)

            print(f"Bot: {response}")

bot = Chatbot('intents.json')
if hasattr(bot, 'intents') and bot.intents:
    bot.chat()