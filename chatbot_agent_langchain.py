import os
import requests
import random
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent

import warnings
warnings.filterwarnings('ignore')

# TheMealDB cuisines
SUPPORTED_CUISINES = {
    'american', 'british', 'canadian', 'chinese', 'croatian', 'dutch', 
    'egyptian', 'filipino', 'french', 'greek', 'indian', 'irish', 'italian', 
    'jamaican', 'japanese', 'kenyan', 'malaysian', 'mexican', 'moroccan', 
    'polish', 'portuguese', 'russian', 'spanish', 'thai', 'tunisian', 
    'turkish', 'vietnamese'
}

@tool
def find_recipe(query: str) -> str:
    """
    Searches for a recipe on TheMealDB API. Use this tool when a user asks for a recipe.
    The input to this tool should be a single main ingredient OR a cuisine type.
    For example: 'chicken' or 'italian'.
    """

    query = query.lower().strip()

    try:
        # Checking if query contains a cuisine
        if query in SUPPORTED_CUISINES:
            url = f"https://www.themealdb.com/api/json/v1/1/filter.php?a={query}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get('meals'):
                recipe = random.choice(data['meals'])
                recipe_url = f"https://www.themealdb.com/meal/{recipe['idMeal']}"
                return f"I found a great {query.capitalize()} recipe for you! Try {recipe['strMeal']}. You can see it here: {recipe_url}"
            else:
                return f"I couldn't find any {query} recipes right now."
        else:
            # Assume query contains ingredient
            url = f"https://www.themealdb.com/api/json/v1/1/filter.php?i={query}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get('meals'):
                recipe = random.choice(data['meals'])
                recipe_url = f"https://www.themealdb.com/meal/{recipe['idMeal']}"
                return f"I found a recipe with {query}! How about {recipe['strMeal']}? You can see it here: {recipe_url}"
            else:
                return f"I couldn't find any recipes with the main ingredient: {query}."
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return "I'm having trouble connecting to my recipe book right now."
    
tools = [find_recipe]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

# Prompt for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful recipe assistant. You have access to a tool that can find recipes by ingredient or cuisine."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
prompt = hub.pull("hwchase17/react-chat")

# Initialize agent with our tools
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Verbose for debugging

def chat():
    print("Welcome to the recipe chatbot! (Type 'quit' to exit)")
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Bot: Goodbye!")
            break
        
        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=result["output"]),
            ])

            print(f"Bot: {result['output']}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Bot: I'm sorry, I encountered a problem. Please try again.")

chat()