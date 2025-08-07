# AI Recipe Chatbot

This is a multi-version conversational AI chatbot that helps users discover recipes, either random or prompted with an ingredient or cuisine. The chatbot was built from the ground up, progressing from a classic hybrid ML model with an NLU core to a state-of-the-art RAG (Retrieval-Augmented Generation) architecture.

## Features

* **Intent Classification:** Uses a Sentence-BERT model to understand user intents for both simple conversational queries and complex recipe searches.
* **RAG System:** Leverages a powerful RAG system using the Gemini 2.5 Pro model through LangChain to answer questions based on a large recipe database of over 20,000 recipes.
* **Advanced Retrieval:** Implements a Multi-Query Retriever to provide more accurate and diverse search results given similar user inputs.
* **Conversational Memory:** Maintains chat history to understand and respond to follow-up questions intelligently.

## Tech Stack

* **Language:** Python 3.11
* **NLU & Machine Learning:** LangChain, Sentence-Transformers (BERT), spaCy, Scikit-learn, PyTorch, FAISS, NLTK
* **API Integration:** Requests, Google Gemini API, TheMealDB API

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nathan-anh-tran/AI-Recipe-Chatbot.git](https://github.com/nathan-anh-tran/AI-Recipe-Chatbot.git)
    cd AI-Recipe-Chatbot
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create --name chatbot_env python=3.11
    conda activate chatbot_env
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Build the Knowledge Base (for RAG models):**
    Download the recipe dataset from Kaggle: Epicurious - Recipes with Rating and Nutrition.

    Place the downloaded full_format_recipes.json file into your project folder.
    
    Run the ingestor script to create the knowledge base file:

    ```bash
    python ingest_from_epicurious.py
    ```

5.  **Set API Key:**
    Set your Google API key as an environment variable.
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

6.  **Run the chatbot:**
    ```bash
    python chatbot_final_ver.py
    ```

## Project Versions

This repository contains multiple versions of the chatbot, showing the progress of the overall project:
* `chatbot_nltk.py`: Initial version using NLTK and a TF-IDF vectorizer for text processing.
* `chatbot_spacy.py`: Upgrade to using spaCy for NLP tasks.
* `chatbot_spacy_with_bert.py`: Implemented a pre-trained Sentence-BERT model for better intent classification.
* `chatbot_agent_langchain.py`: Tested out using a LangChain agent for API calls.
* `chatbot_RAG.py`: Advanced version with a RAG architecture and used an established recipe dataset for more volume of recipes.
* `chatbot_final_ver.py`: Final hybrid version with a RAG architecture for retrieving recipes and the original NLU core for simple tasks.

## Final Notes

I started this project because I was moving into my apartment for college and was worried about my food situation.

The first few chatbots grab recipes from TheMealDB's REST API, which is free but doesn't have many recipes. I wanted to expand the knowledge of the chatbot, so the final few chatbot implementations using RAG actually use data from the 'recipes_knowledge_base.json' file, which I downloaded from Kaggle's dataset: Epicurious - Recipes with Rating and Nutrition, and it has over 20k recipes.

Most of the chatbots work by using an NLU core to classify the user's input intent, which is consistent until the chatbot_RAG.py. I tried to use an LLM (Gemini 2.5 Pro) instead to handle all the heavy work and generate responses that fit each user's intent seamlessly, but this was much slower and way more computationally expensive. For my final version, you can see the implementation where I have the NLU core for simple tasks like greeting the user or stating what the chatbot does, and the RAG system for retrieving specific recipes with queries including ingredients or cuisines. Each query that activates a simple request is almost instantaneous, but finding a recipe using the RAG system can take over 15-30 seconds.

The recipe_ingestor.py is there because I needed a way to convert the Kaggle dataset into a useable format that the chatbot could use. The recipes.json also exists because of earlier stages of the project where I was testing the chatbot with a local recipes file instead of an API.