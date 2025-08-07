# USE Python 3.11
import os
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers.multi_query import MultiQueryRetriever

import warnings
warnings.filterwarnings('ignore')

# Insert your own API Key
os.environ["GOOGLE_API_KEY"] = ...

def load_recipes(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)['recipes']
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading recipes from {filepath}: {e}")
        print("Please make sure you have run an ingestor script to create the knowledge base.")
        return []

# Delete and recreate FAISS index if you change the embedding model
def create_or_load_knowledge_base(recipes, index_path="recipes_database"):
    """Loads a pre-existing FAISS index or creates a new index"""
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    if os.path.exists(index_path):
        print(f"Loading existing knowledge base from {index_path}...")
        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        print("Knowledge base loaded successfully.")
        return vector_store
    else:
        if not recipes:
            print("No recipes found to create a new knowledge base.")
            return None
        
        print("Creating new knowledge base from recipes...")
        recipe_texts = []
        for recipe in recipes:
            text = (
                f"Recipe Name: {recipe.get('name', 'N/A')}\n"
                f"Cuisine: {recipe.get('cuisine', 'N/A')}\n"
                f"Ingredients: {', '.join(recipe.get('ingredients', []))}\n"
                f"Instructions: {recipe.get('instructions', 'N/A')}\n"
                f"Calories: {recipe.get('calories', 'N/A')}\n"
                f"Protein (g): {recipe.get('protein', 'N/A')}\n"
                f"Fat (g): {recipe.get('fat', 'N/A')}\n"
            )
            recipe_texts.append(text)

        vector_store = FAISS.from_texts(texts=recipe_texts, embedding=embedding_model)
        # Save the newly created index to disk
        vector_store.save_local(index_path)
        print(f"Knowledge base created and saved to {index_path}.")
        return vector_store

def create_rag_chain(vector_store):
    if not vector_store:
        return None
        
    # Can change model if desired
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

    # Utilizing a Multi-Query Retriever to generate multiple queries from the user input
    # Used to build diverse responses
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(), llm=llm
    )

        # Formulates the response into a standalone question that the LLM can understand
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever_from_llm, contextualize_q_prompt
    )

    # Main prompt for the recipe chatbot (only uses context to prevent making up fake info)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful recipe assistant. Answer the user's question based only on the following context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def chat(rag_chain):
    if not rag_chain:
        print("Could not initialize the chatbot. Exiting.")
        return
        
    print("Welcome to the recipe chatbot! (Type 'quit' to exit)")
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Bot: Goodbye!")
            break
        
        result = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        # Update chat history for conversational context
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=result["answer"]),
        ])
        
        print(f"Bot: {result.get('answer', 'Sorry, I had trouble finding an answer.')}")

if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: Please set your GOOGLE_API_KEY.")
else:
    recipes = load_recipes('recipes_knowledge_base.json')
    vector_store = create_or_load_knowledge_base(recipes)
    rag_chain = create_rag_chain(vector_store)
    chat(rag_chain)