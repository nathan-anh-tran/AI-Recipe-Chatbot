# USE Python 3.11
import os
import json
import random
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import SentenceTransformer, util
import torch

import warnings
warnings.filterwarnings('ignore')

class Chatbot:
    def __init__(self, intents_file, knowledge_base_file, confidence_threshold=0.6):
        # Initializing intents classifier compnents
        self.intents = self.load_json_file(intents_file)
        self.confidence_threshold = confidence_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # Using a pre-trained Sentence-BERT model
        self.pattern_embeddings = None
        self.pattern_tags = []
        self.train_intent_classifier()

        # Initializing RAG system
        self.recipes = self.load_json_file(knowledge_base_file, is_recipes=True)
        self.vector_store = self.create_or_load_knowledge_base(self.recipes)
        self.rag_chain = self.create_rag_chain(self.vector_store)
    
    def load_json_file(self, filepath, is_recipes=False):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if is_recipes:
                    return data['recipes']
                else:
                    return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def train_intent_classifier(self):
        if not self.intents:
            return
        # Extracts patterns and tags and embeds patterns
        patterns = [p for i in self.intents['intents'] for p in i['patterns']]
        self.pattern_tags = [i['tag'] for i in self.intents['intents'] for _ in i['patterns']]
        self.pattern_embeddings = self.model.encode(patterns, convert_to_tensor=True)

    def classify_intent(self, user_input):
        if self.pattern_embeddings is None:
            return 'fallback'
        # Embeds imput into tensor and computes cosine similarities between the pattern embeddings for all intents
        input_embedding = self.model.encode(user_input, convert_to_tensor=True)
        cosine_similarities = util.pytorch_cos_sim(input_embedding, self.pattern_embeddings)

        # Extracts intent with the highest cosine similarity
        score, index = torch.max(cosine_similarities, dim=1)

        # Returns intent only if cosine similarity is higher than confidence threshold
        if score.item() >= self.confidence_threshold:
            return self.pattern_tags[index.item()]
        return 'fallback'
    
    # Delete and recreate FAISS index if you change the embedding model
    def create_or_load_knowledge_base(self, recipes, index_path="recipes_database"):
        """Loads a pre-existing FAISS index or creates a new index"""
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # Using Sentence-BERT model again

        if os.path.exists(index_path):
            print(f"Loading vectorized recipe database from {index_path}...")
            return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        
        if not recipes:
            return None
        
        print("Creating new vectorized recipe database...")
        texts = []
        for r in recipes:
            text = (
                f"Title: {r.get('name', '')}\n"
                f"Cuisine: {r.get('cuisine', '')}\n"
                f"Ingredients: {', '.join(r.get('ingredients', []))}\n"
                f"Instructions: {r.get('instructions', '')}\n"
                f"Calories: {r.get('calories', 'N/A')}\n"
                f"Protein: {r.get('protein', 'N/A')}g\n"
                f"Fat: {r.get('fat', 'N/A')}g"
            )
            texts.append(text)

        vector_store = FAISS.from_texts(texts=texts, embedding=embedding_model)
        # Saves index to disk
        vector_store.save_local(index_path)
        print(f"Knowledge base saved to {index_path}.")
        return vector_store
    
    def create_rag_chain(self, vector_store):
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
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer the question, just reformulate it."),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever_from_llm, contextualize_q_prompt)
        
        # Main prompt for the recipe chatbot (only uses context to prevent making up fake info)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a recipe assistant. Answer the user's question based only on the following context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def get_generic_response(self, intent):
        # Logic for random recipe
        if intent == 'get_random_suggestion':
            if self.recipes:
                recipe = random.choice(self.recipes)
                response = f"Let's roll the dice! How about this one: **{recipe.get('name', 'Unknown Recipe')}**\n\n"
                response += f"**Cuisine:** {recipe.get('cuisine', 'N/A').capitalize()}\n"
                response += f"**Ingredients:** {', '.join(recipe.get('ingredients', []))}\n\n"
                response += f"**Instructions:** {recipe.get('instructions', 'No instructions available.')}\n\n"
                if 'calories' in recipe:
                    response += f"**Calories:** {recipe.get('calories', 'N/A')}\n"
                if 'protein' in recipe:
                    response += f"**Protein:** {recipe.get('protein', 'N/A')}g\n"
                if 'fat' in recipe:
                    response += f"**Fat:** {recipe.get('fat', 'N/A')}g"
                return response.strip()
            else:
                return "I couldn't find any recipes in my knowledge base to suggest one."
        
        # Grabs a random response from the intents file
        for i in self.intents['intents']:
            if i['tag'] == intent:
                return random.choice(i['responses'])
        return "I'm not sure how to respond to that."

    def chat(self):
        if not self.rag_chain:
            print("Chatbot could not be initialized. Exiting.")
            return
        
        print("Welcome to the recipe chatbot! (Type 'quit' to exit)")
        chat_history = []
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print(f"Bot: {self.get_generic_response('goodbye')}")
                return
            
            intent = self.classify_intent(user_input)

            if intent in ['greet', 'goodbye', 'help', 'affirm', 'deny', 'fallback', 'get_random_suggestion']:
                # For intents other than grab a recipe, use intents classifier
                response = self.get_generic_response(intent)
            else:
                # For recipe questions, use RAG chain
                result = self.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                response = result.get('answer', 'Sorry, I had trouble finding an answer.')
            
            # Adds to chat history both messages
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=response),
            ])
            
            print(f"Bot: {response}")

if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY environment variable not set. Please set it before running.")
else:
    bot = Chatbot('intents.json', 'recipes_knowledge_base.json')
    if bot.rag_chain:
        bot.chat()