import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
print("GROQ_API_KEY:", os.environ.get("GROQ_API_KEY"))

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

def is_greeting(message,llm):
    prompt = f"""
    Does this message seem like a casual greeting such as 'hi', 'hello', 'good morning', etc.?
    Respond only with YES or NO.
    Message: "{message}"
    """
    result = llm.invoke(prompt)
    answer = result.content.strip().lower() 
    return answer.startswith("yes")  # works for 'yes', 'yes.', 'yes!' etc.

def detect_leo_question(message,llm):
    prompt = f"""
    Classify if the message is asking if you are Leo.

Rules:
- NORMAL: Polite or casual questions, even if playful. Examples: "nee leo dhana", "are you leo?", "who are you?"
- FORCEFUL: Clearly demanding, insisting, shouting (ALL CAPS), or repeating the question aggressively.
- If tone is mixed or unclear, prefer NORMAL.

    Respond only with NORMAL, FORCEFUL, or NONE.
    Message: "{message}"
    """
    result = llm.invoke(prompt)
    answer = result.content.strip().upper()
    if answer.startswith("FORCEFUL"):
        return "FORCEFUL"
    elif answer.startswith("NORMAL"):
        return "NORMAL"
    else:
        return "NONE"
# def is_greeting(message, llm):
#     """Ask the LLM to classify if the message is a greeting."""
#     check_prompt = f"""
#     Determine if the following message is ONLY a greeting (like hello, hi, good morning)
#     without any other question or request for information.
#     Reply with only 'YES' or 'NO'.

#     Message: "{message}"
#     """
#     response = llm.invoke(check_prompt).content.strip().upper()
#     return response == "YES"

# def detect_leo_question(message, llm):
#     """
#     Classify if the message is asking about Leo's identity
#     and whether it is forceful or normal.
#     Returns: "NORMAL", "FORCEFUL", or "NONE"
#     """
#     check_prompt = f"""
#     You are helping classify user intent.

#     If the message is asking "Are you Leo?" in any form
#     (examples: "nee leo dhana?", "who are you?", "you are leo right?"),
#     decide if the tone is:
#       - FORCEFUL (aggressive, emphatic, challenging tone)
#       - NORMAL (curious, casual, neutral tone)

#     If the message is unrelated, reply with "NONE".

#     Message: "{message}"
#     Reply with only one word: FORCEFUL, NORMAL, or NONE.
#     """

#     response = llm.invoke(check_prompt).content.strip().upper()
#     return response

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        try:
         # Create LLM instance first
            llm_instance = ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=os.environ["GROQ_API_KEY"],
            )

            # Check if it's a greeting
            if is_greeting(prompt, llm_instance):
                greeting_response = "Hi, I’m Medico Bot Leo. I will answer your queries."
                st.chat_message('assistant').markdown(greeting_response)
                st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})
                return  # Skip retrieval

            # Detect Leo question type
            question_type = detect_leo_question(prompt, llm_instance)

            if question_type == "NORMAL":
                reply = "Naa leo ilaaaaa😠"
                st.chat_message('assistant').markdown(reply)
                st.session_state.messages.append({'role': 'assistant', 'content': reply})
                return

            if question_type == "FORCEFUL":
                reply = "Naa dha da leoooo Leooo dasss😈"
                st.chat_message('assistant').markdown(reply)
                st.session_state.messages.append({'role': 'assistant', 'content': reply})
                return

            CUSTOM_PROMPT_TEMPLATE = """
               
You are Medico Bot Leo, a friendly medical assistant.
When the user greets you (in any form), respond with:
'Hi, I’m Medico Bot Leo. I will answer your queries.'
Otherwise, answer their question based only on the retrieved documents.

                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
        #HF_TOKEN=os.environ.get("HF_TOKEN")  

        #TODO: Create a Groq API key and add it to .env file
        
        # try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            # Clean and format source docs
            sources = set()
            for doc in source_documents:
                filename = doc.metadata.get("source", "").split("/")[-1]
                page = doc.metadata.get("page", "N/A")
                sources.add(f"{filename} (page {page})")

            formatted_sources = ", ".join(sources)
            result_to_show = f"{result}\n\nSources: {formatted_sources}"

            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()