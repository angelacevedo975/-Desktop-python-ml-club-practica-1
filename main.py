from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI, ChatMessage, MessageRole
from llama_index.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from dotenv import load_dotenv
import os
load_dotenv()

vector_index = None
history = []

def initializeService():
    global vector_index
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5)

    promptFile = open('./data/prompt.txt')
    prompt = promptFile.read()
    #print("Using the following system prompt: ", prompt, sep='\n')

    service_context = ServiceContext.from_defaults(
        llm=llm, system_prompt=prompt)
    try:
        reader = SimpleDirectoryReader(
            input_dir='./data/context', recursive=False)
        docs = reader.load_data()
    except ValueError:
        print(
            f"Context directory is empty, using only prompt")
        docs = []

    vector_index = VectorStoreIndex.from_documents(
        docs, service_context=service_context)


def loadChat():
    global vector_index
    global history
    query_engine = vector_index.as_query_engine()

    chat_history = list(map(lambda item: ChatMessage(
        role=item['source'], content=item['message']),
        history
    ))

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        query_engine,
        chat_history=chat_history
    )

    return chat_engine

def chat(message):
    global history

    history.append({'source': MessageRole.USER, 'message': message})
    chat_engine = loadChat()
    response = chat_engine.chat(message)
    history.append({'source': MessageRole.SYSTEM, 'message': response.response})
    return response.response

if __name__ == "__main__":
    initializeService()

    question = input("Ask me anything: ")
    while question != "exit":
        print(chat(question))
        question = input("Ask me anything: ")