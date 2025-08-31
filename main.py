""" 
    The purpose of this project is to learn how to create a local AI agent 
    by following the YouTube video "How to Build a Local AI Agent With
    Python (Ollama,LangChain & RAG)" by Tech With Tim.

    Coder: humble-python
    Date: 08/31/2025
"""
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Set up an LLM using the downloaded llama3.2.
model = OllamaLLM(model="llama3.2")

# Specify what we want to model to do.
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# Use the ChatPromptTemplate class to pass a reviews variable
# and a question variable.
prompt = ChatPromptTemplate.from_template(template)
# Invoke the entire chain to combine all the things together
# to run the llm.
chain = prompt | model

while True:
    print("\n\n-----------------------------------------")
    question = input("Ask your question (press 'q' to quit): ")
    print("\n")
    if question =='q':
        break

    # Use the retriever (in vector.py) to extract relevant reviews
    # and pass them as a parameter to our prompt.
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)