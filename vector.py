"""
Vector store database setup; it vectorizes the documents.
Vector Search is a database which is hosted in the local machine using ChromaDB.
It allows us to quickly look up relevant information to pass to our model.
""" 
# The embedding model inputs and converts text into vectorized info.
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the csv file and store it into variable df (date frame).
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location ="./chroma_langchain_db"
# Check if the database already exists. If not, it converts
# the .csv file into vectors and adds it to the database.
add_documents = not os.path.exists(db_location)

# If we do need to add the file to the database, create an
# empty list for documents and ids.
if add_documents:
    documents = []
    ids = []

    # Iterate through the rows and access the entries.
    for i, row in df.iterrows():
        document = Document(
            page_content = row["Title"] + " " + row["Review"], 
            # metadata is additional info to extract but we won't query.
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            # get indices
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Create the vectore store
vector_store = Chroma(
    collection_name = "restaurant_reviews", 
    persist_directory = db_location, 
    embedding_function = embeddings
)

# If the database does not exist, add documents.
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Connect LLM and vector store.
retriever = vector_store.as_retriever(
    # Look up 5 relevant reviews.
    search_kwargs = {"k":5}
)