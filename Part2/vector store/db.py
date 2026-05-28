from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Converts documents into embeddings and storing them into chromaDB and getting similiar embeddings
# using similarity search and retriever for making vector as search tool and querying questions gives
# most relevant embeddings or docs from the vector store.

load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings(
    model = "mistral-embed"
)

vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma-db"
)

# Similarity Search is the first strategy of Retriever Strategy.
result = vector_store.similarity_search("what is used for data analysis?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vector_store.as_retriever()

docs = retriver.invoke("What is Deep Learning?")

for d in docs:
    print(d.page_content)
