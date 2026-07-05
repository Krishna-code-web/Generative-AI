from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from rich import print

load_dotenv()

embeddings = MistralAIEmbeddings(
    model = 'mistral-embed'
)

texts = [
    "Hello this is Akarsh Vyas",
    "Hello your name is YouTube",
    "And you all are very beautiful"
]

vector = embeddings.embed_documents(texts)

print(vector)

