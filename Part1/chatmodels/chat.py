from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from rich import print

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

response = model.invoke("What is the future of AI?")

print(response.content)
