from dotenv import load_dotenv
from rich import print

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

print("Choose your AI mode")
print("Press 1 for angry mode")
print("Press 2 for funny mode")
print("Press 3 for sad mode")

choice = int(input("Tell your response :- "))

if choice == 1:
    mode = "You are an angry AI agent. You respond aggressively and impatiently."
elif choice == 2:
    mode = "You are a very funny AI agent. You respond with humor and jokes."
else:
    mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."

mode += 'You just have to talk with human within 100 words for every query'

messages = [
    SystemMessage(content=mode)
] # List of all messages like Human Message, AI Message, System Message.

print("----------------- welcom type 0 to exit the application-----------------")


# This chatbot has complete chat history!
while True:
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot : ", response.content)

print(messages)
