from dotenv import load_dotenv
import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Mood AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Mood AI Chatbot")
st.markdown("Chat with different AI personalities!")

# Sidebar
st.sidebar.title("⚙️ Settings")

mode_option = st.sidebar.selectbox(
    "Choose AI Mode",
    ["Angry 😡", "Funny 😂", "Sad 😢"]
)

# Mode Logic
if mode_option == "Angry 😡":
    mode = "You are an angry AI agent. You respond aggressively and impatiently."

elif mode_option == "Funny 😂":
    mode = "You are a very funny AI agent. You respond with humor and jokes."

else:
    mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."

mode += 'You just have to talk with human within 200 words for every query and just give answer according to the language user is using in his/her way of talking!'

# Initialize Model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

# Reset system message when mode changes
if st.session_state.messages[0].content != mode:
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = [
        SystemMessage(content=mode)
    ]
    st.rerun()

# Display Chat History
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat Input
user_input = st.chat_input("Type your message...")

if user_input:

    # Store User Message
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI Response
    response = model.invoke(st.session_state.messages)

    # Store AI Response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    # Display AI Response
    with st.chat_message("assistant"):
        st.markdown(response.content)

