import streamlit as st
from board_game_agent import compiled_graph
from utils.prompts import SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import uuid
import time
import os

st.title("Board Game Help Chatbot")
st.markdown(
    """
I can help with:
 - Wingspan
 - Scythe
 - Spirit Island
 - Perch
 - Moonrakers
 - Tokaido
 
**Ask Away!**
"""
)
# Constants
CHAT_HISTORIES_PATH = "chat_histories/chat_histories.json"
TYPING_SIMULATION_DELAY = 0.0000002


def _load_all_chat_histories():
    """Loads all chat histories from the JSON file."""
    try:
        with open(CHAT_HISTORIES_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_chat_history(chat_uuid: str, chat_history: list):
    """Saves the current chat history to the JSON file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(CHAT_HISTORIES_PATH), exist_ok=True)
    histories = _load_all_chat_histories()
    temp_dict = {chat_uuid: chat_history}
    histories.append(temp_dict)
    with open(CHAT_HISTORIES_PATH, 'w') as f:
        json.dump(histories, f)


def _initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if 'current_chat_history' not in st.session_state:
        st.session_state.current_chat_history = []
        st.session_state.chat_uuid = str(uuid.uuid4())


def _clear_chat_and_reset_state():
    """
    Clears the current chat history and resets the state of the chat session. The current chat history is
    saved to a JSON file specified by `CHAT_HISTORIES_PATH` using the associated `chat_uuid` before
    resetting the session state.
    """
    if st.session_state.current_chat_history:
        _save_chat_history(st.session_state.chat_uuid, st.session_state.current_chat_history)
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.current_chat_history = []
    st.session_state.chat_uuid = str(uuid.uuid4())


def _load_chat_history(chat_uuid):
    """
    Loads a specific chat history by its UUID and sets it as the current chat.
    """
    histories = _load_all_chat_histories()
    for chat in histories:
        if chat_uuid in chat:
            st.session_state.current_chat_history = chat[chat_uuid]
            st.session_state.chat_uuid = chat_uuid
            st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
            for entry in chat[chat_uuid]:
                st.session_state.messages.append(HumanMessage(content=entry['user']))
                st.session_state.messages.append(AIMessage(content=entry['assistant']))
            break


def _display_chat_message(msg):
    """Displays a single chat message."""
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)


# Function to run the agent with user input and simulate typing
def run_agent_via_streamlit(user_message: str) -> None:
    """
    Processes a user message and interacts with a conversational agent to generate an answer with a typing simulation effect.
    :param user_message: A string containing the message input provided by the user for processing.
    :type user_message: str
    :return: None
    """
    # Append the user message to the session state messages
    st.session_state.messages.append(HumanMessage(content=user_message))
    # Only looking at the last 4 messages + system prompt messages to reduce tokens
    if len(st.session_state.messages) > 5:
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-4:]
    st.chat_message('human').write(user_message)
    # We need to extract only the HumanMessage and AIMessage for the graph's state
    graph_messages = [msg for msg in st.session_state.messages if
                      isinstance(msg, (HumanMessage, AIMessage, SystemMessage))]
    agent_state = {
        "messages": graph_messages,
        "current_game_name": st.session_state.get("current_game_name"),
        "current_game_manual": st.session_state.get("current_game_manual"),
        "identified_game_in_query": None,
        "info_message_for_user": None
    }
    # Execute the graph to get the full response first
    config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": st.session_state.chat_uuid}
    }
    result_state = compiled_graph.invoke(agent_state, config=config)
    ai_message_content = result_state['messages'][-1].content
    if streaming_option == "Yes":
        # Simulate typing effect
        with st.chat_message("ai"):
            message_placeholder = st.empty()
            full_response = ""
            for character in ai_message_content:
                full_response += character
                message_placeholder.markdown(full_response + "▌")
                time.sleep(TYPING_SIMULATION_DELAY)
            message_placeholder.markdown(full_response)
    else:
        st.chat_message("ai").write(ai_message_content)
    # Update the session state with the complete response
    st.session_state.messages.append(AIMessage(content=ai_message_content))
    st.session_state.current_game_name = result_state.get('current_game_name')
    st.session_state.current_game_manual = result_state.get('current_game_manual')
    st.session_state.current_chat_history.append({
        'user': user_message,
        'assistant': ai_message_content,
        'game': st.session_state.current_game_name
    })


# Main application logic
_initialize_session_state()
# Select box for streaming option
streaming_option = st.selectbox(
    "Stream response?",
    ("No", "Yes")
)
# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    _display_chat_message(msg)
# Chat input and send button
if user_input := st.chat_input("Ask a question about a board game..."):
    run_agent_via_streamlit(user_input)
st.button("Start New Chat", on_click=_clear_chat_and_reset_state, key="new_chat_button")
