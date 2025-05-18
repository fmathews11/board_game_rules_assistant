import streamlit as st
from board_game_agent import compiled_graph, SYSTEM_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.title("Board Game Rules Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]


def clear_chat():
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]


# Function to run the agent with user input
def run_agent(user_message: str):
    # Append user message to the session state messages
    st.session_state.messages.append(HumanMessage(content=user_message))
    st.chat_message('human').write(user_message)

    # Prepare the state for the graph execution
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

    # Execute the graph
    config = {"recursion_limit": 10}
    result_state = compiled_graph.invoke(agent_state, config=config)

    # Update session state with the results from the graph
    ai_message = result_state['messages'][-1]
    st.session_state.messages.append(ai_message)  # Append the latest AIMessage
    st.chat_message("assistant").write(ai_message.content)  # Write the AI message immediately
    st.session_state.current_game_name = result_state.get('current_game_name')
    st.session_state.current_game_manual = result_state.get('current_game_manual')


# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:  # Not necessary but leaving for readability
        pass

# Chat input and send button
if user_message := st.chat_input("Ask a question about a board game..."):
    run_agent(user_message)

# Add a button to clear the chat history
st.button("Start Over", on_click=clear_chat)
