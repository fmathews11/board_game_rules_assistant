import streamlit as st
from agent import graph, AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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
agent = graph

if "messages" not in st.session_state:
    st.session_state.messages = []


def _clear_chat_and_reset_state():
    """
    Clears the current chat history
    """
    st.session_state.messages = []


if __name__ == "__main__":
    st.button("Start New Chat", on_click=_clear_chat_and_reset_state, key="new_chat_button")
    # Display chat messages from history on app rerun
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    if user_input := st.chat_input("Ask a question about a board game..."):
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=user_input))
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare the state for the agent, including previous messages and the new user message
        current_state = AgentState(messages=st.session_state.messages)

        # Invoke the agent
        for s in graph.stream(current_state, stream_mode="values", debug=False):
            if not s.get('messages'):
                continue
            message = s["messages"][-1]
            if isinstance(message,ToolMessage):
                continue
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            if isinstance(message, AIMessage):
                ai_response = message.content

        # Extract the AI's response and add it to session state
        st.session_state.messages.append(AIMessage(content=ai_response))

        # Display AI's response in chat message container
        with st.chat_message("assistant"):
            st.markdown(ai_response)
