import os
import dotenv
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configuration ---
MANUALS_DIR = "text"
GAME_IDENTIFICATION_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
QA_MODEL_NAME = "gemini-2.5-flash-preview-04-17"


# --- Agent State ---
class BoardGameAgentState(TypedDict):
    # The lambda expression here tells Langgraph how to handle the messages
    messages: Annotated[List[BaseMessage], lambda existing, new: existing + new]
    current_game_name: Optional[str]
    current_game_manual: Optional[str]
    # Temporary field to hold game name identified from the latest query
    identified_game_in_query: Optional[str]
    # Temporary field for messages from nodes before final AIMessage
    info_message_for_user: Optional[str]


# --- Initialize LLMs ---
game_identifier_llm = ChatGoogleGenerativeAI(model=GAME_IDENTIFICATION_MODEL_NAME,
                                             temperature=0,
                                             google_api_key=GEMINI_API_KEY)
qa_llm = ChatGoogleGenerativeAI(model=QA_MODEL_NAME,
                                temperature=0.2,
                                google_api_key=GEMINI_API_KEY)


# --- Node Functions ---

def identify_game_in_query_node(state: BoardGameAgentState):
    """
    Identifies a game name from the latest user message.
    """
    print("---NODE: IDENTIFY GAME IN QUERY---")
    last_message = state["messages"][-1]
    # Should not happen in normal flow - edge case
    if not isinstance(last_message, HumanMessage):
        return {"identified_game_in_query": None}

    prompt = f"""
    Extract the board game name from the following user query.
    The possible options are ['spirit_island','wingspan]
    If no specific board game is mentioned, or if the query is a follow-up that doesn't restate the game, output 'None'.
    User query: "{last_message.content}"
    Game Name:
    """
    try:
        response = game_identifier_llm.invoke(prompt)
        identified_name = response.content.strip()
        if identified_name.lower() == "none" or not identified_name:
            print(f"No game identified in query, or explicitly 'None'. Output: {identified_name}")
            return {"identified_game_in_query": None}
        print(f"Game identified in query: {identified_name}")
        return {"identified_game_in_query": identified_name}
    except Exception as e:
        print(f"Error in identify_game_in_query_node: {e}")
        return {"identified_game_in_query": None,
                "info_message_for_user": "Sorry, I had trouble understanding which game you meant."}


def manage_game_context_and_load_manual_node(state: BoardGameAgentState):
    """
    Manages the game context (current_game_name, current_game_manual).
    If a new game is identified, it attempts to load its manual.
    If no game in current query, relies on existing context.
    """
    print("---NODE: MANAGE GAME CONTEXT AND LOAD MANUAL---")
    identified_game = state.get("identified_game_in_query")
    current_game = state.get("current_game_name")
    current_manual = state.get("current_game_manual")
    info_message = None

    game_to_use = None
    manual_to_use = None

    if identified_game:
        # New game mentioned in the query
        if identified_game != current_game:
            print(f"New game identified: {identified_game}. Attempting to load manual.")
            game_to_use = identified_game
            manual_path = os.path.join(MANUALS_DIR, f"{identified_game}.txt")
            try:
                with open(manual_path, 'r', encoding='utf-8') as f:
                    manual_to_use = f.read()
                print(f"Manual loaded for {identified_game}.")
            except FileNotFoundError:
                print(f"Manual not found for {identified_game} at {manual_path}.")
                info_message = f"Sorry, I couldn't find the rules manual for '{identified_game}'."
                # Keep game_to_use as identified_game, but manual_to_use will be None
            except Exception as e:
                print(f"Error loading manual for {identified_game}: {e}")
                info_message = f"Sorry, an error occurred while trying to load the manual for '{identified_game}'."
                manual_to_use = None
        else:
            # Same game mentioned again, use existing context if manual is there
            print(f"Same game '{identified_game}' mentioned. Using existing context.")
            game_to_use = current_game
            manual_to_use = current_manual  # Assumes it was loaded correctly before
            if not manual_to_use:  # If for some reason it wasn't loaded before, try now
                manual_path = os.path.join(MANUALS_DIR, f"{game_to_use}.txt")
                try:
                    with open(manual_path, 'r', encoding='utf-8') as f:
                        manual_to_use = f.read()
                    print(f"Manual loaded for {game_to_use} (previously missed).")
                except FileNotFoundError:
                    info_message = f"I previously couldn't find the rules manual for '{game_to_use}', and still can't."
                except Exception as e:
                    info_message = f"Sorry, an error occurred while trying to re-load the manual for '{game_to_use}'."


    else:  # No game identified in the current query
        if current_game:
            print(f"No new game in query, using existing context: {current_game}")
            game_to_use = current_game
            manual_to_use = current_manual
            if not manual_to_use and not state.get(
                    "info_message_for_user"):  # if game context exists but manual is missing and no prior error
                info_message = f"I know we were talking about {current_game}, but I don't have its manual loaded."
        else:
            print("No game identified in query and no existing game context.")
            info_message = "Which board game are you asking about?"

    return {
        "current_game_name": game_to_use,
        "current_game_manual": manual_to_use,
        "info_message_for_user": state.get("info_message_for_user") or info_message  # Prioritize existing error
    }


def generate_answer_node(state: BoardGameAgentState):
    """
    Generates an answer based on the game context, manual, and user query.
    """
    print("---NODE: GENERATE ANSWER---")
    info_message = state.get("info_message_for_user")
    current_game = state.get("current_game_name")
    manual = state.get("current_game_manual")
    last_user_query = state["messages"][-1].content

    if info_message:  # If there was an error or clarification needed from previous step
        print(f"Responding with info_message: {info_message}")
        return {"messages": [AIMessage(content=info_message)]}

    if not current_game:  # Should be caught by info_message, but as a fallback
        print("No current game, asking for clarification.")
        return {
            "messages": [AIMessage(content="I'm not sure which game you're referring to. Could you please specify?")]}

    if not manual:  # Game identified, but manual not loaded/found
        print(f"Game {current_game} identified, but no manual content.")
        # This case should ideally be handled by manage_game_context setting an info_message
        # If it reaches here, it means manage_game_context_and_load_manual_node thought it was okay,
        # or an earlier node set an info_message that got cleared.
        # For safety, let's ensure a response.
        no_manual_message = f"I understand you're asking about '{current_game}', but I was unable to load its rule manual."
        return {"messages": [AIMessage(content=no_manual_message)]}

    # We have a game, a manual, and no prior error messages for this turn
    print(f"Generating answer for '{current_game}' using its manual.")
    prompt_template = f"""
    You are a helpful board game rules assistant.
    You have the following rules manual for the game '{current_game}':
    --- MANUAL START ---
    {manual}
    --- MANUAL END ---

    Answer the user's question based ONLY on the provided manual.
    If the answer is not found in the manual, clearly state that.
    Do not make assumptions or use external knowledge.

    User's question: "{last_user_query}"
    Answer:
    """
    try:
        ai_response = qa_llm.invoke(prompt_template)
        print(f"LLM Answer: {ai_response.content}")
        return {"messages": [AIMessage(content=ai_response.content)]}
    except Exception as e:
        print(f"Error during QA LLM call: {e}")
        return {"messages": [AIMessage(
            content=f"Sorry, I encountered an error while trying to answer your question about {current_game}.")]}


# --- Graph Definition ---
builder = StateGraph(BoardGameAgentState)

builder.add_node("identify_game", identify_game_in_query_node)
builder.add_node("manage_context_and_load_manual", manage_game_context_and_load_manual_node)
builder.add_node("generate_answer", generate_answer_node)

builder.set_entry_point("identify_game")
builder.add_edge("identify_game", "manage_context_and_load_manual")
builder.add_edge("manage_context_and_load_manual", "generate_answer")
builder.add_edge("generate_answer", END)

compiled_graph = builder.compile()


# --- Main Interaction Loop ---
def run_agent():
    print("Board Game Rules Assistant is ready!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Please create a 'manuals' directory with .txt files for your games (e.g., manuals/Catan.txt)")
    if not os.path.exists(MANUALS_DIR):
        print(f"\nWARNING: The '{MANUALS_DIR}' directory does not exist. Please create it to load game manuals.\n")

    # Initial state for a conversation session
    # The state can persist across multiple turns for the same "config"
    # if you use a checkpointer or manage it outside the loop per session.
    # For this example, we'll create a fresh initial state for each session,
    # but the `current_game_name` and `current_game_manual` within the state
    # *will* persist across turns of a single `compiled_graph.invoke` sequence
    # if the graph is designed to pass them along.
    # LangGraph's `add_messages` automatically handles message history.

    # This state will be passed and updated by the graph.
    # For a continuous conversation, you'd reuse the output state as input for the next turn.
    current_conversation_state = {
        "messages": [SystemMessage(content="You are a helpful board game rules assistant.")],
        "current_game_name": None,
        "current_game_manual": None,
        "identified_game_in_query": None,
        "info_message_for_user": None
    }
    # config for invoke, useful with checkpointers for threading conversations
    # For now, a simple config. If using checkpointer, this would be e.g. {"configurable": {"thread_id": "user_123"}}
    config = {"recursion_limit": 10}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break

        # Append new user message to the messages list in the current state
        current_conversation_state["messages"].append(HumanMessage(content=user_input))
        # Reset temporary fields for the new turn
        current_conversation_state["identified_game_in_query"] = None
        current_conversation_state["info_message_for_user"] = None

        try:
            # Invoke the graph with the current state
            # The graph will update the state, including adding its AIMessage response
            result_state = compiled_graph.invoke(current_conversation_state, config=config)

            # The result_state now contains the full history, including the AI's last response
            ai_message_content = ""
            if result_state["messages"] and isinstance(result_state["messages"][-1], AIMessage):
                ai_message_content = result_state["messages"][-1].content
            else:  # Should not happen if graph ends with AIMessage
                ai_message_content = "Sorry, I had an issue formulating a response."

            print(f"Assistant: {ai_message_content}")

            # Persist the relevant parts of the state for the next turn
            current_conversation_state["current_game_name"] = result_state.get("current_game_name")
            current_conversation_state["current_game_manual"] = result_state.get("current_game_manual")
            # messages list is already updated by the `add_messages` annotator in GraphState

        except Exception as e:
            print(f"Assistant: An unexpected error occurred: {e}")
            # Optionally reset state or parts of it
            current_conversation_state = {
                "messages": [SystemMessage(content="You are a helpful board game rules assistant."),
                             HumanMessage(content="Let's start over.")],  # Start fresh
                "current_game_name": None,
                "current_game_manual": None,
                "identified_game_in_query": None,
                "info_message_for_user": None
            }


if __name__ == "__main__":
    run_agent()
