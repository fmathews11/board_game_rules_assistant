import os
import dotenv
import logging
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

dotenv.load_dotenv()

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("AgentStatusLogger")

POSSIBLE_BOARD_GAMES = ["spirit_island", "wingspan"]
MANUALS_DIR = "text"
# Using flash preview for both now, but may move to a lower latency model for the game ID model
GAME_IDENTIFICATION_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
QA_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
You are a helpful assistant.  Your job is to help users answer questions about board games by referencing the rules.
"""


# Define agent state
class BoardGameAgentState(TypedDict):
    # The lambda expression here tells Langgraph how to handle the messages
    messages: Annotated[
        list[HumanMessage | AIMessage | SystemMessage], lambda existing_msg, new_msg: existing_msg + new_msg]
    current_game_name: Optional[str]
    current_game_manual: Optional[str]
    # Temporary field to hold game name identified from the latest query - Only is populated if the first layer
    # Detects a new game
    identified_game_in_query: Optional[str]
    # Placeholder for any messages outside the final AIMessage
    info_message_for_user: Optional[str]


game_identifier_llm = ChatGoogleGenerativeAI(model=GAME_IDENTIFICATION_MODEL_NAME,
                                             temperature=0,
                                             google_api_key=GEMINI_API_KEY)
qa_llm = ChatGoogleGenerativeAI(model=QA_MODEL_NAME,
                                temperature=0.2,  # Going with a bit of temperature here for some creativity
                                google_api_key=GEMINI_API_KEY)


# ----- NODES ---
def identify_game_query_node(state: BoardGameAgentState) -> dict[str, None] | dict[str, str]:
    """
    This node checks for a new game in the latest user message.
    
    If the LLM detects a new game, it updates the state with the new game name. If no new game name is detected,
    none is returned and the state is not updated.
    :param state: BoardGameAgentState
    :return : A dictionary with the key "identified_game_in_query" and the value as the identified game name.
    """
    # Get last message
    last_message = state["messages"][-1]
    # Edge case - If the last message is not a HumanMessage, return None
    if not isinstance(last_message, HumanMessage):
        return {"identified_game_in_query": None}

    prompt = f"""
        You are a helpful assistant.  You have one task -  Extract a board game name from the following user's message.
        The possible options are {str(POSSIBLE_BOARD_GAMES)}
        If no specific board game is mentioned, or if the message is a follow-up that doesn't restate the game, output 'None'.
        User query: "{last_message.content}"
        Game Name:
        """
    response = game_identifier_llm.invoke(prompt)
    text_response = response.content.strip()
    # Handling "None" vs a Python None object
    if text_response.lower() == "none" or not text_response:
        logger.debug(f"No game identified in query, or explicitly 'None'. Output: {text_response}")
        return {"identified_game_in_query": None}
    logger.debug(f"Game identified in query: {text_response}")
    return {"identified_game_in_query": text_response}


def manage_game_context_and_load_manual_node(state: BoardGameAgentState):

    # Get attributes from current state
    newly_identified_game = state.get("identified_game_in_query")
    current_game = state.get("current_game_name")
    current_manual = state.get("current_game_manual")
    info_message = None
    game_to_use = None
    manual_to_use = None

    # If the user does not mention a new game
    if not newly_identified_game:
        # One of two options here - we already know what game we're talking about, or we don't know yet
        if current_game:
            # We already know what game we're talking about, so we can use the existing context
            logger.debug(f"Same game '{current_game}' mentioned. Using existing context.")
            game_to_use = current_game
            manual_to_use = current_manual
        else:
            # We don't know what game we're talking about, so we need to ask the user
            logger.debug(f"No new game in query, using existing context: {current_game}")
            game_to_use = current_game
            # Here we want to create a message to let the user know that we don't know what game is being talked about
            info_message = "Which board game are you asking about?"


# Define the graph
builder = StateGraph(BoardGameAgentState)
builder.add_node("identify_game_query", identify_game_query_node)
builder.set_entry_point("identify_game_query")
compiled_graph = builder.compile()


def execute_agent():
    agent_conversation_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "current_game_name": None,
        "current_game_manual": None,
        "identified_game_in_query": None,
        "info_message_for_user": None
    }
    config = {"recursion_limit": 10}

    # Initiate loop
    while True:
        user_input = input("I am here to help you with all of your board game questions!")
        # Break condition
        if user_input in {'exit', 'quit'}:
            break

        agent_conversation_state["messages"].append(HumanMessage(content=user_input))
        # Since we're checking at each message, we need to reset these values to None
        agent_conversation_state['identified_game_in_query'] = None
        agent_conversation_state['info_message_for_user'] = None

        result_state = compiled_graph.invoke(agent_conversation_state, config=config)
        logger.debug(f"Result state: {result_state}")


if __name__ == '__main__':
    execute_agent()
