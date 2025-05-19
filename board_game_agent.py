import os
import dotenv
import logging
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

dotenv.load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentStatusLogger")

POSSIBLE_BOARD_GAMES = ["spirit_island", "wingspan", "scythe"]
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
                                temperature=1,  # Going with a bit of temperature here for some creativity
                                google_api_key=GEMINI_API_KEY,
                                max_tokens=50000)


def _load_game_manual(game_name: str) -> str:
    try:
        with open(f"{MANUALS_DIR}/{game_name}.txt", encoding='utf-8') as f:
            output = f.read()
    except Exception as e:
        print(f"Error loading manual for {game_name}: {e}")
        raise
    return output


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


def manage_game_context_and_load_manual_node(state: BoardGameAgentState) -> dict[str, str | None]:
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

    # Handling cases where the first node determined the user has mentioned game
    else:
        if newly_identified_game == current_game:
            game_to_use = current_game
            manual_to_use = current_manual or _load_game_manual(current_game)  # Edge case - have game but no manual
        else:
            game_to_use = newly_identified_game
            manual_to_use = _load_game_manual(newly_identified_game)

    return {
        "current_game_name": game_to_use,
        "current_game_manual": manual_to_use,
        "info_message_for_user": state.get("info_message_for_user") or info_message
    }


def generate_answer_node(state: BoardGameAgentState) -> dict:
    """
    Generate an answer node for the given board game state and user query.

    This function interprets the current state of the game as provided in the
    state object, determines the appropriate response to the last user query,
    and returns a message structure to relay the response.

    :param state: The current state of the board game agent.

    :return: A dictionary containing a single key "messages". The value of this key
        is a list of `AIMessage` objects, each representing a crafted response to
        the user's query.

    :raises KeyError: If required keys ("messages", "current_game_name", etc.)
        are missing from the `state` parameter.
    """
    info_message = state.get("info_message_for_user")
    current_game = state.get("current_game_name")
    manual = state.get("current_game_manual")

    # Edge cases
    # Check if messages list is not empty and the last message is a HumanMessage
    if not state.get("messages") or not isinstance(state["messages"][-1], HumanMessage):
        return {"messages": [AIMessage(content="An unexpected error occurred. Please try again.")]}
    # We now know the user's latest query is a HumanMessage, so we can extract it
    last_user_query = state["messages"][-1].content
    # If there was an error or clarification needed from previous step, prioritize info message
    if info_message:
        return {"messages": [AIMessage(content=info_message)]}
    if not current_game:
        return {
            "messages": [AIMessage(content="I'm not sure which game you're referring to. Could you please specify?")]}

    prompt_template = f"""
       You are a helpful board game rules assistant.
       You have the following rules manual for the game '{current_game}':
       --- MANUAL START ---
       {manual}
       --- MANUAL END ---

       --- HOW TO ANSWER --
        - Answer the user's question based ONLY on the provided manual.
        - Provide page number references to cite where the user can find this information.
        - If the answer is not found in the manual, clearly state that.
        - Do not make assumptions or use external knowledge.
        - Answer the question directly, as you are a subject matter expert.
        - DO NOT include "based on the provided manual" or "based on the context".
        - Be as verbose as necessary.  First provide a detailed explanation of the answer, then provide a short summary.
        - Use bullet points and/or markdown format to make the answer as easily-interpreted as possible.
        - Generate some potential follow up questions and suggest them to the user in a conversational manner.
        For instance: "Would you like to know more about *INSERT SUGGESTION(S) HERE*?"

       User's question: "{last_user_query}"
       Now it's your turn. Begin!:
       """
    llm_response = qa_llm.invoke(prompt_template)
    return {"messages": [AIMessage(content=llm_response.content)]}


# Define the graph
builder = StateGraph(BoardGameAgentState)
builder.add_node("identify_game_query", identify_game_query_node)
builder.add_node("manage_current_game_and_manual", manage_game_context_and_load_manual_node)
builder.add_node("generate_answer", generate_answer_node)

builder.add_edge(start_key="identify_game_query", end_key="manage_current_game_and_manual")
builder.add_edge(start_key="manage_current_game_and_manual", end_key="generate_answer")
builder.add_edge("generate_answer", END)
builder.set_entry_point("identify_game_query")
compiled_graph = builder.compile()


def execute_agent() -> None:
    """
    Executes an interactive board game assistant loop.

    This function facilitates communication with a conversational agent which specializes in board game rules assistance.
    It maintains a state for the conversation, including the ongoing messages, current game name, manual,
    newly identified game, and user-facing informational messages. Users can interact by typing queries,
    and the agent will process them and respond accordingly.

    :return: None
    """
    agent_conversation_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "current_game_name": None,
        "current_game_manual": None,
        "identified_game_in_query": None,
        "info_message_for_user": None
    }
    config = {"recursion_limit": 10}
    print("I am here to help you with all of your board game questions!")
    # Initiate loop
    while True:
        user_input = input("You: ")
        # Break condition
        if user_input in {'exit', 'quit'}:
            break
        if user_input == "debug_":
            print(f"Current game: {agent_conversation_state['current_game_name']}")
            print(f"Current manual: {agent_conversation_state['current_game_manual']}")
            print(f"Identified game in query: {agent_conversation_state['identified_game_in_query']}")
            print(f"Info message for user: {agent_conversation_state['info_message_for_user']}")
            continue

        # Ensure message list does not exceed 5
        if len(agent_conversation_state["messages"]) >= 5:
            # Keep the first message (SystemMessage) and the last 4 messages
            agent_conversation_state["messages"] = [agent_conversation_state["messages"][0]] + agent_conversation_state[
                                                                                                   "messages"][-4:]

        agent_conversation_state["messages"].append(HumanMessage(content=user_input))
        # Since we're checking at each message, we need to reset these values to None
        agent_conversation_state['identified_game_in_query'] = None
        agent_conversation_state['info_message_for_user'] = None

        result_state = compiled_graph.invoke(agent_conversation_state, config=config)
        logger.debug(f"Messages: {result_state['messages']}, Current game: {result_state['current_game_name']}")
        print(f"Agent: {result_state['messages'][-1].content}")

        # Ensure manual and current game persist for the next interaction
        agent_conversation_state['current_game_manual'] = result_state.get('current_game_manual')
        agent_conversation_state['current_game_name'] = result_state.get('current_game_name')


if __name__ == '__main__':
    execute_agent()
