import os
import uuid
import dotenv
import logging
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from utils.prompts import SYSTEM_PROMPT, GAME_IDENTIFIER_PROMPT, QA_PROMPT_TEMPLATE

dotenv.load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentStatusLogger")

POSSIBLE_BOARD_GAMES = ["spirit_island", "wingspan", "scythe", "perch"]
POSSIBLE_BOARD_GAMES_FORMATTED = ", ".join([game.replace("_", " ").title() for game in POSSIBLE_BOARD_GAMES])
MANUALS_DIR = "text"
# Using flash preview for both now, but may move to a lower latency model for the game ID model
GAME_IDENTIFICATION_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
QA_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Set an arbitrary recursion limit and a thread ID for memory.
AGENT_CONFIG = {"recursion_limit": 10, "configurable": {"thread_id": str(uuid.uuid4())}}


# Define agent state
class BoardGameAgentState(TypedDict):
    # The lambda expression here tells Langgraph how to handle the messages
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], lambda x, y: x + y]
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
    with open(f"{MANUALS_DIR}/{game_name}.txt", encoding='utf-8') as f:
        output = f.read()
    return output


def _extract_latest_human_message_content(state: BoardGameAgentState) -> Optional[str]:
    """
    Extracts the content of the latest HumanMessage from the state's messages list.
    Returns None if no HumanMessage is found as the last message or if the message list is empty.
    """
    if not state.get("messages"):
        return None
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return last_message.content
    return None


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
    last_message_content = _extract_latest_human_message_content(state)
    # Edge case - If the last message is not a HumanMessage, return None
    if not last_message_content:
        return {"identified_game_in_query": None}

    prompt = GAME_IDENTIFIER_PROMPT.format(possible_board_games=str(POSSIBLE_BOARD_GAMES),
                                           last_message_content=last_message_content)
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
            info_message = f"Which board game are you asking about? I am able to answer questions about {POSSIBLE_BOARD_GAMES_FORMATTED}."

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
    """
    info_message = state.get("info_message_for_user")
    current_game = state.get("current_game_name")
    manual = state.get("current_game_manual")

    # Edge case
    last_user_query = _extract_latest_human_message_content(state)
    if not last_user_query:
        return {"messages": [AIMessage(content="An unexpected error occurred. Please try again.")]}
    # If there was an error or clarification needed from previous step, prioritize info message
    if info_message:
        return {"messages": [AIMessage(content=info_message)]}
    if not current_game:
        return {
            "messages": [AIMessage(content="I'm not sure which game you're referring to. Could you please specify?")]}

    previous_messages = state["messages"][:-1]
    previous_messages_formatted = {type(i).__name__: i.content for i in previous_messages}
    prompt_template = QA_PROMPT_TEMPLATE.format(current_game=current_game,
                                                manual=manual,
                                                previous_messages_formatted=previous_messages_formatted,
                                                last_user_query=last_user_query)
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
memory = MemorySaver()
compiled_graph = builder.compile(checkpointer=memory)


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
            agent_conversation_state["messages"] = ([agent_conversation_state["messages"][0]] +
                                                    agent_conversation_state["messages"][-4:])

        agent_conversation_state["messages"].append(HumanMessage(content=user_input))
        # Since we're checking at each message, we need to reset these values to None
        agent_conversation_state['identified_game_in_query'] = None
        agent_conversation_state['info_message_for_user'] = None

        result_state = compiled_graph.invoke(agent_conversation_state, config=AGENT_CONFIG)
        logger.debug(f"Messages: {result_state['messages']}, Current game: {result_state['current_game_name']}")
        print(f"Agent: {result_state['messages'][-1].content}")

        # Ensure manual and current game persist for the next interaction
        agent_conversation_state['current_game_manual'] = result_state.get('current_game_manual')
        agent_conversation_state['current_game_name'] = result_state.get('current_game_name')


if __name__ == '__main__':
    execute_agent()
