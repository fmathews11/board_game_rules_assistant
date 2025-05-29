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
from utils.tools import create_spirit_island_search_chain, create_standalone_question_from_chat_history

dotenv.load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentStatusLogger")

# Establish Constants
POSSIBLE_BOARD_GAMES = ["spirit_island", "wingspan", "scythe", "perch", "moonrakers"]
POSSIBLE_BOARD_GAMES_FORMATTED = ", ".join([game.replace("_", " ").title() for game in POSSIBLE_BOARD_GAMES])
MANUALS_DIR = "text"
GAME_IDENTIFICATION_MODEL_NAME = "gemini-2.0-flash-lite"
QA_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Key phrase for using Tavily search as a tool
TAVILY_SEARCH_MARKER = "[TAVILY_SEARCH_RECOMMENDED_FOR_SPIRIT_ISLAND]"
AGENT_CONFIG = {"recursion_limit": 10, "configurable": {"thread_id": str(uuid.uuid4())}}


class BoardGameAgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], lambda x, y: x + y]
    current_game_name: Optional[str]
    current_game_manual: Optional[str]
    identified_game_in_query: Optional[str]
    info_message_for_user: Optional[str]


game_identifier_llm = ChatGoogleGenerativeAI(model=GAME_IDENTIFICATION_MODEL_NAME,
                                             temperature=0,
                                             google_api_key=GEMINI_API_KEY,
                                             timeout=30)
qa_llm = ChatGoogleGenerativeAI(model=QA_MODEL_NAME,
                                temperature=1,
                                google_api_key=GEMINI_API_KEY,
                                max_tokens=50000)


def _load_game_manual(game_name: str) -> str:
    try:
        with open(f"{MANUALS_DIR}/{game_name}.txt", encoding='utf-8') as f:
            output = f.read()
        return output
    except FileNotFoundError:
        logger.error(f"Manual file not found: {MANUALS_DIR}/{game_name}.txt")
        return ""


def _extract_latest_human_message_content(state: BoardGameAgentState) -> Optional[str]:
    if not state.get("messages"):
        return None
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        return last_message.content
    return None


def identify_game_query_node(state: BoardGameAgentState) -> dict[str, None] | dict[str, str]:
    last_message_content = _extract_latest_human_message_content(state)
    if not last_message_content:
        return {"identified_game_in_query": None}
    prompt = GAME_IDENTIFIER_PROMPT.format(possible_board_games=str(POSSIBLE_BOARD_GAMES),
                                           last_message_content=last_message_content)
    logger.debug(f"Prompt for game identification: {last_message_content}")
    response = game_identifier_llm.invoke(prompt)
    text_response = response.content.strip()
    if text_response.lower() == "none" or not text_response:
        logger.debug(f"No game identified in query, or explicitly 'None'. Output: {text_response}")
        return {"identified_game_in_query": None}
    logger.debug(f"Game identified in query: {text_response}")
    return {"identified_game_in_query": text_response}


def manage_game_context_and_load_manual_node(state: BoardGameAgentState) -> dict[str, str | None]:
    newly_identified_game = state.get("identified_game_in_query")
    current_game = state.get("current_game_name")
    current_manual = state.get("current_game_manual")
    info_message = None
    game_to_use = None
    manual_to_use = None
    if not newly_identified_game:
        if current_game:
            logger.debug(f"Same game '{current_game}' mentioned. Using existing context.")
            game_to_use = current_game
            manual_to_use = current_manual
        else:
            logger.debug(f"No new game in query, using existing context: {current_game}")
            game_to_use = current_game
            info_message = f"Which board game are you asking about? I am able to answer questions about {POSSIBLE_BOARD_GAMES_FORMATTED}."
    else:
        if newly_identified_game == current_game:
            game_to_use = current_game
            manual_to_use = current_manual or _load_game_manual(current_game)
        else:
            game_to_use = newly_identified_game
            manual_to_use = _load_game_manual(newly_identified_game)

    return {
        "current_game_name": game_to_use,
        "current_game_manual": manual_to_use,
        "info_message_for_user": state.get("info_message_for_user") or info_message
    }


def generate_answer_node(state: BoardGameAgentState) -> dict:
    info_message = state.get("info_message_for_user")
    current_game = state.get("current_game_name")
    manual = state.get("current_game_manual")
    last_user_query = _extract_latest_human_message_content(state)

    if not last_user_query:
        return {"messages": [AIMessage(content="An unexpected error occurred. Please try again.")]}

    if info_message:
        return {"messages": [AIMessage(content=info_message)]}

    if not current_game or not manual:
        return {
            "messages": [AIMessage(
                content="I'm not sure which game you're referring to. Could you please specify?")]}

    previous_messages = state["messages"][:-1]
    previous_messages_formatted = {type(i).__name__: i.content for i in previous_messages}
    prompt_template = QA_PROMPT_TEMPLATE.format(current_game=current_game,
                                                manual=manual,
                                                previous_messages_formatted=previous_messages_formatted,
                                                last_user_query=last_user_query)
    llm_response = qa_llm.invoke(prompt_template)
    llm_output_text = llm_response.content

    should_use_tavily = False
    # Condition under which we want to use Tavily Search to enhance the spirit island text
    if current_game == "spirit_island" and TAVILY_SEARCH_MARKER in llm_output_text:
        should_use_tavily = True
        # Remove the marker from the LLM's output
        llm_output_text = llm_output_text.replace(TAVILY_SEARCH_MARKER, "").strip()

    if should_use_tavily:
        question_to_pass = create_standalone_question_from_chat_history(chat_history=previous_messages,
                                                                        users_question=last_user_query,
                                                                        llm=game_identifier_llm)
        logger.debug(f"Using Tavily Search to enhance the spirit island text to answer: {question_to_pass}")
        tavily_search_tool = create_spirit_island_search_chain(qa_llm)
        return {"messages": [AIMessage(content=tavily_search_tool.invoke(question_to_pass))]}

    return {"messages": [AIMessage(content=llm_output_text)]}


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
    agent_conversation_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "current_game_name": None,
        "current_game_manual": None,
        "identified_game_in_query": None,
        "info_message_for_user": None
    }
    print("I am here to help you with all of your board game questions!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {'exit', 'quit'}:
            print("Goodbye!")
            break
        if user_input == "debug_":
            print(f"Current game: {agent_conversation_state['current_game_name']}")
            print(f"Manual loaded: {'Yes' if agent_conversation_state['current_game_manual'] else 'No'}")
            print(f"Identified game in query: {agent_conversation_state['identified_game_in_query']}")
            print(f"Info message for user: {agent_conversation_state['info_message_for_user']}")
            print(f"Message history count: {len(agent_conversation_state['messages'])}")
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

        ai_response_message = result_state['messages'][-1]
        print(f"Agent: {ai_response_message.content}")

        agent_conversation_state['current_game_name'] = result_state.get('current_game_name')
        agent_conversation_state['current_game_manual'] = result_state.get('current_game_manual')
        agent_conversation_state['messages'] = result_state['messages']


if __name__ == '__main__':
    execute_agent()
