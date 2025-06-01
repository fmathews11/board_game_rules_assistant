import os
import uuid
import dotenv
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from utils.create_logger import create_logger
from utils.prompts import SYSTEM_PROMPT, GAME_IDENTIFIER_PROMPT, QA_PROMPT_TEMPLATE
from utils.tools import create_spirit_island_search_chain, create_standalone_question_from_chat_history, \
    get_tavily_search_spirit_island_text, rephrase_query_for_better_search
from utils.constants import TAVILY_SEARCH_MARKER, INSUFFICIENT_SEARCH_RESULTS_MARKER

dotenv.load_dotenv()

# Configure logger
logger = create_logger("AgentLogger", 'debug')

# Establish Constants
POSSIBLE_BOARD_GAMES = ["spirit_island", "wingspan", "scythe", "perch", "moonrakers"]
POSSIBLE_BOARD_GAMES_FORMATTED = ", ".join([game.replace("_", " ").title() for game in POSSIBLE_BOARD_GAMES])
MANUALS_DIR = "text"
GAME_IDENTIFICATION_MODEL_NAME = "gemini-2.0-flash-lite"
QA_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Configuration for the agent
AGENT_CONFIG = {"recursion_limit": 10, "configurable": {"thread_id": str(uuid.uuid4())}}


class BoardGameAgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], lambda x, y: x + y]
    current_game_name: Optional[str]
    current_game_manual: Optional[str]
    identified_game_in_query: Optional[str]
    info_message_for_user: Optional[str]
    additional_context: Optional[str]


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


def _execute_tavily_search(previous_messages: List, query: str) -> str:
    """
    Executes a search using the Tavily search tool to enhance answers related to
    Spirit Island content. It attempts to improve insufficient search results by
    rephrasing the query and retrying.

    :param previous_messages: List containing the chat history
    :param query: Either a specific search query extracted from the LLM's response or the user's original question
    :return: The final answer obtained from the Tavily search tool
    """
    # Check if this is a specific search query or a general user query
    is_specific_query = len(query.split()) <= 10 and not query.endswith('?')

    if is_specific_query:
        # For specific queries extracted from the LLM's response, use them directly
        question_to_pass = query
        logger.debug(f"Using Tavily Search with specific query: {question_to_pass}")
    else:
        # For general user queries, create a standalone question from the chat history
        question_to_pass = create_standalone_question_from_chat_history(chat_history=previous_messages,
                                                                        users_question=query,
                                                                        llm=game_identifier_llm)
        logger.debug(f"Using Tavily Search with standalone question: {question_to_pass}")

    tavily_search_tool = create_spirit_island_search_chain(qa_llm)

    # Get initial search results
    initial_answer = tavily_search_tool.invoke(question_to_pass)

    # Check if the search results were insufficient
    if INSUFFICIENT_SEARCH_RESULTS_MARKER in initial_answer:
        logger.debug("Initial search results were insufficient. Rephrasing query and trying again.")
        # Remove the marker from the answer
        clean_initial_answer = initial_answer.replace(INSUFFICIENT_SEARCH_RESULTS_MARKER, "").strip()

        # Rephrase the query for better search results
        rephrased_query = rephrase_query_for_better_search(
            original_query=question_to_pass,
            previous_answer=clean_initial_answer,
            llm=qa_llm
        )
        logger.debug(f"Rephrased query: {rephrased_query}")

        # Try again with the rephrased query
        final_answer = tavily_search_tool.invoke(rephrased_query)

        # If still insufficient, use the best answer we have
        if INSUFFICIENT_SEARCH_RESULTS_MARKER in final_answer:
            logger.debug("Rephrased query still yielded insufficient results. Using best answer available.")
            final_answer = final_answer.replace(INSUFFICIENT_SEARCH_RESULTS_MARKER, "").strip()

            if is_specific_query:
                final_answer += (f"\n\nI've tried to find specific information about '{query}', but I couldn't find "
                                 f"detailed information. Would you like to know about other aspects of Spirit Island "
                                 f"instead?")
            else:
                final_answer += ("\n\nI've tried to find the most relevant information, but I may not have all the "
                                 "details"
                                 "you're looking for. Could you rephrase your question or ask about a different aspect of "
                                 "Spirit Island?")

        return final_answer

    return initial_answer


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
    additional_context = state.get("additional_context")
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

    # Prepare additional context section if it exists
    additional_context_section = ""
    if additional_context:
        additional_context_section = f"""
       You also have the following additional information from a search:
       --- ADDITIONAL CONTEXT START ---
       {additional_context}
       --- ADDITIONAL CONTEXT END ---"""

    # First pass: Check if the manual has enough information
    prompt_template = QA_PROMPT_TEMPLATE.format(
        current_game=current_game,
        manual=manual,
        previous_messages_formatted=previous_messages_formatted,
        additional_context_section=additional_context_section,
        last_user_query=last_user_query
    )

    llm_response = qa_llm.invoke(prompt_template)
    llm_output_text = llm_response.content

    # Check if the LLM indicates it needs more information
    if current_game == "spirit_island" and TAVILY_SEARCH_MARKER in llm_output_text:
        # Extract the specific missing information from the LLM's response
        import re
        search_pattern = re.compile(r'\[TAVILY_SEARCH_RECOMMENDED_FOR_SPIRIT_ISLAND\]\[(.*?)\]')
        search_match = search_pattern.search(llm_output_text)

        if search_match:
            specific_search_query = search_match.group(1).strip()
            logger.debug(f"Extracted specific search query: {specific_search_query}")

            # Get additional information from Tavily search using get_tavily_search_spirit_island_text
            search_results = get_tavily_search_spirit_island_text(specific_search_query)

            # Remove the search marker and specific query from the output
            clean_output = search_pattern.sub('', llm_output_text).strip()

            # Store the search results as additional context for future use
            updated_additional_context = search_results

            # Second pass: Generate answer with the additional context
            if not additional_context:
                # If this is the first time we're adding additional context, do a second pass
                additional_context_section = f"""
               You also have the following additional information from a search:
               --- ADDITIONAL CONTEXT START ---
               {search_results}
               --- ADDITIONAL CONTEXT END ---"""

                second_prompt_template = QA_PROMPT_TEMPLATE.format(
                    current_game=current_game,
                    manual=manual,
                    previous_messages_formatted=previous_messages_formatted,
                    additional_context_section=additional_context_section,
                    last_user_query=last_user_query
                )

                second_llm_response = qa_llm.invoke(second_prompt_template)
                final_answer = second_llm_response.content

                # Return both the updated state and the final answer
                return {
                    "messages": [AIMessage(content=final_answer)],
                    "additional_context": updated_additional_context
                }
            else:
                # If we already had additional context but still need more, combine the explanation with search results
                combined_answer = f"{clean_output}\n\nAdditional information:\n{search_results}"
                return {
                    "messages": [AIMessage(content=combined_answer)],
                    "additional_context": updated_additional_context
                }
        else:
            # Fallback to the original behavior if no specific query is found, using get_tavily_search_spirit_island_text
            search_results = get_tavily_search_spirit_island_text(last_user_query)

            # Store the search results as additional context for future use
            updated_additional_context = search_results

            return {
                "messages": [AIMessage(content=search_results)],
                "additional_context": updated_additional_context
            }

    # If the manual has enough information, just return the answer
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
        "info_message_for_user": None,
        "additional_context": None
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
            print(f"Additional context: {'Yes' if agent_conversation_state['additional_context'] else 'No'}")
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
        agent_conversation_state['additional_context'] = result_state.get('additional_context')


if __name__ == '__main__':
    execute_agent()
