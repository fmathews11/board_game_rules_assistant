import unittest
from deprecated.board_game_agent import generate_answer_node, BoardGameAgentState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from answer_benchmarks import (WHAT_ARE_DAHAN_IN_SPIRIT_ISLAND_ANSWER,
                               HOW_TO_GAIN_PRESENCE_IN_SPIRIT_ISLAND_ANSWER,
                               WHAT_IS_FEAR_IN_SPIRIT_ISLAND_ANSWER,
                               VITAL_STRENGTH_CARDS_ANSWER)
from tests.test_utility_functions import get_gemini_embeddings, cosine_similarity, create_spirit_island_state
from utils.prompts import SYSTEM_PROMPT
import time


class TestGenerateAnswerNode(unittest.TestCase):

    def _assert_cosine_similarity_between_answer_and_benchmark(
            self,
            result_messages: list,
            known_valid_answer: str,
            similarity_threshold: float,
            test_description: str,
            is_follow_up_message: bool = False
    ):
        ai_message_content = result_messages[-1].content if is_follow_up_message else result_messages[0].content
        result_embedding = get_gemini_embeddings(ai_message_content)
        time.sleep(10)  # Slowing down to eliminate rate limit errors
        known_valid_answer_embedding = get_gemini_embeddings(known_valid_answer)
        similarity = cosine_similarity(result_embedding, known_valid_answer_embedding)

        print(f"Cosine similarity for '{test_description}': {similarity}")
        self.assertIsNotNone(similarity, f"Cosine similarity was None for '{test_description}'")
        self.assertGreater(similarity, similarity_threshold,
                           f"Similarity for '{test_description}' ({similarity}) was not greater than {similarity_threshold}")

    def test_generate_answer_node_with_info_message(self):
        state = BoardGameAgentState(
            messages=[HumanMessage(content="What is the rule for scoring?")],
            current_game_name="Chess",
            current_game_manual=None,
            identified_game_in_query=None,
            info_message_for_user="The game is currently unavailable."
        )
        result = generate_answer_node(state)
        expected_message = "The game is currently unavailable."
        self.assertEqual(result["messages"][0].content, expected_message)

    def test_generate_answer_node_without_current_game(self):
        state = BoardGameAgentState(
            messages=[HumanMessage(content="How do I move the pawn?")],
            current_game_name=None,
            current_game_manual=None,
            identified_game_in_query=None,
            info_message_for_user=None
        )
        result = generate_answer_node(state)
        expected_message = "I'm not sure which game you're referring to. Could you please specify?"
        self.assertEqual(result["messages"][0].content, expected_message)

    def test_generate_answer_node_with_missing_keys(self):
        state = BoardGameAgentState(
            messages=[],
            current_game_name=None,
            current_game_manual=None,
            identified_game_in_query=None,
            info_message_for_user=None
        )
        result = generate_answer_node(state)
        expected_message = "An unexpected error occurred. Please try again."
        self.assertEqual(result["messages"][0].content, expected_message)

    def test_what_are_dahan_in_spirit_island_message(self):
        state = create_spirit_island_state("What are dahan in spirit island?")
        result = generate_answer_node(state)
        self._assert_cosine_similarity_between_answer_and_benchmark(
            result_messages=result["messages"],
            known_valid_answer=WHAT_ARE_DAHAN_IN_SPIRIT_ISLAND_ANSWER,
            similarity_threshold=0.9,
            test_description="What are Dahan"
        )

    def test_how_to_gain_presence_in_spirit_island_message(self):
        state = create_spirit_island_state("How do I gain presence in spirit island?")
        result = generate_answer_node(state)
        self._assert_cosine_similarity_between_answer_and_benchmark(
            result_messages=result["messages"],
            known_valid_answer=HOW_TO_GAIN_PRESENCE_IN_SPIRIT_ISLAND_ANSWER,
            similarity_threshold=0.85,
            test_description="How to gain presence"
        )

    def test_follow_up_what_is_fear_in_spirit_island_message_with_history(self):
        previous_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="How do you gain presence in spirit island?"),
            AIMessage(content=HOW_TO_GAIN_PRESENCE_IN_SPIRIT_ISLAND_ANSWER)
        ]
        state = create_spirit_island_state(
            user_message_content="yes,the last option",
            previous_messages=previous_messages
        )
        result = generate_answer_node(state)
        self._assert_cosine_similarity_between_answer_and_benchmark(
            result_messages=result["messages"],
            known_valid_answer=WHAT_IS_FEAR_IN_SPIRIT_ISLAND_ANSWER,
            similarity_threshold=0.9,
            test_description="What is fear (follow-up)",
            is_follow_up_message=True
        )

    def test_tavily_search_response(self):
        state = create_spirit_island_state(
            "In spirit island What are vital strength of the earth's starting cards and what do they do?"
        )
        result = generate_answer_node(state)
        self._assert_cosine_similarity_between_answer_and_benchmark(
            result_messages=result["messages"],
            known_valid_answer=VITAL_STRENGTH_CARDS_ANSWER,
            similarity_threshold=0.9,
            test_description="Vital strength of the earth's starting cards"
        )


if __name__ == "__main__":
    unittest.main()
