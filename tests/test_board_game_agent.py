import unittest
from board_game_agent import generate_answer_node, BoardGameAgentState
from langchain_core.messages import HumanMessage, AIMessage
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import numpy as np
from answer_benchmarks import WHAT_ARE_DAHAN_IN_SPIRIT_ISLAND_ANSWER

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
client = genai.Client(api_key=GEMINI_API_KEY)


def _get_gemini_embeddings(text: str) -> np.ndarray:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY'))

    return np.array(result.embeddings[0].values)


def _load_spirit_island_manual():
    with open("../text/spirit_island.txt", "r", encoding="utf-8") as f:
        return f.read()


def _cosine_similarity(vector1, vector2) -> float | None:
    """
  Computes the cosine similarity between two vectors using NumPy.

  """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return None  # Cosine similarity is undefined for zero vectors

    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


SPIRIT_ISLAND_TEXT = _load_spirit_island_manual()


class TestGenerateAnswerNode(unittest.TestCase):
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
        state = BoardGameAgentState(
            messages=[HumanMessage(content="What are dahan in spirit island?")],
            current_game_name="spirit_island",
            current_game_manual=SPIRIT_ISLAND_TEXT,
            identified_game_in_query=None,
            info_message_for_user=None
        )
        result = generate_answer_node(state)
        known_valid_answer = WHAT_ARE_DAHAN_IN_SPIRIT_ISLAND_ANSWER
        result_embedding = _get_gemini_embeddings(result["messages"][0].content)
        known_valid_answer_embedding = _get_gemini_embeddings(known_valid_answer)
        cosine_similarity_to_benchmark_answer = _cosine_similarity(result_embedding,
                                                                   known_valid_answer_embedding)
        print(cosine_similarity_to_benchmark_answer)
        self.assertGreater(cosine_similarity_to_benchmark_answer, 0.9)
        self.assertLess(cosine_similarity_to_benchmark_answer, 1.0)


if __name__ == "__main__":
    unittest.main()
