import unittest
from board_game_agent import generate_answer_node, BoardGameAgentState
from langchain_core.messages import HumanMessage, AIMessage


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


if __name__ == "__main__":
    unittest.main()