# Board Game Rules Assistant
Board game rules can often be complex and challenging to understand. 
This repository was created to explore the use of language models in tandem with board game rule books to simplify the process of answering questions about game rules.
## How to Run
1.  **Set up  Environment:**
    *   Create a file named `.env` in the root directory of this project.
    *   Inside the `.env` file, add your Gemini API key like this:
        ```
        GEMINI_API_KEY=your_api_key_here
        ```
2.  **Create Manuals Directory:**
    *   Create a directory named `text` in the root directory of this project. This is where the agent will look for game manuals.
3.  **Add Game Manuals:**
    *   Place the pdf files (`.pdf`) of the board game manuals you want to use into the `pdf` directory.
    *   Execute `Extract_Raw_Text_From_PDF_Files.py](Extract_Raw_Text_From_PDF_Files.py` for each file
    *   The agent currently is configured to look for games like "spirit_island" and "wingspan" as defined in `POSSIBLE_BOARD_GAMES` in `board_game_agent.py`. You may need to update this list if you add other games.
4.  **Run the Streamlit App:**
    *   Open your terminal or command prompt.
    *   Navigate to the root directory of this project.
    *   Run the app using the following command:
        ```bash
        streamlit run app.py
        ```
5.  **Interact with the App:**
    *   Your web browser should open automatically with the Streamlit app running. If not, navigate to `http://localhost:8501`.
    *   Enter your questions about board game rules in the text input field.
    *   To specify a game, mention its name in your query (e.g., "In Spirit Island, how does...").