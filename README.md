MongoDB Persistent Memory Chatbot
An intelligent, persistent AI assistant built with Streamlit, LangChain, MongoDB and Groq. Unlike standard chatbots that forget everything when you refresh the page, this bot uses MongoDB to actively extract, update, and remember personal facts, preferences, and long-term context across all your conversations.

✨ Key Features
Smart Long-Term Memory: Automatically extracts durable, personal facts (e.g., name, location, preferences) and stores them in MongoDB.

Recency & Context Override: Dynamically injects updated memory facts right before the user's prompt, forcing the LLM to prioritize your current profile over outdated chat history.

General Knowledge Filter: Strictly trained to only remember personal data. It intelligently ignores general trivia, coding questions, and casual greetings to prevent database bloat.

Privacy & Security Guardrails: Built-in secret detection prevents the AI from saving passwords, API keys, or sensitive tokens to the database.

Session Management: Start new chats, resume past conversations, and delete old histories right from the sidebar.

🛠️ Tech Stack
Frontend: Streamlit

AI & Orchestration: LangChain, Groq API (openai/gpt-oss-20b or equivalent)

Database: MongoDB (Local)

Data Validation: Pydantic

📋 Prerequisites
Before you begin, ensure you have the following installed:

Python 3.8+

MongoDB: Must be installed and running locally on port 27017 (or update the URI in .env).

Groq API Key: Get one for free from the Groq Console.

🚀 Installation & Setup
1. Clone the repository

Bash
git clone https://github.com/damini-tomer/MongoDB-Persistent-Memory-Chatbot.git
cd MongoDB-Persistent-Memory-Chatbot
2. Create a virtual environment

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies

Bash
pip install -r requirements.txt
4. Set up environment variables
Create a .env file in the root directory and add your credentials:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
5. Verify MongoDB is running
Ensure your local MongoDB instance is active. The application will connect to mongodb://localhost:27017/ and automatically create the chatbot_db database along with messages and memories collections.

💻 Running the App
Start the Streamlit server:

Bash
streamlit run main.py
The app will automatically open in your browser at http://localhost:8501.

📁 Project Structure
main.py: The core Streamlit application, chat interface, and LangChain memory extraction logic.

database.py: MongoDB connection setup and collection indexing.

requirements.txt: Python package dependencies.

.gitignore: Security exclusions for virtual environments and API keys.

👤 Author
Damini Tomer

GitHub: @damini-tomer
