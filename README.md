# AgroMate

## Getting Started
 
The Agromate uses the same virtual environment created in the root directory. Make sure you have:
 
1. Activated the virtual environment from the root directory:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```
 
2. Set up your API key:
   - Rename `.env.example` to `.env` in the greeting_agent folder
   - Add your Google API key to the `GOOGLE_API_KEY` variable in the `.env` file
 
## Running the Example
 
To run this basic agent example, you'll use the ADK CLI tool which provides several ways to interact with your agent:
 
1. Navigate to the 1-basic-agent directory containing your agent folder.
2. Start the interactive web UI:
```bash
adk web
```
 
3. Access the web UI by opening the URL shown in your terminal (typically http://localhost:8000)
 
4. Select your agent from the dropdown menu in the top-left corner of the UI
 
5. Start chatting with your agent in the textbox at the bottom of the screen
 
### Troubleshooting
 
If your agent doesn't appear in the dropdown menu:
- Make sure you're running `adk web` from the parent directory (1-basic-agent), not from inside the agent directory
- Check that your `__init__.py` properly imports the agent module
- Verify that `agent.py` defines a variable named `root_agent`
 
### Alternative Run Methods
 
The ADK CLI tool provides several options:
 
- **`adk web`**: Launches an interactive web UI for testing your agent with a chat interface
- **`adk run [agent_name]`**: Runs your agent directly in the terminal
- **`adk api_server`**: Starts a FastAPI server to test API requests to your agent

