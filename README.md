# 🧀 Cheese Recommendation Agent

A powerful AI agent for cheese recommendations using MongoDB, Pinecone, and langgraph.

## Features

- **Advanced Query Analysis**: The agent analyzes user queries to determine the best data source (MongoDB or Pinecone) for retrieving information.
- **Semantic Search**: Uses Pinecone vector database for semantic similarity searches on cheese data.
- **Human-in-the-Loop**: Implements a robust human feedback mechanism to clarify ambiguous queries.
- **Reasoning Steps**: Displays the agent's reasoning process to help users understand how it arrived at recommendations.
- **MongoDB Integration**: Connects to a MongoDB database containing detailed cheese product information.
- **Streamlit UI**: User-friendly chat interface with reasoning step visualization and human-in-the-loop capabilities.

## Architecture

The agent is built using the langgraph library for workflow orchestration with several key components:

1. **Central Node**: The main reasoning engine that plans queries and coordinates the workflow.
2. **MongoDB Node**: Executes queries against the MongoDB database.
3. **Pinecone Node**: Performs semantic searches using the Pinecone vector database.
4. **Human-in-the-Loop Node**: Determines when human input is needed for clarification.
5. **Response Node**: Generates the final response to the user.

## Requirements

- Python 3.9+
- MongoDB connection
- Pinecone account (for vector database)
- OpenAI API key

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cheese-recommendation-agent.git
cd cheese-recommendation-agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

Create a `.env` file in the project root with the following variables:

```
MONGODB_URI="mongodb+srv://fullmax0111:Lhv4o4AEuoQVMfGg@cheese-agent.ozp8uxm.mongodb.net/"
MONGODB_DB="Cheese_Data"
OPENAI_API_KEY="your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_ENVIRONMENT="your-pinecone-environment"
```

## Running the Application

You can run the application in one of the following ways:

```bash
# Option 1: Using the run_app.py script
python run_app.py

# Option 2: Directly with Streamlit (recommended)
streamlit run streamlit_app.py

# Option 3: Running the app module directly
streamlit run src/app.py
```

## Using the Streamlit UI

The Streamlit UI provides a user-friendly way to interact with the cheese recommendation agent:

1. **Chat Interface**: Type your cheese-related queries in the chat input at the bottom of the page.
2. **Reasoning Steps**: View the agent's reasoning process in the expandable "Reasoning Steps" section.
3. **Human-in-the-Loop**: Provide additional information when prompted by the agent.

## First-time Setup

When you first run the application:

1. Click the "Test Connections" button in the sidebar to verify MongoDB and Pinecone connectivity.
2. Click "Load Data to Pinecone" to populate the vector database with embeddings from the MongoDB data.

## Usage

- Type your cheese-related queries in the chat input.
- View the agent's reasoning process in the expandable section below responses.
- Provide clarification when requested by the agent.
- Toggle display of reasoning steps in the sidebar.
- Start a new conversation with the "New Conversation" button.

## Example Queries

- "What are some good American cheeses?"
- "Find me cheeses that pair well with red wine"
- "What's the most popular cheese under $20?"
- "Tell me about cheddar cheese varieties"
- "What cheese is best for mac and cheese?"

## Project Structure

```
cheese-recommendation-agent/
├── main.py                        # Main entry point
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── src/
│   ├── app.py                     # Streamlit frontend
│   ├── config.py                  # Configuration settings
│   ├── database/
│   │   ├── mongodb_connector.py   # MongoDB connection
│   │   └── pinecone_connector.py  # Pinecone connection
│   ├── graph/
│   │   ├── nodes.py               # LangGraph node implementations
│   │   └── workflow.py            # LangGraph workflow definition
│   ├── models/
│   │   ├── prompts.py             # LLM prompts
│   │   └── schema.py              # Data schemas
│   └── utils/
│       └── data_loader.py         # Data loading utilities
```

## License

MIT 