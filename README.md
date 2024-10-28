# GraphMind 🧠

GraphMind is an intelligent knowledge graph assistant that combines the power of Large Language Models with graph-based information retrieval as well as traditional unstructured information retrieval. It processes YouTube content into a structured knowledge graph along with text chunks and provides contextually aware responses to user queries.


## 🌟 Features

- **YouTube Content Processing**: Automatically extracts and processes transcripts from YouTube videos
- **Knowledge Graph Generation**: Creates a structured Neo4j graph database from processed content
- **Intelligent RAG**: Implements Hybrid Retrieval-Augmented Generation for accurate responses
- **Context-Aware Responses**: Maintains conversation context for more relevant answers
- **Interactive Chat Interface**: User-friendly Streamlit interface for easy interaction

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Neo4j Database
- Google AI API Key
- Groq API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Krish-Goyani/GraphMind
cd GraphMind
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_ai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
GROQ_API_KEY=your_groq_api_key
```

### Running the Application

```bash
streamlit run app.py
```

## 🎯 Usage

1. **Adding Content**:
   - Enter YouTube URLs in the sidebar
   - Click "Populate Graph" to process content
   - Wait for confirmation of successful processing

2. **Querying the Knowledge Graph**:
   - Type your question in the chat input
   - GraphMind will provide context-aware responses
   - Follow-up questions maintain conversation context

3. **Managing the Graph**:
   - Use "Reset Graph" to clear all data

## 🛠️ Technical Details

- **Frontend**: Streamlit
- **LLM Integration**: LangChain with Google Gemini and Llama-3.2(Groq)
- **Database**: Neo4j Graph Database
- **RAG Implementation**: Hybrid GraphRAG with structured as well as unstructured data

Made with ❤️ by Krish Goyani
