import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from graph_builder import GraphBuilder
from graph_rag import GraphRAG

from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


llm = GoogleGenerativeAI(
            model = "gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )

def graph_content(progress_bar, status_text):
    """
    Entry point to generate a new graph. Will add controls to the UI 
    to perform these actions in the future
    """
    print("Building graph from content")
    graph_builder = GraphBuilder()
    urls = [
        "https://youtu.be/-apCR8R2_8A?si=pLK6szYqYHZgV9jI"
    ]

    status_text.text("Starting YouTube Content")
    progress_bar.progress(1/3)
    graph_builder.extract_youtube_transcripts(urls)
    status_text.text("Complete YouTube Content, starting Wikipedia Content")

    graph_builder.graph_index_generator()

def reset_graph():
    """
    Will reset the graph by deleting all relationships and nodes
    """
    graph_builder = GraphBuilder()
    graph_builder.reset_graph()

def get_response(question: str) -> str:
    """
    For the given question will formulate a search query and use a custom GraphRAG retriever 
    to fetch related content from the knowledge graph. 

    Args:
        question (str): The question posed by the user for this graph RAG

    Returns:
        str: The results of the invoked graph based question
    """
    graph_rag = GraphRAG()
    search_query = graph_rag.create_context_aware_search_query(st.session_state.chat_history, question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": lambda x: graph_rag.retriever(search_query),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"chat_history": st.session_state.chat_history, "question": question})

def landing_page():
    """
    Primary entry point for the app. Creates the chat interface that interacts with the LLM. 
    """
    st.title("Langchain RAG Bot")

    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm here to help. Ask me anything!")
        ]

    user_query = st.chat_input("Ask a question....")
    if user_query is not None and user_query != "":
        response = get_response(user_query)

         # Add the current chat to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Print the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

    with st.sidebar:
        st.header("Graph Management")
        st.write("Below are options to populate and reset your graph database")

        # Create two columns for the buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Populate Graph"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                graph_content(progress_bar, status_text)

        with col2:
            if st.button("Reset Graph"):
                reset_graph()

if __name__ == "__main__":
    landing_page()

    




