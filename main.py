import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from core.graph_builder import GraphBuilder
from core.graph_rag import GraphRAG
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from GraphMind_Logger import logger
from dotenv import load_dotenv
load_dotenv()


llm = GoogleGenerativeAI(
            model = "gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )


def populate_graph(urls):
    """
    Entry point to generate a new graph. Will add controls to the UI 
    to perform these actions in the future
    """
    try:
        graph_builder = GraphBuilder()
        graph_builder.extract_youtube_transcripts(urls)
        graph_builder.graph_index_generator()
        logger.info("Successfully populated graph")
    except Exception as e:
        logger.error(f"Error populating graph: {str(e)}")
        raise

def reset_graph():
    """
    Will reset the graph by deleting all relationships and nodes
    """
    try:
        graph_builder = GraphBuilder()
        graph_builder.reset_graph()
        logger.info("Successfully reset graph")
    except Exception as e:
        logger.error(f"Error resetting graph: {str(e)}")
        raise

def get_response(question: str) -> str:
    """
    For the given question will formulate a search query and use a custom GraphRAG retriever 
    to fetch related content from the knowledge graph. 

    Args:
        question (str): The question posed by the user for this graph RAG

    Returns:
        str: The results of the invoked graph based question
    """

    logger.info(f"Processing question: {question}")
    try:
        graph_rag = GraphRAG()
        search_query = graph_rag.create_context_aware_search_query(st.session_state.chat_history, question)
        logger.info(f"Created search query: {search_query}")

        template = """You are a knowledgeable assistant tasked with answering questions based on the provided context.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Answer ONLY based on the information provided in the context above
            2. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation
            3. Provide specific examples or references from the context to support your answer

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

        response = chain.invoke({"chat_history": st.session_state.chat_history, "question": question})
        logger.info("Successfully generated response")
        return response
    except Exception as e:
        logger.error(f"Error getting response: {str(e)}")
        raise
