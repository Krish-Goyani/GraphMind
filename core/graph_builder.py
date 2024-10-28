import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader, TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import GoogleGenerativeAI
from GraphMind_Logger import logger
from dotenv import load_dotenv
load_dotenv()

class GraphBuilder:
    """
    Encapsulates the core functionality requires to build a full knowledge graph 
    from multiple sources of unstructured text
    """


    def __init__(self) -> None:

        self.graph = Neo4jGraph(
            url= os.getenv('NEO4J_URI'),
            username= os.getenv("NEO4J_USERNAME"),
            password= os.getenv("NEO4J_PASSWORD")
        )

        logger.info("Neo4j connection established")



        self.llm = GoogleGenerativeAI(
            model = "gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        logger.info("Google Gemini LLM initialized")
        
    def chunker(self, raw_docs):
        """
        Accepts raw text context extracted from source and applies a chunking 
        algorithm to it. 

        Args:
            raw_docs (str): The raw content extracted from the source

        Returns:
            List: List of document chunks
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256, 
            chunk_overlap=50
            )
        docs = text_splitter.split_documents(raw_docs)
        logger.info(f"Document chunking completed. Generated {len(docs)} chunks")

        return docs

    def graph_transformer(self, text_chunks):
        """
        Uses experimental LLMGraphTransformer to convert unstructured text into a knowledge graph

        Args:
            text_chunks (List): List of document chunks
        """
        logger.info("Initializing LLM Graph Transformer")
        llm_transformer = LLMGraphTransformer(llm=self.llm)

        logger.info("Converting text chunks to graph documents")
        graph_docs = llm_transformer.convert_to_graph_documents(text_chunks)
        
        logger.info("Adding documents to Neo4j graph")
        self.graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=True,
            include_source=True
        )
        logger.info("Graph transformation and storage completed")

    def chunk_and_graph(self, raw_docs):
        """
        Breaks the raw text into chunks and converts into a knowledge graph

        Args:
            raw_docs (str): The raw content extracted from the source
        """
        text_chunks = self.chunker(raw_docs)
        if text_chunks is not None:
            logger.info(f"Processing {len(text_chunks)} chunks for graph transformation")
            self.graph_transformer(text_chunks)
        else:
            logger.warning("No text chunks were generated from the raw documents")

    def extract_youtube_transcripts(self, urls):
        """
        Extracts multiple URLs YouTube transcripts

        Args:
            urls (List): A list of YouTube urls
        """
        for url in urls:
            try:
                transcript = YoutubeLoader.from_youtube_url(url).load()
                logger.info(f"Successfully extracted transcript from: {url}")
                self.chunk_and_graph(transcript)
            except Exception as e:
                logger.error(f"Failed to extract transcript from {url}. Error: {str(e)}")


    def graph_index_generator(self):
        """
        Creates an index on the populated graph tp assist with efficient searches
        """
        try:
            self.graph.query(
                "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
            )
            logger.info("Successfully created graph index")
        except Exception as e:
            logger.error(f"Failed to create graph index. Error: {str(e)}")

    def reset_graph(self):
        """
        WARNING: Will clear entire graph, use with caution
        """
        try:
            self.graph.query(
                """
                MATCH (n)
                DETACH DELETE n
                """
            )
            logger.info("Graph reset completed successfully")
        except Exception as e:
            logger.error(f"Failed to reset graph. Error: {str(e)}")



