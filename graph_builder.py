import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader, TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import GoogleGenerativeAI
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

        self.llm = GoogleGenerativeAI(
            model = "gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
    def chunker(self, raw_docs):
        """
        Accepts raw text context extracted from source and applies a chunking 
        algorithm to it. 

        Args:
            raw_docs (str): The raw content extracted from the source

        Returns:
            List: List of document chunks
        """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        docs = text_splitter.split_documents(raw_docs)
        return docs

    def graph_transformer(self, text_chunks):
        """
        Uses experimental LLMGraphTransformer to convert unstructured text into a knowledge graph

        Args:
            text_chunks (List): List of document chunks
        """
        llm_transformer = LLMGraphTransformer(llm=self.llm)

        graph_docs = llm_transformer.convert_to_graph_documents(text_chunks)
        self.graph.add_graph_documents(
            graph_docs,
            baseEntityLabel=True,
            include_source=True
        )

    def chunk_and_graph(self, raw_docs):
        """
        Breaks the raw text into chunks and converts into a knowledge graph

        Args:
            raw_docs (str): The raw content extracted from the source
        """
        text_chunks = self.chunker(raw_docs)
        if text_chunks is not None:
            self.graph_transformer(text_chunks)


    def extract_youtube_transcripts(self, urls):
        """
        Extracts multiple URLs YouTube transcripts

        Args:
            urls (List): A list of YouTube urls
        """
        for url in urls:
            transcript = YoutubeLoader.from_youtube_url(url).load()
            self.chunk_and_graph(transcript)


    def graph_index_generator(self):
        """
        Creates an index on the populated graph tp assist with efficient searches
        """
        self.graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )

    def reset_graph(self):
        """
        WARNING: Will clear entire graph, use with caution
        """
        self.graph.query(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )





