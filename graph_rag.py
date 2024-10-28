from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from ResponseStructure.entities import Entities
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from GraphMind_Logger import logger
from dotenv import load_dotenv
load_dotenv()


class GraphRAG:

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
        self.g_llm = ChatGroq(
            groq_api_key = os.getenv("GROQ_API_KEY"), 
            model_name= "llama-3.2-90b-text-preview" 
            )
        logger.info("LLM models initialized successfully")
        


    def create_entity_extracttion_chain(self):
        """
        Creates a chain which will extract entities from the question posed by the user. 
        This allows us to search the graph for nodes which correspond to entities more efficiently

        Returns:
            Runnable: Runnable chain which uses the LLM to extract entities from the users question
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert entity extractor. Your task is to identify and extract named entities from the input text.
                    - If an entity appears multiple times, include it only once
                    - If no valid entities are found, return an empty list

                    Examples:
                    Input: "What projects did John Smith work on at Microsoft in 2023?"
                    Output: ["John Smith", "Microsoft"]

                    Input: "Tell me about the partnership between Apple and Tesla"
                    Output: ["Apple", "Tesla"]"""
                ),
                (
                    "human",
                    "Use the given format to Extract and list all entities from this text: {question} "
                ),
            ]
        )

        try:
            entity_extraction_chain = prompt | self.g_llm.with_structured_output(Entities)
            logger.info("Entity extraction chain created successfully")
            return entity_extraction_chain
        except Exception as e:
            logger.error(f"Failed to create entity extraction chain: {str(e)}")
            raise

    
    def full_text_query_generator(self, input_query) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspellings.

        Args:
            input_query (str): The extracted entity name pulled from the users question

        Returns:
            str: _description_
        """

        full_text_query = ""
        try:
            words = [word for word in remove_lucene_chars(input_query).split() if word]
            for word in words[:-1]:
                full_text_query += f" {word}~2 AND"
            full_text_query += f" {words[-1]}~2"
            logger.info(f"Generated full-text query: {full_text_query}")
            return full_text_query.strip()
        
        except Exception as e:
            logger.error(f"Error generating full-text query: {str(e)}")
            raise   


    def structured_retriever(self, question: str) -> str:
        """
        Creates a retriever which will use entities extracted from the users query to 
        request context from the Graph and return the neighboring nodes and edges related
        to that query. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The fully formed Graph Query which will retrieve the 
                 context relevant to the users question
        """
        result = ""
        try:
            entity_extraction_chain = self.create_entity_extracttion_chain()
            entities = entity_extraction_chain.invoke({"question": question})
            logger.info(f"Extracted entities: {entities.names}")

            for entity in entities.names:
                logger.info(f"Querying graph for entity: {entity}")
                response = self.graph.query(
                    """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                    YIELD node,score
                    CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                    }
                    RETURN output LIMIT 50
                    """,
                    {"query": self.full_text_query_generator(entity)},
                )
                result += "\n".join([x['output'] for x in response])
                logger.info(f"Retrieved relationships for entity: {entity}")

            logger.info("Structured retrieval completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in structured retrieval: {str(e)}")
            raise
    
    def create_vector_index(self) -> Neo4jVector:
        """
        Uses the existing graph to create a vector index. This vector representation
        is based off the properties specified. Using OpenAIEmbeddings since we are using 
        GPT-4o as the model. 

        Returns:
            Neo4jVector: The vector representation of the graph nodes specified in the configuration
        """

        try:
            vector_index = Neo4jVector.from_existing_graph(
                embedding=GoogleGenerativeAIEmbeddings(
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    model="models/embedding-001"
                ),
                search_type="hybrid",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding"
            )
            logger.info("Vector index created successfully")
            return vector_index
        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            raise
    
    def retriever(self, question):
        """
        The graph RAG retriever which combines both structured and unstructured methods of retrieval 
        into a single retriever based off the users question. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The retrieved data from the graph in both forms
        """
        logger.info(f"Starting combined retrieval for question: {question}")

        try:
            vector_index = self.create_vector_index()
            logger.info("Performing vector similarity search")
            unstructured_data = [chunk.page_content for chunk in vector_index.similarity_search(question)]
            
            logger.info("Performing structured retrieval")
            structured_data = self.structured_retriever(question)
            
            final_data = f"""Structured data:
                {structured_data}
                Unstructured data:
                {"#Document ". join(unstructured_data)}
            """
            logger.info("Combined retrieval completed successfully")
            return final_data
        except Exception as e:
            logger.error(f"Error in combined retrieval: {str(e)}")
            raise
    
    def create_context_aware_search_query(self, chat_history: List, question: str) -> str:
        """
        Combines chat history along with the current question into a prompt that 
        can be executed by the LLM to answer the new question with history.

        Args:
            chat_history (List): List of messages captured during this conversation
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The formatted prompt that can be sent to the LLM with question & chat history
        """ 

        try:
            search_query = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """Given the following conversation and a follow up question, You are a context-aware query reformulation expert. 
                    Your task is to create a self-contained, 
                searchable question that incorporates relevant context from the chat history.

                    Chat History: "Tell me about Microsoft's cloud services"
                    Follow-up: "When did they launch it?"
                    Reformulation: "When did Microsoft launch their cloud services?"

                    Chat History: "Who is the CEO of Apple?"
                    Follow-up: "What about his previous company?"
                    Reformulation: "What was Tim Cook's previous company before Apple?"

                    Chat History: {chat_history}
                    Follow Up Input: {question}
                    Reformulated question:"""
                )
            ])
            formatted_query = search_query.format(
                chat_history=chat_history,
                question=question
            )
            logger.info(f"Generated context-aware query: {formatted_query}")
            return formatted_query
        except Exception as e:
            logger.error(f"Error creating context-aware search query: {str(e)}")
            raise
 