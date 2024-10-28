from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from entities import Entities
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
class GraphRAG:

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
                    "You are extracting objects, person, organization, " +
                    "or business entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )

        entity_extraction_chain = prompt  | self.llm.with_structured_output(Entities)
        return entity_extraction_chain
    
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

        words = [word for word in remove_lucene_chars(input_query).split() if word]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
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
        entity_extraction_chain = self.create_entity_extracttion_chain()

        entities = entity_extraction_chain.invoke({"question": question})

        for entity in entities.names:
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
        return result
    
    def create_vector_index(self) -> Neo4jVector:
        """
        Uses the existing graph to create a vector index. This vector representation
        is based off the properties specified. Using OpenAIEmbeddings since we are using 
        GPT-4o as the model. 

        Returns:
            Neo4jVector: The vector representation of the graph nodes specified in the configuration
        """

        vector_index = Neo4jVector.from_existing_graph(
            embedding= GoogleGenerativeAIEmbeddings(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model="models/embedding-001"
                ),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index
    
    def retriever(self, question):
        """
        The graph RAG retriever which combines both structured and unstructured methods of retrieval 
        into a single retriever based off the users question. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The retrieved data from the graph in both forms
        """

        print(f"Search query : {question}")
        vector_index = self.create_vector_index()
        unstructured_data = [chunk.page_content for chunk in vector_index.similarity_search(question)]
        structured_data = self.structured_retriever(question)
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data
    
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

        search_query = ChatPromptTemplate.from_messages([
            (
                "system",
                """Given the following conversation and a follow up question, rephrase the follow 
                up question to be a standalone question, in its original language.
                Chat History:
                {chat_history}
                Follow Up Input: {question}
                Standalone question:"""
            )
        ])
        formatted_query = search_query.format(
            chat_history = chat_history,
            question = question
        )
        return formatted_query

 