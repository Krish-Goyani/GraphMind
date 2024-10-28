import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from main import populate_graph, reset_graph, get_response

# Set page configuration
st.set_page_config(
    page_title="GraphMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def landing_page():
    """
    Primary entry point for the app. Creates the chat interface that interacts with the LLM. 
    """
    # Main title and description
    st.title("GraphMind 🧠")
    st.subheader("Your Intelligent Knowledge Graph Assistant")

    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="👋 Welcome to GraphMind! I'm your knowledge graph assistant. How can I help you today?")
        ]

    with st.sidebar:
        st.title("Graph Management 🛠️")
        st.divider()
        urls = st.text_input("Enter Youtube URL to populate graph (comma separated if you have multiple)", 
                      placeholder="https://www.youtube.com/watch?v=xyzxyzxyz")
        urls = urls.split(",")
        urls = [url.strip() for url in urls]
        

        # Create two columns for the buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Populate Graph 🔄", type="primary"):
                with st.spinner("Building knowledge graph..."):
                    populate_graph(urls=urls)
                st.success("✅ Graph populated successfully!")
        
        with col2:
            if st.button("Reset Graph 🗑️", type="secondary"):
                reset_graph()
                st.success("Graph successfully reset!")


    # Main chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="🧑‍🎓"):
                    st.write(message.content)
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🧠"):
                    st.write(message.content)

    user_query = st.chat_input("Ask a question....")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
         # Add the current chat to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.rerun()



if __name__ == "__main__":
    landing_page()

    




