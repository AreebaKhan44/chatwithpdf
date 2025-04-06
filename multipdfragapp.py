



import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Load environment variables
load_dotenv()

# Ensure Spacy model is installed
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


def pdf_read(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def get_chunks(text):
    """Split text into smaller chunks for better vector search."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


def vector_store(text_chunks):
    """Create and save FAISS index for efficient retrieval."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(tools, question):
    """Retrieve information using OpenAI Chat model."""
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"), verbose=True)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant. Answer the question as detailed as possible from the provided context. 
            If the answer is not in the provided context, say: 'Answer is not available in the context' and do not provide a wrong answer."""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [tools], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tools], verbose=True)
    
    response = agent_executor.invoke({"input": question})
    st.write("Reply:", response['output'])


def user_input(user_question, pdf_docs):
    """Handle user queries by ensuring FAISS index exists and retrieving relevant data."""
    
    faiss_path = "faiss_db/index.faiss"

    # ✅ Check if FAISS index exists
    if not os.path.exists(faiss_path):
        if not pdf_docs:
            st.error("No FAISS index found! Please upload PDFs and process them first.")
            return

        st.warning("No FAISS index found! Processing PDFs now...")
        raw_text = pdf_read(pdf_docs)
        text_chunks = get_chunks(raw_text)
        vector_store(text_chunks)
        st.success("FAISS index created! You can now ask questions.")

    # ✅ Load FAISS index
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "Retrieves answers from PDFs.")
    
    get_conversational_chain(retrieval_chain, user_question)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config("Chat with PDF")
    st.header("RAG-based Chat with PDF")

    pdf_docs = st.sidebar.file_uploader("Upload PDFs and click 'Submit & Process'", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = pdf_read(pdf_docs)
            text_chunks = get_chunks(raw_text)
            vector_store(text_chunks)
            st.success("Processing complete! Now you can ask questions.")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question, pdf_docs)


if __name__ == "__main__":
    main()
