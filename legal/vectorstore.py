import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st

def setup_vectorstore(user_specific=False, vector_db_dir=None, user_id=None):
    try:
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS(embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.session_state.error = f"Error setting up vector store: {str(e)}"
        return None

def vectorize_data(data_dir, vector_db_dir, user_id, user_files=None):
    try:
        st.session_state.processing = True
        user_data_dir = os.path.join(data_dir, user_id)
        os.makedirs(user_data_dir, exist_ok=True)
        if user_files:
            for uploaded_file in user_files:
                file_bytes = uploaded_file.read()
                file_path = os.path.join(user_data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
        loader = DirectoryLoader(
            path=user_data_dir,
            glob="**/*.pdf",
            loader_cls=UnstructuredFileLoader
        )
        documents = loader.load()
        if not documents:
            st.session_state.processing = False
            return False, "No documents found or could not be processed."
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        vectordb = FAISS.from_documents(
            documents=text_chunks,
            embedding=embeddings
        )
        st.session_state.processing = False
        st.session_state.documents_vectorized = True
        return True, "Documents successfully vectorized!"
    except Exception as e:
        st.session_state.processing = False
        return False, f"Error vectorizing documents: {str(e)}"
