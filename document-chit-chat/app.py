import os 
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrival_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(model="gemma-7b-it", api_key=groq_api_key)

prompt=ChatPromptTemplate(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
question:{input}
"""
)

def create_vector_embedding():
    if  "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.textsplitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.texts=st.session_state.textsplitter.split_documents(
            st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents( st.session_state.texts, st.session_state.embeddings)
        




userprompt=st.text_input("Enter your queries")
if st.button("Create embedding"):
    create_vector_embedding()
    st.write("vectot database is ready")

import time 

if userprompt:
    document_chain=create_stuff_documents_chain(model,prompt=prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrival_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({"input":userprompt})
    print(f"response time is equal to {time.process_time()-start}")

    st.write(response['answer'])


    with st.expander("Document similarity serach"):
        for i , doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write('-------------')
         



    







      



















