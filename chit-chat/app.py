import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Qz&A Chatbot with OPENAI"
groqapi_key=os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

#prompt template 


prompt=ChatPromptTemplate(
    [
         ("system","you are a helpful assistant .Please respons to user questions"),

         ("user","question:{question}")
    ]
)

outputparser=StrOutputParser()

model= ChatGroq(model="gemma2-9b-it",api_key=groqapi_key)


chain=prompt| model|outputparser


response=chain.invoke({"question":"waht is capital of india"})

response


def generate_response(question,api_key,temperature,max_tokens):
    # openai.api_key=api_key
    # llm=ChatOpenAI(model=llm)
    outputpraser=StrOutputParser()
    chain=prompt|model|outputparser
    answer=chain.invoke({"question":question})
    return answer

st.title("enhanced q& A with Chatbot")
api_key=st.sidebar.text_input("enter your api password", type="password")
llm=st.sidebar.selectbox(label="Select an open AI Model",options=["gpt-4o","gpt-4-turbo","gpt-4"])

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Token",min_value=50,max_value=300,value=150)
st.write("Go Ahead and ask any questions")
userinput=st.text_input("You:")

if userinput:
    response=generate_response(userinput,api_key,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide the query")
