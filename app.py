import os
import pickle
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv

import requests

load_dotenv()

azure_endpoint = os.getenv('AZURE_ENDPOINT')
api_key = os.getenv('AZURE_API_KEY')

print('endpoint : ',azure_endpoint)
print('key : ', api_key)

def ask_question(question, documents):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    body = {
        'question': question,
        'documents': documents,
        'top_k': 3  
    }

    response = requests.post(azure_endpoint, headers=headers, json=body)
    return response.json()


with st.sidebar:
    st.title('😊💭 PDF Chat Companion')
    st.markdown("""
        :red[About] :  
            This App is an LLM-powerd chatbot built using :  
                -[Streamlit](https://streamlit.io/)  
                -[LangChain](https://python-langchain.com/)  
                -[OpenAI](https://platform.openai.com/docs/models) (LLM Models ) 
""")

    
    st.write("Made with ❤ by Aaditya Mohan")


def main():
    st.header("Chat with PDF 💭")

    #upload pdf
    pdf = st.file_uploader("Upload your file here",type='pdf')
    
    if pdf is not None:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages : 
            text += page.extract_text()
        
        # st.write(text)

        #CHUNKING
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text = text)
    
        st.write(chunks)

        #EMBEDDING
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl",'rb') as f:
                VectorStore = pickle.load(f)
        else :
             embeddings = OpenAIEmbeddings()
             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
             with open(f"{store_name}.pkl",'wb') as f:
                pickle.dump(VectorStore,f)

        #Accepts User questions/query

        query = st.text_input("Ask question about yur PDF file :")

        if(query):
            docs = VectorStore.similarity_search(query = query, k=3) #similarity searching ...

            llm = OpenAI(temprature=0,model_name='gpt-3.5-turbo' )
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb :
                response = chain.run(input_documents=docs, question = query)
                print(cb)
            st.write(response)

# def main():
#     st.header("Chat with PDF 💭")

#     # Upload pdf
#     pdf = st.file_uploader("Upload your file here", type='pdf')
    
#     if pdf is not None:
#         st.write(pdf.name)
#         pdf_reader = PdfReader(pdf)
        
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
        
#         # Chunking
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text=text)
    
#         st.write(chunks)
        
#         # Embedding 
#         embeddings = OpenAIEmbeddings()
#         VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#         store_name = pdf.name[:-4]
        
#         if os.path.exists(f"{store_name}.pkl"):
#             with open(f"{store_name}.pkl", 'rb') as f:
#                 VectorStore = pickle.load(f)
#         else:
#             embeddings = OpenAIEmbeddings()
#             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#             with open(f"{store_name}.pkl", 'wb') as f:
#                 pickle.dump(VectorStore, f)
        
#         # Accepts User questions/query
#         query = st.text_input("Ask question about your PDF file:")
        
#         if query:
#             docs = VectorStore.similarity_search(query=query, k=3)  # Similarity searching ...
            
#             # Call Azure ChatGPT API for question answering
#             api_response = ask_question(query, [doc['text'] for doc in docs])
            
#             # Process API response
#             if 'answers' in api_response:
#                 answers = api_response['answers']
#                 st.write(answers)
#             else:
#                 st.write("No answer found.")




if __name__ == '__main__':
    main()