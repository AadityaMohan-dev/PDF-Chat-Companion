import os
import pickle
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

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

load_dotenv()

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



if __name__ == '__main__':
    main()