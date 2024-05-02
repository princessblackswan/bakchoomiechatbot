import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading {file}")
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('only pdf, docx or txt')
    data = loader.load()
    return data

def chunk_data(data, chunk_size=300, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings_chroma(chunks):
    from langchain.vectorstores import chroma
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    vector_store = chroma.Chroma.from_documents(chunks, embeddings)
    return vector_store



#Cretae flexible prompt template -- pls feel free to use
def prompt(q, *contexts):
    from langchain.prompts import PromptTemplate
    context_dict = {f"context{i}": context for i, context in enumerate(contexts, start=1)}
    context_string = ", ".join(f"{{{key}}}" for key in context_dict)
    prompt_template = "Please answer precisely based on the context: " + context_string
    prompt_obj = PromptTemplate.from_template(prompt_template)
    prompt_final = prompt_obj.format(question=q, **context_dict)  
    return prompt_final

def ask_and_get_answer(vector_store, prompt, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature=0.5)
    vector_store = vector_store
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": k})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory = memory,
        retriever = retriever,
        return_source_documents = True
    )
    result = crc.invoke({'question': prompt})
    return result

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens/1000 * 0.0004
  

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)


    st.image('bcm.png')
    st.subheader('Lets chat with your documents')


    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key  # corrected syntax


        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change = clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change = clear_history)
    
        
        add_data = st.button('Add Data', on_click = clear_history)

        if uploaded_file and add_data:
            with st.spinner(text='Reading, chunking and embedding file...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size = chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                tokens, embeddings_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embeddings_cost:.4f}')

                vector_store =create_embeddings_chroma(chunks)
                
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully')

    q= st.text_input('Ask a question about content of your file:')
    
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('LLM Answer: ', value=answer['answer'])
            st.divider() 

            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q:{q} \nA: {answer["answer"]}' 
            st.session_state.history = f'{value} \n{"-"*100}\n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label="Chat history", value=h, key='history', height=400)




