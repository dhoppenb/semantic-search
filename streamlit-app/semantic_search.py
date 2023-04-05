import streamlit as st
import pandas as pd
import openai
import numpy as np
import tiktoken

#initialize a state variable if it does not exist
if 'documents' not in st.session_state:
    st.session_state.documents = []

if 'embedding_cost' not in st.session_state:
    st.session_state.embedding_cost = 0

if 'completion_cost' not in st.session_state:
    st.session_state.completion_cost = 0

#Openai API key and organization
openai.api_key = st.secrets["api_key"]
openai.organization = st.secrets["organization"]

#default models
COMPLETION_MODEL = 'text-davinci-003'
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CHUNK_SIZE = 8000

#initialize tiktoken for encoding
encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)

st.title("Semantic search demo") 
st.write(f"Cost accumulated for embeddings: {round(st.session_state.embedding_cost,4)} USD")
st.write(f"Cost accumulated for completions: {round(st.session_state.completion_cost,4)} USD")


#Function that takes a string and retrieves an embedding using the OpenAI API and ada model
def get_embedding(text):   
    add_embedding_cost(text)
    embedding = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)["data"][0]["embedding"]
    return embedding

def get_token_length(text):
    return len(encoding.encode(text))

#Add embedding cost to state var
def add_embedding_cost(text):
    num_tokens = get_token_length(text)
    #currently ada costs 0.0004 per 1k tokens
    price = (0.0004 / 1000) * num_tokens
    st.session_state.embedding_cost += price

#Add completion cost to state var
def add_completion_cost(text):
    num_tokens = get_token_length(text)
    #currently davinci costs 0.0004 per 1k tokens
    price = (0.02 / 1000) * num_tokens
    st.session_state.completion_cost += price

#Function that takes a string, retrieves an embedding and adds both to the documents variable as dictionary
def add_document_with_embedding(doc):
    embedding = get_embedding(doc)
    num_tokens = get_token_length(doc)
    st.session_state.documents.append({"document": doc, "#tokens": num_tokens,"embedding": embedding})

#Cache completion calls, they are somewhat expensive and would be called with every code change otherwise
#only add completion cost if the function is called
@st.cache_data
def completion(text):
    add_completion_cost(text)
    response = openai.Completion.create(
        model=COMPLETION_MODEL,
        prompt= prompt,
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )
    answer = response["choices"][0]["text"]
    return answer

#numpy fucntion that calculates the cosine similarity between two vectors
def cosine_similarity(a, b):    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


st.header("Document database")
with st.form(key='upload_form', clear_on_submit=True):
    uploaded = st.file_uploader("Upload a file", type=["txt"], accept_multiple_files=True)
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        for file in uploaded:
            #read the file content
            file_content = file.getvalue().decode("utf-8")
            add_document_with_embedding(file_content)

st.write(pd.DataFrame(st.session_state.documents))

st.header("Question")
question = st.text_input("Ask a question about the documents")
if question:
    question_embedding = get_embedding(question)
    num_tokens = get_token_length(question)
    st.write(pd.DataFrame([(question, num_tokens, question_embedding)], columns=["Question", "#tokens", "Embedding"]))

st.header("Question and document matching")
if question:
    #calculate the cosine similarity between the question and each document
    similarities = []
    for doc in st.session_state.documents:
        similarities.append(cosine_similarity(question_embedding, doc["embedding"]))
    #sort the documents by similarity
    sorted_documents = [x for _,x in sorted(zip(similarities, st.session_state.documents), reverse=True)]
    #write the results
    st.write(pd.DataFrame(sorted_documents))

st.header("Ask OpenAI to answer the question using the best matching document")
if question:
    #get the best matching document
    best_document = sorted_documents[0]
    #ask OpenAI to answer the question using the best matching document
    prompt = f"Content:\n{best_document['document']} \nPlease answer the question below using only the above content:\n{question}\nAnswer:"
    add_completion_cost(prompt)
    answer = completion(prompt)
    st.text(prompt)
    st.write(answer)