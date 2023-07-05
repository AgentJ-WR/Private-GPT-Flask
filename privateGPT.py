#!/usr/bin/env python3
from flask import Flask, request, jsonify, after_this_request
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import time

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

app = Flask(__name__)

# Initialize the necessary components globally, so they aren't re-initialized with every request
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, backend='gptj', n_batch=model_n_batch, verbose=False)
    case _default:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

@app.route('/')
def answer_query():
    query = request.args.get('query')

    @after_this_request
    def add_header(response):
        response.headers['Connection'] = 'Keep-Alive'
        response.headers['Keep-Alive'] = 'timeout=600'
        return response

    start = time.time()
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    end = time.time()

    result = {
        "question": query,
        "answer": answer,
        "time_taken": round(end - start, 2),
        "source_documents": [{**doc.metadata, "content": doc.page_content} for doc in docs],
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
