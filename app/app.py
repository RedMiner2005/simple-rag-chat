#!/usr/bin/env python
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import SitemapLoader, WebBaseLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langserve import add_routes


def refresh_vectorstore():
    global vectorstore
    print("Saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        persist_directory="chroma_data/",
        documents=all_splits,
        embedding=embeddings
        )
    vectorstore.persist()
    print(f"Loaded {len(all_splits)} documents")

def load_website():
    global all_splits
    webloader = WebBaseLoader(loader_url)
    # Use the sitemap loader to load the entire website. It takes time though: 
    # webloader = SitemapLoader(web_path=f"{loader_url}/sitemap.xml")
    webdata = webloader.load()
    all_splits = text_splitter.split_documents(webdata)
    print(f"Website split into {len(all_splits)} chunks")

def load_file(filename: str):
    global all_splits
    loader = PyPDFLoader(filename)
    curr = loader.load()
    all_splits.extend(curr)
    print(f"{filename} split into {len(curr)} chunks")

def load_all_files():
    global all_splits
    for file in os.listdir("data/"):
        if file.endswith('.pdf'):
            pdf_path = os.path.join("data/", file)
            load_file(pdf_path)
    print("Completed splitting PDFs")


app = FastAPI(
    title="Headfitted Chatbot API",
    version="1.0",
    description="A simple api for the Headfitted Chatbot",
)

if "ISINDOCKER" not in os.environ:
    ollama_url = "localhost"
else:
    ollama_url = "ollama-container"

loader_url = os.environ.get("URL", "https://example.com")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Ollama(
    model="llama3",
    base_url=f"http://{ollama_url}:11434",
    # verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=f"http://{ollama_url}:11434",)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = []
vectorstore = None

load_website()
load_all_files()
refresh_vectorstore()

# System Message
system_message = """You are a customer support agent for Headfitted Solutions. Your role is to assist customers with inquiries related to products, services, or information found on the company website.
Your knowledge base is comprised of information scraped from the company website, along with uploaded PDFs, if any.
If you don't know the answer, say so. If a question is outside the scope of customer support, politely inform the user.
Maintain a professional, helpful, and friendly tone."""

prompt_template = PromptTemplate(
    template=system_message + "\nContext: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

interface_template = PromptTemplate(
    template="{question}",
    input_variables=["question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

add_routes(
    app,
    interface_template|qa_chain,
    path="/chat",
)

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        assert file.filename.endswith(".pdf")
        global vectorstore
        contents = file.file.read()
        file_path = os.path.join("data", file.filename)
        with open(file_path, 'wb') as f:
            f.write(contents)
        loader = PyPDFLoader(file_path)
        curr = loader.load()
        all_splits.extend(curr)
        print(f"{file} split into {len(curr)} chunks")
        vectorstore = Chroma.from_documents(
            persist_directory="chroma_data/",
            documents=all_splits,
            embedding=embeddings
        )
        vectorstore.persist()
    except AssertionError:
        return {"message": "Upload pdf only"}
    except Exception as e:
        return {"message": "There was an error uploading the file: " + str(e)}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded data/{file.filename}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)