import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def load_and_split_pdfs(pdf_dir):
    """
    Charge tous les fichiers PDF depuis un dossier et les découpe en chunks.
    """
    all_docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(pdf_dir, file))
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)
    return all_docs

def create_or_load_vectorstore(documents, index_path="faiss_index"):
    """
    Crée ou recharge un index FAISS à partir de documents.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(index_path)
        return vectorstore

def build_qa_chain(vectorstore):
    """
    Crée une chaîne RAG avec Ollama (LLama) et un récupérateur FAISS.
    """
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3")  # Assure-toi d'avoir `ollama pull llama3`
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
                     