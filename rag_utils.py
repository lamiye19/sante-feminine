import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_and_split_pdfs(pdf_dir):
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
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(index_path)
        return vectorstore


def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es un assistant intelligent. Tu réponds toujours en français.
Voici le contexte : {context}

Question : {question}
Réponse :
"""
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # ou mistralai/Mistral-7B-Instruct-v0.1 si tu veux + puissant
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

