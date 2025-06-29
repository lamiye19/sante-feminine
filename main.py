# app.py
import streamlit as st
from rag_utils import load_and_split_pdfs, create_or_load_vectorstore, build_qa_chain
import time
import os

# Création de l'arborescence et des fichiers
base_dir = "data"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


documents = load_and_split_pdfs(base_dir)
#st.write("documents")
vectors = create_or_load_vectorstore(documents)
#st.write("vectors")
qa_chain = build_qa_chain(vectors)
#st.write("")

st.title("ChatBot - Santé féminine")

query = st.text_input("Pose ta question")

if query:
    start = time.time()
    with st.spinner("Recherche..."):
        answer = qa_chain.invoke(query)
        duration = time.time() - start
    
        st.write("Réponse :", answer)
        st.markdown(f"⏱️ Temps de réponse : `{duration:.2f}` secondes")
