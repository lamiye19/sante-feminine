# app.py
import streamlit as st
from rag_utils import load_and_split_pdfs, create_or_load_vectorstore, build_qa_chain
import os

# Création de l'arborescence et des fichiers
base_dir = "data"
#os.makedirs(base_dir, exist_ok=True)
#os.makedirs(f"{base_dir}", exist_ok=True)  # Dossier pour les PDF

documents = load_and_split_pdfs(base_dir)
vectors = create_or_load_vectorstore(documents)
qa_chain = build_qa_chain(vectors)
st.title("🦙 Chat PDF - RAG avec LLama")

query = st.text_input("Pose ta question sur les documents PDF")

if query:
    with st.spinner("Recherche..."):
        answer = qa_chain.run(query)
        st.write("💬 Réponse :", answer)
