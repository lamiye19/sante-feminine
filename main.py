# app.py
import streamlit as st
from rag_utils import load_and_split_pdfs, create_or_load_vectorstore, build_qa_chain
import time

# Création de l'arborescence et des fichiers
base_dir = "data"
#os.makedirs(base_dir, exist_ok=True)
#os.makedirs(f"{base_dir}", exist_ok=True)  # Dossier pour les PDF

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
