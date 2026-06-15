import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="GFC Financial Chatbot", page_icon="📊")

# --- Build the RAG chain once and cache it (so it doesn't reload each message) ---
@st.cache_resource
def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3.2:3b")

    prompt = ChatPromptTemplate.from_template(
        """You are a financial assistant for GFC. Answer the user's question using ONLY the
financial data provided in the context below. The figures are for Microsoft, Tesla, and
Apple across fiscal years 2023-2025.

If the answer isn't in the context, say you don't have that information. Use clear,
plain language. Do not make up numbers. Do not give investment advice.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

rag_chain = load_rag_chain()

# --- Page header ---
st.title("📊 GFC Financial Chatbot")
st.caption("Ask about Microsoft, Tesla, and Apple — fiscal years 2023 to 2025.")

# --- Keep chat history in session ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if question := st.chat_input("Ask a financial question..."):
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})