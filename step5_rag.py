from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Load retriever (from the saved vector store) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Load the local LLM ---
llm = OllamaLLM(model="llama3.2:3b")

# --- The prompt: tells the model to answer ONLY from the context ---
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

# --- Helper to format retrieved docs into one text block ---
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# --- The RAG chain: retrieve -> build prompt -> LLM -> text ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Try it ---
questions = [
    "What was Apple's revenue in 2025?",
    "How did Tesla's net income change from 2023 to 2024?",
    "Which company had the highest revenue in 2025?",
]

for q in questions:
    print(f"\n=== Question: {q} ===")
    answer = rag_chain.invoke(q)
    print(answer)