from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the SAME embedding model used to build the store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing vector store from disk (no rebuild)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# Turn it into a retriever that returns the top 3 closest documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Try some questions
questions = [
    "What was Apple's revenue in 2025?",
    "How profitable was Tesla in 2024?",
    "Tell me about Microsoft's finances",
]

for q in questions:
    print(f"\n=== Question: {q} ===")
    results = retriever.invoke(q)
    for i, doc in enumerate(results):
        print(f"  [Match {i+1}] {doc.page_content}")