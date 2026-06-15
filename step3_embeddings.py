import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Load and clean the data (same as Step 2) ---
df = pd.read_csv("data/Financial_Data.csv")
num_cols = ['Total Revenue', 'Net Income', 'Total Assets',
            'Total Liabilities', 'Operating Cash Flow']
for c in num_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')

def row_to_text(row):
    return (
        f"In fiscal year {int(row['Fiscal Year'])}, {row['Company']} reported "
        f"total revenue of ${row['Total Revenue']/1000:,.1f} billion, "
        f"net income of ${row['Net Income']/1000:,.1f} billion, "
        f"total assets of ${row['Total Assets']/1000:,.1f} billion, "
        f"total liabilities of ${row['Total Liabilities']/1000:,.1f} billion, "
        f"and operating cash flow of ${row['Operating Cash Flow']/1000:,.1f} billion."
    )

# Wrap each sentence as a LangChain Document (with metadata for reference)
documents = [
    Document(
        page_content=row_to_text(row),
        metadata={"company": row["Company"], "year": int(row["Fiscal Year"])},
    )
    for _, row in df.iterrows()
]
print(f"Prepared {len(documents)} documents.")

# --- Create embeddings locally (free) ---
print("Loading the embedding model (first run downloads it, ~90 MB)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Build the vector store and save it to disk ---
print("Building the vector store...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db",   # saved here so we don't rebuild every time
)

print(f"Vector store built and saved. It contains {vectorstore._collection.count()} documents.")