\# GFC Financial Chatbot — RAG over 10-K Data



A retrieval-augmented generation (RAG) chatbot that answers natural-language

questions about the financial performance of Microsoft, Tesla, and Apple

(fiscal years 2023–2025), built entirely with free, local, open-source tools.



\## What it does

Ask plain-English questions like:

\- "What was Apple's revenue in 2025?"

\- "How did Tesla's net income change from 2023 to 2024?"

\- "Which company had the highest revenue in 2025?"



The chatbot retrieves the relevant financial facts and uses a local LLM to

generate a grounded, conversational answer — including simple reasoning such as

computing year-over-year changes and comparing companies.



\## How it works (RAG pipeline)

1\. \*\*Data\*\* — financial figures extracted from 10-K filings (CSV).

2\. \*\*Document creation\*\* — each company-year is converted into a natural-language sentence.

3\. \*\*Embeddings\*\* — sentences are embedded locally with `sentence-transformers` (all-MiniLM-L6-v2).

4\. \*\*Vector store\*\* — embeddings are stored in a local Chroma database for semantic search.

5\. \*\*Retrieval\*\* — a question is embedded and the most relevant documents are retrieved.

6\. \*\*Generation\*\* — retrieved context + the question are passed to a local LLM (Llama 3.2 via Ollama),

&#x20;  which writes a grounded answer.

7\. \*\*Interface\*\* — a Streamlit chat UI.



\## Tech stack

\- \*\*LangChain\*\* — RAG orchestration

\- \*\*Ollama\*\* (Llama 3.2 3B) — local LLM, no API costs

\- \*\*sentence-transformers\*\* — local embeddings

\- \*\*ChromaDB\*\* — local vector store

\- \*\*Streamlit\*\* — chat interface

\- \*\*pandas\*\* — data handling



\## Running it locally

1\. Install \[Ollama](https://ollama.com) and pull the model:

