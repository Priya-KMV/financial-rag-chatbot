from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b")

response = llm.invoke("In one sentence, what is a 10-K filing?")
print(response)