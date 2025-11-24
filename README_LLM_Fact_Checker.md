# 🧠 LLM Fact Checker

A **Fact-Checking System** powered by **LangChain**, **FAISS**, and **OpenAI’s LLMs**, designed to verify the truthfulness of claims using retrieved evidence from a dataset.  
This project demonstrates how to use Retrieval-Augmented Generation (RAG) for factual consistency.

---

## 🚀 Features

- 🧩 **Data Loading** from CSV using `UnstructuredCSVLoader`  
- ✂️ **Text Splitting** with `RecursiveCharacterTextSplitter`  
- 🧠 **Vector Embedding** using `text-embedding-3-large` from OpenAI  
- 📚 **Vector Store** with FAISS for efficient retrieval  
- 🤖 **LLM Fact Checking** using a deterministic system prompt  
- 🔍 **Query-Based Evidence Retrieval** for factual evaluation  

---

## 🏗️ Project Structure

```
LLM_fact_checker.ipynb
sample_facts.csv
README.md
```

---

## ⚙️ Installation

This notebook is designed for **Google Colab**, but it can also be run locally.

### 1. Clone or upload the notebook
```bash
git clone <repo_url>
cd LLM_fact_checker
```

### 2. Install dependencies
```bash
!pip install langchain-community unstructured langchain-openai openai faiss-cpu
```

### 3. Set up OpenAI API key
In Colab:
```python
from google.colab import userdata
openai_api_key = userdata.get('openai_api_key')
```

Or manually:
```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

---

## 📚 Workflow Overview

### 1. Load Dataset
```python
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
loader = UnstructuredCSVLoader(file_path="sample_facts.csv", mode="elements")
docs = loader.load()
```

### 2. Split Text into Chunks
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

### 3. Create Embeddings and Store in FAISS
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
df_faiss = FAISS.from_documents(chunks, embeddings)
```

### 4. Perform Retrieval
```python
query = "The Indian government has announced free electricity to all farmers starting July 2025"
docs_faiss = df_faiss.similarity_search_with_score(query, k=5)
```

### 5. LLM Fact Checking Prompt
```python
prompt_template = '''
You are a deterministic fact-checking assistant.
You must evaluate claims ONLY using the evidence provided by the retrieval system.

Claim: {query}

Evidence: {context_text}

Follow these strict rules:
1. Use ONLY the provided evidence. No external knowledge.
2. If the evidence part is empty, answer: "Insufficient evidence to verify the claim."
3. Output one of: True, False, or Insufficient Evidence.
4. Provide reasoning in 1-2 sentences.
'''
```

---

## 🧩 Technologies Used

- LangChain
- FAISS (Facebook AI Similarity Search)
- OpenAI API
- Google Colab
- Python 3.10+

---

## 📈 Example Use Case

| Claim | Result | Explanation |
|-------|---------|-------------|
| "The Indian government has announced free electricity to all farmers starting July 2025." | ✅ True / ❌ False / ⚠️ Insufficient Evidence | Based on retrieved facts from dataset. |

---

## 🧰 Future Improvements

- Integrate CrewAI agents for multi-step reasoning  
- Add RAG-based retrievers (e.g., using LangGraph)  
- Deploy as a Streamlit app for interactive use  
- Connect to a live factual dataset API  

---

