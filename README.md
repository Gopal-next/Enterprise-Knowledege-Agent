# Enterprise Knowledge Agent

## Overview

Enterprise Knowledge Agent is an AI-powered assistant that answers questions using both company documents and structured database information. Users can ask questions in natural language, and the system automatically retrieves relevant information before generating a response.

---

## Features

- Chat with company documents (PDFs)
- Natural language to SQL querying
- Retrieval-Augmented Generation (RAG)
- Source citations for answers
- Multi-document support
- AI-powered question answering
- Streamlit-based user interface

---

## Technologies Used

- Python
- LangChain
- ChromaDB
- SQLite / PostgreSQL
- Streamlit
- Gemini / OpenAI
- Pandas

---

## Project Structure

```text
enterprise-knowledge-agent/
│
├── app/
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── rag/
│   ├── database/
│   ├── agents/
│   └── services/
│
├── data/
│   ├── pdfs/
│
├── tests/
│
├── requirements.txt
├── .env
└── README.md
```


## How It Works

```text
User Question
      ↓
AI Agent
      ↓
PDF Retriever / SQL Tool
      ↓
Relevant Information
      ↓
LLM
      ↓
Final Answer
```

The agent analyzes the user's question and automatically decides whether to retrieve information from documents, databases, or both before generating a response.


## How to Run Locally

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd enterprise-knowledge-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

### 6. Run the Application

```bash
streamlit run app.py
```

### 7. Open in Browser

```text
http://localhost:8501
```

### Note

Before asking questions, upload sample documents such as:

- Employee Handbook.pdf
- Company Leave Policy.pdf

---

## Screenshots

Add screenshots here after implementation:

### Chat Assistant

![Chat Assistant](Images\chat.png)

### Analytics Dashboard

![Analytics Dashboard](Images\analytics.png)

---


## Author

**Gopal Kumar**