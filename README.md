# RAG Chatbot

A simple command-line chatbot that answers questions from a company knowledge base using **Retrieval-Augmented Generation (RAG)**. It's built with **LangChain**, **Groq** (Llama 3.3 70B), **FAISS**, and local **HuggingFace** embeddings.

## Features

- **Knowledge-base answers:** responds using only your company documents (`.txt`, `.md`, `.pdf`).
- **Conversation memory:** follow-up questions work (e.g. "Can I carry them forward?").
- **Honest "not found":** says so when the answer isn't in the knowledge base instead of making something up.
- **`/clear`:** resets the chat history and starts a fresh conversation.
- **`/quit`:** exits the chatbot.

## How it works

1. Documents in `knowledge_base/` are split into chunks and embedded locally with `all-MiniLM-L6-v2`.
2. The embeddings are stored in a FAISS index, which is saved to `faiss_index/` for reuse.
3. For each question, the chat history is used to rewrite follow-ups into a standalone question.
4. The 4 most relevant chunks are retrieved and passed to the Groq LLM, which answers only from that context.

## Project structure

```
rag-chatbot/
├── rag_chatbot.py          # Main chatbot script
├── requirements.txt        # Python dependencies
├── knowledge_base/         # Your company documents
│   └── company_policies.txt
├── .env                    # API keys (not committed)
└── .gitignore
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Sruchika-Doyensys/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> Note: `sentence-transformers` installs PyTorch (~2 GB). If you're short on disk space, use GitHub Codespaces (see below).

### 3. Add your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here

# Optional settings
CHAT_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
KB_DIR=knowledge_base
INDEX_DIR=faiss_index
```
You can get a free API key at [console.groq.com](https://console.groq.com).

### 4. Run
```bash
python rag_chatbot.py
```

## Running in GitHub Codespaces

1. On the repo page, click **Code → Codespaces → Create codespace on main**.
2. In the terminal:
   ```bash
   echo "GROQ_API_KEY=your_key_here" > .env
   pip install -r requirements.txt
   python rag_chatbot.py
   ```

## Example session

```
Company Knowledge Base Chatbot
Commands: /clear = new conversation, /quit = exit

You: How many annual leave days do I get?
Bot: You get 18 days of paid annual leave per year, plus 12 casual/sick leave days.

You: Can I carry them forward?
Bot: Yes, unused annual leave of up to 10 days can be carried forward to the next year.

You: Who is the CEO?
Bot: I'm sorry, I couldn't find that information in the company knowledge base.

You: /clear
Chat history cleared. Starting a fresh conversation.

You: /quit
Goodbye!
```

## Using your own documents

Add `.txt`, `.md`, or `.pdf` files to `knowledge_base/`, then **delete the `faiss_index/` folder** so the index is rebuilt on the next run.

## Tech stack

| Component    | Tool                                   |
|--------------|----------------------------------------|
| Framework    | LangChain                              |
| LLM          | Groq – `llama-3.3-70b-versatile`       |
| Embeddings   | HuggingFace – `all-MiniLM-L6-v2` (local) |
| Vector store | FAISS                                  |