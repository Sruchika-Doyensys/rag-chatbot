"""Command-line RAG chatbot over a company knowledge base.

Stack: LangChain + Groq (Llama 3.3 70B) + FAISS + local HuggingFace embeddings.
Commands: /clear = new conversation, /quit = exit.
"""

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
KB_DIR = Path(os.getenv("KB_DIR", "knowledge_base"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "faiss_index"))
FINGERPRINT_FILE = "kb_fingerprint.txt"

SUPPORTED = {".txt", ".md", ".pdf"}
TOP_K = 4
MAX_HISTORY_MESSAGES = 12  # last 6 exchanges
NOT_FOUND_MSG = "I'm sorry, I couldn't find that information in the company knowledge base."

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rewrite the question "
     "as a standalone question that can be understood without the history. "
     "Do NOT answer it. If it is already standalone, return it unchanged. "
     "Return only the question."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for a company knowledge base. "
     "Answer the question using ONLY the context below. "
     "If the context does not contain the answer, reply exactly with: "
     f"\"{NOT_FOUND_MSG}\" "
     "Do not use outside knowledge and do not make anything up. "
     "Be concise.\n\nContext:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


# --------------------------------------------------------------------------- #
# Knowledge base / index
# --------------------------------------------------------------------------- #
def list_kb_files():
    if not KB_DIR.is_dir():
        sys.exit(f"Knowledge base folder '{KB_DIR}' not found.")
    files = sorted(p for p in KB_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        sys.exit(f"No .txt, .md or .pdf files found in '{KB_DIR}'.")
    return files


def fingerprint(files):
    """Hash of file names + contents, so the index rebuilds when docs change."""
    h = hashlib.sha256()
    for p in files:
        h.update(str(p).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def load_documents(files):
    docs = []
    for p in files:
        try:
            if p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            else:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception as e:  # keep going if one file is bad
            print(f"Warning: could not load {p}: {e}")
    if not docs:
        sys.exit("No documents could be loaded.")
    return docs


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_or_load_index(embeddings):
    files = list_kb_files()
    fp = fingerprint(files)
    fp_path = INDEX_DIR / FINGERPRINT_FILE

    if (INDEX_DIR / "index.faiss").exists() and fp_path.exists() \
            and fp_path.read_text().strip() == fp:
        print("Loading saved index...")
        return FAISS.load_local(str(INDEX_DIR), embeddings,
                                allow_dangerous_deserialization=True)

    print("Building index from knowledge base (first run or documents changed)...")
    docs = load_documents(files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        sys.exit("Knowledge base documents are empty.")
    store = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    fp_path.write_text(fp)
    print(f"Indexed {len(chunks)} chunks from {len(files)} file(s).")
    return store


# --------------------------------------------------------------------------- #
# Chat logic
# --------------------------------------------------------------------------- #
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def answer_question(question, history, retriever, rewriter, answerer):
    recent = history[-MAX_HISTORY_MESSAGES:]
    standalone = question
    if recent:  # only rewrite follow-ups
        standalone = rewriter.invoke({"history": recent, "question": question}).strip() or question
    docs = retriever.invoke(standalone)
    return answerer.invoke({
        "context": format_docs(docs),
        "history": recent,
        "question": question,
    })


def main():
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Create a .env file with "
                 "GROQ_API_KEY=your_key (free key at https://console.groq.com).")

    print("Loading embedding model (first run downloads it)...")
    store = build_or_load_index(get_embeddings())
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGroq(model=CHAT_MODEL, temperature=0)
    parser = StrOutputParser()
    rewriter = REWRITE_PROMPT | llm | parser
    answerer = ANSWER_PROMPT | llm | parser

    print("\nCompany Knowledge Base Chatbot")
    print("Commands: /clear = new conversation, /quit = exit\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit"):
            print("Goodbye!")
            break
        if cmd == "/clear":
            history.clear()
            print("Chat history cleared. Starting a fresh conversation.\n")
            continue

        try:
            reply = answer_question(user_input, history, retriever, rewriter, answerer)
        except Exception as e:
            print(f"Bot: Sorry, something went wrong ({type(e).__name__}: {e})\n")
            continue

        print(f"Bot: {reply}\n")
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=reply))


if __name__ == "__main__":
    main()
