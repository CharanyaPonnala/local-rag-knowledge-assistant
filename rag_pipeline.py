from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


VECTOR_DB_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"


def load_document(file_path: str):
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError("Only PDF and TXT files are supported.")

    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vectorstore(chunks):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    vectorstore.persist()
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )


def retrieve_context(question: str, k: int = 4):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(question)


def generate_answer(question: str, docs):
    context = "\n\n".join(
        [f"Source {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I do not know based on the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    llm = Ollama(model=LLM_MODEL)
    return llm.invoke(prompt)