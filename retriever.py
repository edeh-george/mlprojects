import os

import docx
import pdfplumber
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600000"
huggingface_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
current_dir = os.path.dirname(os.path.abspath(__file__))
book_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(book_dir):
    raise FileNotFoundError(
        f"The directory {book_dir} does not exist. Please check the path."
    )


def convert_docx_to_text(file_path):
    doc = docx.Document(file_path)
    txt_path = file_path.replace(".docx", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for paragraph in doc.paragraphs:
            f.write(paragraph.text + "\n")


def extract_text_from_pdf(file_path):
    text_chunks = []
    pdf_path = file_path.replace(".pdf", ".txt")
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    with open(pdf_path, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")


def process_documents(book_dir):
    for root, _, files in os.walk(book_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(".docx"):
                print(f"Converting {file} to text")
                convert_docx_to_text(file_path)
            elif file.lower().endswith(".pdf"):
                print(f"Extracting text from {file}")
                extract_text_from_pdf(file_path)


loader = TextLoader(
    os.path.join(book_dir, "black_parrot.txt"), autodetect_encoding=True
)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(documents)


def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        os.makedirs(persistent_directory)

    if not os.path.exists(os.path.join(persistent_directory, "chroma.sqlite3")):
        print(f"\n--- Creating vector store {store_name} ---")
        print(f"Starting embedding and storage process for {len(docs)} documents...")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. \
                Loading it instead of re-initializing."
        )
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )
    return db


def get_retriever():
    db = create_vector_store(chunked_documents, huggingface_embedding, "chroma_db")
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.1},
    )
    return db, retriever


if __name__ == "__main__":
    _, retriever = get_retriever()
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        print("Retrieving relevant documents...\n")
        docs = retriever.invoke(query)
