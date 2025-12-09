import os
import docx
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import pdfplumber
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "algorithms.pdf")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
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

extract_text_from_pdf(file_path)

def extract_text_from_txt(file_stream):
    return file_stream.read().decode("utf-8", errors="ignore")


loader = TextLoader(file_path.replace('.pdf', '.txt'), autodetect_encoding=True)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")


def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
create_vector_store(docs, openai_embeddings, "chroma_db")
