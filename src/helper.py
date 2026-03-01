from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List
from langchain.schema import Document


#Extract text from the PDF files

def load_pdf_files(data):
    Loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = Loader.load()
    return documents


#Making it minimal documents

def filter_to_minimal_docs (docs:List[Document])-> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source" : src}

            )
        )
    return minimal_docs


# Split the documents into smaller chunks


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


#Download embedding model

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name
    )
    return embeddings

