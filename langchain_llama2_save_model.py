
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def save_doc():
    # Load PDF file from data path
    loader = TextLoader("./docs/data2_tlp.txt", encoding="utf-8")
    documents = loader.load()
    # Split text from PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Build and persist FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local('db/faiss/db_faiss_tlp.')


if __name__ == "__main__":
    save_doc()


