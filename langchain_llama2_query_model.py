
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def query_doc(query):
    # Load PDF file from data path
    # loader = TextLoader("./docs/data.txt", encoding="utf-8")
    # documents = loader.load()
    # # Split text from PDF into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # texts = text_splitter.split_documents(documents)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Build and persist FAISS vector store

    new_db = FAISS.load_local('db/faiss/db_faiss_tlp', embeddings)
    docs = new_db.similarity_search(query)
    print("Answer------------------------")
    for doc in docs:
        print("************************")
        print(doc.page_content)


if __name__ == "__main__":
    query = "What's the name of the last president?？"
    query_doc(query)


