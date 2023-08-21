from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
import argparse
import timeit
from langchain.llms import CTransformers

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def save_doc():
    # Load PDF file from data path
    loader = TextLoader("./docs/data2_tlp.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Build and persist FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local('db/faiss/db_faiss')


def query_doc():
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
    query = "孙悟空师父是谁？"
    new_db = FAISS.load_local('db/faiss/db_faiss', embeddings)
    docs = new_db.similarity_search(query)
    print(docs[0])


def set_qa_prompt():
    qa_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=qa_template,
    input_variables=['context', 'question'])
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dba_r = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
        )
    return dba_r


# Instantiate QA object
def setup_dbqa():
    llm = CTransformers(model='./models/llama2/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256, 'temperature': 0.01}
                        )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local('db/faiss/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
    return dbqa


if __name__ == "__main__":
    query_doc()
    start = timeit.default_timer() # Start timer

    # Setup QA object
    dbqa = setup_dbqa()

    # Parse input from argparse into QA object
    response = dbqa({'query': "孙悟空的师父是谁？"})
    end = timeit.default_timer() # End timer

    # Print document QA response
    print(f'\nAnswer: {response["result"]}')
    print('='*50) # Formatting separator
    # Display time taken for CPU inference
    print(f"Time to retrieve response: {end - start}")
    # Process source documents for better display
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i+1}\n')
        print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata["source"]}')
        print('='* 50) # Formatting separator

