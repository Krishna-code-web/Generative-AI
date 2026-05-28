#load pdf -> Document Loading
#split into chunks -> Text Splitting
#create the embeddings -> Embedding
#store into chroma -> Vector database
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

model = MistralAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=model,
    persist_directory="chroma-database"
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

query = "What is deep learning?"

result = retriever.invoke(query)

for i in result:
    print(i.page_content)
