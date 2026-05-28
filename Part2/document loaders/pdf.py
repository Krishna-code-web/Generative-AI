from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

data = PyPDFLoader("GRU.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)

for i in chunks:
    print(i.page_content, end="\n\n")