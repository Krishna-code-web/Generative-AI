from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import TokenTextSplitter

url = "https://www.apple.com/in/macbook-pro/"

splitter = TokenTextSplitter(
    chunk_size=40,
    chunk_overlap=2
)

data = WebBaseLoader(url)

docs = data.load()

chunks = splitter.split_documents(docs)
for i in chunks:
    print(i.page_content)
    print()