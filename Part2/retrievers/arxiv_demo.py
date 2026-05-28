from langchain_community.retrievers import ArxivRetriever

# ArxivRetriever is a built in retriever for getting the data of research papers.
# create the retriever
retriever = ArxivRetriever(
    load_max_docs=2, # number of papers to retrive
    load_all_available_meta =True
)

# query arxiv
docs = retriever.invoke("Large Language Models")

# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content)  # print first 500 characters