from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")

# lazy_load()
doc = loader.load()

print(len(doc))

print(doc[0].page_content)
print(doc[0].metadata)