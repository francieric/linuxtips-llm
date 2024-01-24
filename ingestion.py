import os
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

model_name = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(
    model=model_name
)

#embeddings = OpenAIEmbeddings()

loader = PyPDFLoader("docs/CLT.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "linuxtips"
#if index_name not in pc.list_indexes():
#    pc.create_index(
#        name="linuxtips",dimension=1536,metric="cosine",spec=ServerlessSpec(
#            cloud="aws",region="us-west-2")
#            )

index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embeddings.embed_query, texts
)

#docsearch = pc.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

# Test
query = "o que são as férias?"
docs=vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)
#docs = docsearch.similarity_search(query)
print(docs[0])