from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
import os

load_dotenv()



loader = DirectoryLoader('./docs', glob="**/*.pdf", show_progress=True)

docs = loader.load()