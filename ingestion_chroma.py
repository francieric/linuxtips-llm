from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key=os.getenv('OPENAI_API_KEY', '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')

os.environ["OPENAI_API_KEY"] = "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"

loader = DirectoryLoader('./docs', glob="**/*.pdf", show_progress=True)

docs = loader.load()