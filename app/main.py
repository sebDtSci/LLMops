from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from app.config import config

llm = OpenAI()
embeddings = OpenAIEmbeddings()
db = FAISS.load_local(config.DB_PATH, embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
