import sys
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import  ChatOllama
import os  
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

from langchain_community.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_community.callbacks.manager import CallbackManager
from langchain_community.callbacks.manager import StreamingStdOutCallbackHandler
from langchain_community.indexes import VectorstoreIndexCreator



  

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

PERSIST = True
if PERSIST and os.path.exists("stores/pets"):
   print("Reusing index...\n")
   vectorstore = Chroma(persist_directory="stores/pets", embedding_function=GPT4AllEmbeddings())
   index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = PyPDFLoader('pet.pdf', extract_images=True)
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"stores/pets"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOllama(model="mistral:7b",verbose=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k":1}),
  chain_type="stuff"
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None

