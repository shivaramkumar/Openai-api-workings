import chromadb
from core import DATA_PATH, INDEX_PATH

#  from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# create client and a new collection
chroma_client = chromadb.PersistentClient(path=INDEX_PATH)
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
documents = SimpleDirectoryReader(DATA_PATH).load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response.response)
