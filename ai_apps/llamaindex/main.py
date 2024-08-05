import chromadb
from core import DATA_PATH, INDEX_PATH

#  from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

# load documents
doc = SimpleDirectoryReader(DATA_PATH).load_data()
# doc = documents.pop()
text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, page in enumerate(doc):
    page_text = page.get_text()
    cur_text_chunks = text_parser.split_text(page_text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

# create client and a new collection
chroma_client = chromadb.PersistentClient(path=INDEX_PATH)
chroma_collection = chroma_client.get_or_create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(doc, storage_context=storage_context, embed_model=embed_model)

# Query Data
chat_engine = index.as_chat_engine()
response = chat_engine.chat("What to do if the dishwasher will not start ?")
print(response.response)
