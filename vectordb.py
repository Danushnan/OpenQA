import chromadb
import time
from tqdm import tqdm
from dataset import Dataset
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./vectordb")
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.collection = self.client.get_or_create_collection(name="hotpot", embedding_function=self.embedding_function)

    def add_chunks(self, chunks: list[dict]) -> None:
        start = time.perf_counter()
        batch_size = 250

        progress_bar = tqdm(total=len(chunks), desc="Adding Chunks to VectorDB", unit="chunks")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [str(chunk["id"]) for chunk in batch]
            documents = [f"Title: {chunk['title']}\nSentences: {' '.join(chunk['sentences'])}" for chunk in batch]
            metadatas = [{"title": chunk["title"]} for chunk in batch]

            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            progress_bar.update(len(batch))

        progress_bar.close()
        elapsed = time.perf_counter() - start
        print(f"Adding and Embedding chunks took {elapsed:.2f} seconds")