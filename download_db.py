import ast
import chromadb
from datasets import load_dataset


client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_or_create_collection(
    name="hotpot",
    metadata={"hnsw:space": "cosine"}
)

print("Downloading dataset from Hugging Face.")

dataset = load_dataset("NLP-CW/OpenQA", split="train")

BATCH_SIZE = 5000
total_records = len(dataset)

print(f"Restoring {total_records} chunks into local ChromaDB.")

for i in range(0, total_records, BATCH_SIZE):
    batch = dataset.select(range(i, min(i + BATCH_SIZE, total_records)))
    
    ids = list(batch["id"])
    documents = list(batch["document"])
    embeddings = list(batch["embedding"])
    
    metadatas = [ast.literal_eval(m) if isinstance(m, str) else m for m in batch["metadata"]]
    
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Processed batch: {min(i + BATCH_SIZE, total_records)} / {total_records}")

print(f"Successfully restored. Total items in local collection: {collection.count()}")