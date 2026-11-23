import os
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from vectordb import VectorDB

vector_db = VectorDB()
collection = vector_db.collection

total_records = collection.count()
BATCH_SIZE = 5000 
OUTPUT_DIR = "hf_dataset_export"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Exporting {total_records} chunks from collection '{collection.name}' in batches of {BATCH_SIZE}...")

file_index = 0
parquet_files = []

for offset in range(0, total_records, BATCH_SIZE):
    batch = collection.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=["documents", "embeddings", "metadatas"]
    )
    
    metadatas = [str(m) if m else "{}" for m in batch["metadatas"]]
    embeddings = [list(map(float, emb)) for emb in batch["embeddings"]]
    
    table = pa.table({
        "id": pa.array(batch["ids"], type=pa.string()),
        "document": pa.array(batch["documents"], type=pa.string()),
        "embedding": pa.array(embeddings, type=pa.list_(pa.float32())),
        "metadata": pa.array(metadatas, type=pa.string())
    })
    
    total_batches = (total_records // BATCH_SIZE) + 1
    file_name = f"train-{file_index:05d}-of-{total_batches:05d}.parquet"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    pq.write_table(table, file_path)
    parquet_files.append(file_path)
    print(f"Exported batch file: {file_name} ({min(offset + BATCH_SIZE, total_records)}/{total_records})")
    file_index += 1


print("Uploading files to Hugging Face")
api = HfApi()

for file_path in parquet_files:
    file_name = os.path.basename(file_path)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"data/{file_name}",
        repo_id="NLP-CW/OpenQA",
        repo_type="dataset"
    )
    print(f"Uploaded: {file_name}")

print("Successfully uploaded all chunks")