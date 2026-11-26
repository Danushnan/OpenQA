import gc
import time
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class VectorDB:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.client = chromadb.PersistentClient(path="./vectordb")
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": torch.float16},
        )
        self.collection = self.client.get_or_create_collection(
            name="hotpot",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict],
        gpu_batch_size: int = 128,
        db_batch_size: int = 5000,
    ) -> None:
        start = time.perf_counter()

        # Sort indices by text length to eliminate GPU padding waste
        sorted_indices = sorted(
            range(len(chunks)),
            key=lambda idx: len(chunks[idx]["title"])
            + sum(map(len, chunks[idx]["sentences"])),
        )

        progress_bar = tqdm(
            total=len(sorted_indices),
            desc="Indexing HotpotQA",
            unit="chunks",
        )

        for i in range(0, len(sorted_indices), db_batch_size):
            batch_idx = sorted_indices[i : i + db_batch_size]
            candidate_ids = [str(chunks[idx]["id"]) for idx in batch_idx]

            existing_ids = set(
                self.collection.get(ids=candidate_ids, include=[])["ids"]
            )
            filtered_idx = [
                idx
                for idx in batch_idx
                if str(chunks[idx]["id"]) not in existing_ids
            ]

            if not filtered_idx:
                progress_bar.update(len(batch_idx))
                continue

            ids = [str(chunks[idx]["id"]) for idx in filtered_idx]
            documents = [
                f"Title: {chunks[idx]['title']}\nSentences: {' '.join(chunks[idx]['sentences'])}"
                for idx in filtered_idx
            ]
            metadatas = [{"title": chunks[idx]["title"]} for idx in filtered_idx]

            with torch.inference_mode():
                embeddings = self.model.encode(
                    documents,
                    batch_size=gpu_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

            torch.cuda.empty_cache()

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            del ids, documents, metadatas, embeddings
            progress_bar.update(len(batch_idx))

        progress_bar.close()
        gc.collect()
        elapsed = time.perf_counter() - start
        print(
            f"Done! Total collection count: {self.collection.count()} (took {elapsed:.2f}s)"
        )

    def query_db(
        self,
        query: str,
        n_results: int = 3
    ) -> list[dict]:
        
        with torch.inference_mode():
            embeddings = self.model.encode(
                query,
                normalize_embeddings=True
            ).tolist()

        torch.cuda.empty_cache()

        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=n_results
        )

        return [
            {"id": results["ids"][0][index],
             "distances": results["distances"][0][index],
             "documents": results["documents"][0][index]
             }
            for index in range(n_results)]