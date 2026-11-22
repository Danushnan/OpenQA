import os
import pandas as pd
from datasets import load_dataset

class Dataset:
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.chunks = []

    def load_hotpot(self) -> None:
        ds_train, ds_test = load_dataset("hotpotqa/hotpot_qa", "distractor", split=["train", "validation"])
        
        self.df_train = ds_train.to_pandas()
        self.df_test = ds_test.to_pandas()

        print(f"Columns: {", ".join(self.df_train.columns.tolist())}")
        print(f"Train Set Shape: {self.df_train.shape}")
        print(f"Test Set Shape: {self.df_test.shape}")

    
    def create_chunks(self) -> list[dict]:
        chunks = []
        seen = set()

        df = pd.concat([self.df_train, self.df_test], ignore_index=True)

        for context in df["context"]:
            for title, sentences in zip(context["title"], context["sentences"]):
                key = (title, " ".join(sentences))
                if key not in seen:
                    seen.add(key)
                    chunk = {"id": len(chunks), "title": title, "sentences": list(sentences)}
                    chunks.append(chunk)

        self.chunks = chunks
        print(f"Number of chunks: {len(self.chunks)}")

        return self.chunks

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load_hotpot()
    chunks = dataset.create_chunks()
    print(len(chunks))