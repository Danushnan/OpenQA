import os
import pandas as pd
from datasets import load_dataset

def load_hotpot() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds_train, ds_test = load_dataset("hotpotqa/hotpot_qa", "distractor", split=["train", "validation"])
    df_train = ds_train.to_pandas()
    df_test = ds_test.to_pandas()

    print(f"Columns: {", ".join(df_train.columns.tolist())}")
    print(f"Train Set Shape: {df_train.shape}")
    print(f"Test Set Shape: {df_test.shape}")

    return df_train, df_test

def create_chunks(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame
    ) -> list[dict]:

    chunks = []
    seen = set()

    df = pd.concat([df_train, df_test], ignore_index=True)

    for context in df["context"]:
        for title, sentences in zip(context["title"], context["sentences"]):
            key = (title, " ".join(sentences))
            if key not in seen:
                seen.add(key)
                chunk = {"title": title, "sentences": list(sentences)}
                chunks.append(chunk)

    print(f"Number of chunks: {len(chunks)}")

    return chunks

if __name__ == "__main__":
    df_train, df_test = load_hotpot()
    chunks = create_chunks(df_train, df_test)