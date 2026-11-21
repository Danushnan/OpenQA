import os
import json
import pandas as pd
from datasets import load_dataset

def load_hotpot():
    if "hotpotQA_test.csv" not in os.listdir() or "hotpotQA_train.csv" not in os.listdir():
        ds_train, ds_test = load_dataset("hotpotqa/hotpot_qa", "distractor", split=["train", "validation"])
        ds_train.to_csv("hotpotQA_train.csv")
        ds_test.to_csv("hotpotQA_test.csv")

    df_train = pd.read_csv("hotpotQA_train.csv")
    df_test = pd.read_csv("hotpotQA_test.csv")
    
    print(f"Columns: {", ".join(df_train.columns.tolist())}")
    print(f"Train Set Shape: {df_train.shape}")
    print(f"Test Set Shape: {df_test.shape}")

if __name__ == "__main__":
    load_hotpot()