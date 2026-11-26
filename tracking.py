import os
import csv
import json
from collections import deque
from functools import reduce

FILE_PATH = "token_logs.csv"
FIELDNAMES = ["model", "in_tok", "out_tok", "tot_toks", "cumulative_toks"]

# Model: (input price per million tokens, output price per million tokens)
PRICES = {
    "ministral-3b-latest": (0.1, 0.1)
}


def get_last_cumulative_tokens(filepath: str) -> int:
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return 0
    
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        last_row = deque(reader, maxlen=1)

        if not last_row:
            return 0
        return int(last_row[-1]["cumulative_toks"])

def log_tokens(
        model: str, 
        response: str
    ) -> None:    
    file_exists = os.path.exists(FILE_PATH) and os.path.getsize(FILE_PATH) > 0
    last_cumulative_tokens = get_last_cumulative_tokens(FILE_PATH)

    llm_response = response.choices[0].message.content
    in_token = response.usage.prompt_tokens
    out_token = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    new_cumulative_tokens = last_cumulative_tokens + total_tokens

    with open(FILE_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        to_write = {
            "model": model,
            "in_tok": in_token,
            "out_tok": out_token,
            "tot_toks": total_tokens,
            "cumulative_toks": new_cumulative_tokens
        }

        writer.writerow(to_write)

def estimate_price(path=FILE_PATH) -> float:
    total_cost = 0
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        
        total_input = 0
        total_output = 0

        for row in reader:
            model = row["model"]
            in_tok = int(row["in_tok"])
            out_tok = int(row["out_tok"])

            total_input += in_tok
            total_output += out_tok

            input_price, output_price = PRICES[model]

            total_cost += (in_tok * input_price + out_tok * output_price) / 1_000_000

    return total_cost

def save_run(data):
    os.makedirs("Experiments", exist_ok=True)
    runs = os.listdir("Experiments")

    if runs:
        last_experiment_number = max(int(run.replace(".json", "").split("_")[-1]) for run in runs)
        path = f"Experiments/Experiments_{last_experiment_number+1}.json"
    else:
        path = "Experiments/Experiments_1.json"

    with open(path, "w") as f:
        json.dump(data, f, indent=3)

if __name__ == "__main__":
    print(estimate_price())