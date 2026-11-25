import os
import csv
from collections import deque

FILE_PATH = "token_logs.csv"
FIELDNAMES = [
    "model", "prompt", "response", "in_tok", 
    "out_tok", "tot_toks", "cumulative_toks"
    ]

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
        prompt: str, 
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
            "prompt": prompt,
            "response": llm_response,
            "in_tok": in_token,
            "out_tok": out_token,
            "tot_toks": total_tokens,
            "cumulative_toks": new_cumulative_tokens
        }

        writer.writerow(to_write)
