import nltk
import json
import numpy as np
import argparse
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from collections import Counter

def normalise_answer(text):
    text = text.lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    words = text.split()
    words = [word for word in words if word not in {"a", "an", "the"}]

    return " ".join(words)

def calculate_token_f1(prediction, reference):
    prediction_tokens = normalise_answer(prediction).split()
    reference_tokens = normalise_answer(reference).split()

    if not prediction_tokens or not reference_tokens:
        if prediction_tokens == reference_tokens:
            return 1
        return 0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)

def evaluate_model_answers(results):
    predictions = [result["response"] for result in results]
    references = [result["ground_truth"] for result in results]

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    meteor_scores = meteor.compute(predictions=predictions, references=references)
    bleu_scores =[sentence_bleu([reference.split()], prediction.split(), smoothing_function=SmoothingFunction().method1) 
                  for prediction, reference in zip(predictions, references)]
    bleurt_scores = bleurt.compute(predictions=predictions, references=references)
    token_f1 = [calculate_token_f1(prediction, reference)
                for prediction, reference in zip(predictions, references)]
    bert_precision, bert_recall, bert_f1 = bert_score(predictions, references, lang="en")
    
    avg_bleu = np.mean(bleu_scores)
    avg_bleurt = np.mean(bleurt_scores["scores"])
    avg_token_f1 = np.mean(token_f1)

    return {
        "rouge_1": float(rouge_scores["rouge1"]),
        "rouge_2": float(rouge_scores["rouge2"]),
        "rouge_l": float(rouge_scores["rougeL"]),
        "bleu": float(avg_bleu),
        "meteor": float(meteor_scores["meteor"]),
        "bleurt": float(avg_bleurt),
        "token_f1": float(avg_token_f1),
        "bert_score_precision": bert_precision.mean().item(),
        "bert_score_recall": bert_recall.mean().item(),
        "bert_score_f1": bert_f1.mean().item()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Path to experiment JSON file")
    args = parser.parse_args()

    with open(args.experiment, "r") as f:
        experiment = json.load(f)

    evaluation = evaluate_model_answers(experiment["runs"])
    experiment = {
        "configuration": experiment["configuration"],
        "evaluation": experiment["evaluation"],
        "runs": experiment["runs"]
    }

    with open(args.experiment, "w") as f:
        json.dump(experiment, f, indent=3)

    for metric, score in evaluation.items():
        print(f"{metric}: {score:.4f}")
    
if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    rouge = load("rouge")
    meteor = load('meteor')
    bleurt = load('bleurt', 'bleurt-base-128')
    main()