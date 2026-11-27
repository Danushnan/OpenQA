from dataset import Dataset
from vectordb import VectorDB
from llm import LLM
from tracking import save_run

NUM_RUNS = 100
TOP_K = 3

def main():
    print("Initialising VectorDB")
    db = VectorDB()

    print("Initialising Mistral")
    mistral = LLM()

    print(f"Current DB count: {db.collection.count()}")
    print("Loading Dataset")
    dataset = Dataset()
    dataset.load_hotpot()

    print("Creating Chunks")
    chunks = dataset.create_chunks()

    if db.collection.count() < len(chunks):
        db.add_chunks(chunks)
    else:
        print("All chunks already indexed!")

    runs = []
    sample = dataset.df_train.iloc[:NUM_RUNS]
    for id, question, ground_truth in zip(sample["id"], sample["question"], sample["answer"]):
        answers = db.query_db(query=question, n_results=TOP_K)
        context = "\n\n".join(
            f"Document {index + 1}:\n{answer["documents"]}"
            for index, answer in enumerate(answers)
            )
        prompt = f"""
            You are a question-answering assistant.

            Answer the question using only the provided context.

            Instructions:
            - Combine information from multiple documents when necessary.
            - Do not make assumptions or introduce unsupported information.
            - If the context is insufficient, answer "I don't know".
            - Keep your answer concise.
            - Return only the answer, without explanations.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        response = mistral.response(prompt, id)
        print(f"Question: {question}")
        print(f"Answer: {response}\n")

        run = {
            "id": id,
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "retrieved_documents": answers,
            "prompt": prompt
        }

        runs.append(run)

    experiment = {
        "configuration": {
            "top_k": 3,
            "retrieval_model": db.model_name,
            "llm_model": mistral.model
        },
        "runs": runs

    }

    save_run(experiment)

if __name__ == "__main__":
    print("Starting Program")
    main()