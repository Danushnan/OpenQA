from dataset import Dataset
from vectordb import VectorDB
from llm import LLM

def main():
    print("Initialising VectorDB")
    db = VectorDB()

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


if __name__ == "__main__":
    print("Starting Program")
    main()