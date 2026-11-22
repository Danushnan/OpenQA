from dataset import Dataset
from vectordb import VectorDB

def main():
    print("Initialising VectorDB")
    db = VectorDB()
    
    if db.collection.count() == 0:
        print("Loading Dataset")
        dataset = Dataset()
        dataset.load_hotpot()

        print("Creating Chunks")
        chunks = dataset.create_chunks()
        db.add_chunks(chunks)

if __name__ == "__main__":
    print("Starting Program")
    main()