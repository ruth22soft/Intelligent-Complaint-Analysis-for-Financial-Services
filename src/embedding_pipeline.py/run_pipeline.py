import pandas as pd
from chunking import chunk_complaints
from embedding import VectorStoreBuilder

def run_pipeline():
    # Load Task 1 output

     # Use raw string (r prefix) for Windows paths
    csv_path = "D:/week 6/Intelligent-Complaint-Analysis-for-Financial-Services/notebooks/data/filtered_complaints.csv"
    df = pd.read_csv(csv_path)
    
    # Generate chunks (adjust params here)
    chunks = chunk_complaints(df, chunk_size=256, chunk_overlap=32)
    print(f"Created {len(chunks)} chunks from {len(df)} complaints")
    
    # Batch processing for memory efficiency
    builder = VectorStoreBuilder()
    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        builder.add_documents(batch)
        print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    
    # Save vector store
    builder.save("vector_store")
    print("Vector store saved to vector_store/")

if __name__ == "__main__":
    run_pipeline()