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
    import gc
    import psutil
    builder = VectorStoreBuilder()
    batch_size = 50  # Reduce batch size to lower memory usage

    process = psutil.Process()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Batch {i//batch_size+1}: Processing {len(batch)} chunks...")
        builder.add_documents(batch)
        mem = process.memory_info().rss / (1024 * 1024)
        print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks | Memory usage: {mem:.2f} MB")
        del batch
        gc.collect()

    # Save vector store
    builder.save("vector_store")
    print("Vector store saved to vector_store/")

if __name__ == "__main__":
    run_pipeline()