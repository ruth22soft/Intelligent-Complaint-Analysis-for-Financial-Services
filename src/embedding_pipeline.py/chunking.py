from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_complaints(df, chunk_size=256, chunk_overlap=32):
    """
    Split narratives into chunks optimized for embedding models
    with 256-token max length (like all-MiniLM-L6-v2)
    
    Parameters:
    chunk_size = 256   # Model's max token limit (256 for MiniLM)
    chunk_overlap = 32  # Preserves context between chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    chunks = []
    for _, row in df.iterrows():
        text = row['clean_narrative']
        if not isinstance(text, str) or len(text) < 10:  # Skip empty
            continue
            
        text_chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "chunk_id": f"{row['Complaint ID']}-{i}",
                "complaint_id": row['Complaint ID'],
                "product": row['Product']
            })
    return chunks