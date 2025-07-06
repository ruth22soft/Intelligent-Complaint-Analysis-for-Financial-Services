from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

class VectorStoreBuilder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
    
    def add_documents(self, chunks):
        """Generate embeddings and store with metadata"""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Initialize FAISS index if first batch
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(chunks)
    
    def save(self, save_path):
        """Persist index and metadata"""
        faiss.write_index(self.index, f"{save_path}/index.faiss")
        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(self.metadata, f)