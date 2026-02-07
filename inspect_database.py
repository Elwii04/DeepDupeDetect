"""
Quick script to inspect the database and determine embedding type and dimensions
"""
import sqlite3
import numpy as np

db_path = r"H:\Raw Datasets\Pinterest Downloader\Downloads\image_database - Kopie.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in database: {tables}")
    
    # Get schema
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print(f"\n{table_name} schema:")
        for col in columns:
            print(f"  {col}")
    
    # Get sample embeddings
    cursor.execute("SELECT hash, final_embedding_hash FROM images LIMIT 5")
    samples = cursor.fetchall()
    
    print(f"\n\nNumber of sample embeddings retrieved: {len(samples)}")
    
    if samples:
        for i, sample in enumerate(samples):
            embedding_blob = sample[0] if sample[0] is not None else sample[1]
            if embedding_blob is None:
                print(f"\nSample {i+1}: No embedding data")
                continue
            
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embedding_type = "hash (last hidden)" if sample[0] is not None else "final_embedding_hash"
            print(f"\nSample {i+1} ({embedding_type}):")
            print(f"  Embedding dimension: {len(embedding)}")
            print(f"  First 10 values: {embedding[:10]}")
            print(f"  Value range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM images")
    total = cursor.fetchone()[0]
    print(f"\n\nTotal images in database: {total}")
    
    conn.close()
    
    # Determine model type based on dimension
    print("\n" + "="*60)
    if samples:
        embedding_blob = samples[0][0] if samples[0][0] is not None else samples[0][1]
        if embedding_blob:
            dim = len(np.frombuffer(embedding_blob, dtype=np.float32))
            if dim == 512:
                print("DETECTED: CLIP ViT-B-32 (base) embeddings")
            elif dim == 768:
                print("DETECTED: CLIP ViT-L-14 (large) embeddings")
            elif dim == 1024:
                print("DETECTED: DINOv3 ViT-L (large) embeddings")
            elif dim == 384:
                print("DETECTED: DINOv3 ViT-S (small) or ViT-B (base) embeddings")
            else:
                print(f"UNKNOWN: Embedding dimension {dim} doesn't match known models")
        print("="*60)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
