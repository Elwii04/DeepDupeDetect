"""
Embedding Compression Test
===========================

This script tests compression of existing DINOv3 Large embeddings from the database:
1. Load existing float32 embeddings (1024-dim DINOv3 Large)
2. Test multiple compression methods (float16, several int8 variants, binary)
3. Measure quality degradation with expanded metrics

Based on research:
- Float16: 2x compression, ~0% loss
- Int8: 4x compression, ~0-5% loss  
- Binary: 32x compression, ~4-11% loss

NO MODEL DOWNLOADS - uses existing embeddings only!
"""

import numpy as np
import sqlite3
from pathlib import Path
from tqdm import tqdm
import time
import json
from datetime import datetime

# Configuration
DB_PATH = r"H:\Raw Datasets\Pinterest Downloader\Downloads\image_database - Kopie.db"
OUTPUT_DIR = Path("embedding_test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test settings
NUM_TEST_IMAGES = None  # None = use all embeddings in the database
NUM_QUERY_SAMPLES = 1000  # Number of queries for ranking tests
TOP_K_VALUES = [1, 5, 10, 20, 50, 100]  # k values for ranking metrics
SIMILARITY_SAMPLE_SIZE = 500  # Pairwise similarity sample size
SPEED_QUERY_SAMPLES = 200  # Queries for speed benchmark
SPEED_CANDIDATE_LIMIT = 20000  # Limit candidates for speed benchmark
RNG_SEED = 42

print(f"Output directory: {OUTPUT_DIR}")
RNG = np.random.default_rng(RNG_SEED)


# ============================================================================
# Quantization Functions
# ============================================================================

def quantize_float16(embeddings):
    """Convert float32 to float16 (2x compression)."""
    return embeddings.astype(np.float16)


def dequantize_float16(embeddings):
    """Convert back to float32."""
    return embeddings.astype(np.float32)


def quantize_int8_per_vector_minmax(embeddings):
    """Int8 per-vector min-max scaling."""
    mins = embeddings.min(axis=1, keepdims=True)
    maxs = embeddings.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    scaled = ((embeddings - mins) / ranges) * 254 - 127
    quantized = np.clip(scaled, -127, 127).astype(np.int8)
    return quantized, {'mins': mins, 'maxs': maxs}


def dequantize_int8_per_vector_minmax(quantized, metadata):
    mins = metadata['mins']
    maxs = metadata['maxs']
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    scaled = (quantized.astype(np.float32) + 127) / 254
    return scaled * ranges + mins


def quantize_int8_per_vector_maxabs(embeddings):
    """Int8 symmetric per-vector scaling using max-abs."""
    max_abs = np.max(np.abs(embeddings), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    scaled = embeddings / max_abs
    quantized = np.clip(np.round(scaled * 127), -127, 127).astype(np.int8)
    return quantized, {'max_abs': max_abs}


def dequantize_int8_per_vector_maxabs(quantized, metadata):
    max_abs = metadata['max_abs']
    return (quantized.astype(np.float32) / 127) * max_abs


def quantize_binary_sign(embeddings):
    """Binary sign quantization with per-vector magnitude storage."""
    magnitudes = np.linalg.norm(embeddings, axis=1, keepdims=True)
    binary = (embeddings > 0).astype(np.uint8)
    packed = np.packbits(binary, axis=1)
    return packed, {'magnitudes': magnitudes, 'original_dim': embeddings.shape[1]}


def dequantize_binary_sign(packed, metadata):
    """Approximate float32 by unpacking sign bits and restoring magnitude."""
    binary = np.unpackbits(packed, axis=1)
    binary = binary[:, :metadata['original_dim']]
    float_approx = binary.astype(np.float32) * 2 - 1
    norms = np.linalg.norm(float_approx, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    float_approx = float_approx / norms
    return float_approx * metadata['magnitudes']


# ============================================================================
# Similarity and Distance Functions
# ============================================================================

def cosine_similarity_matrix(A, B):
    """
    Compute cosine similarity between all pairs in A and B
    A: (n, d), B: (m, d)
    Returns: (n, m) similarity matrix
    """
    # Normalize
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity
    return np.dot(A_norm, B_norm.T)


def hamming_similarity_vector(query_packed, candidates_packed):
    """Compute Hamming similarity between one packed vector and many."""
    xor = np.bitwise_xor(candidates_packed, query_packed)
    hamming = np.unpackbits(xor, axis=1).sum(axis=1)
    max_dist = candidates_packed.shape[1] * 8
    return 1.0 - (hamming / max_dist)


def hamming_similarity_matrix(packed):
    """Compute pairwise Hamming similarity for packed binary vectors."""
    n = packed.shape[0]
    sims = np.empty((n, n), dtype=np.float32)
    max_dist = packed.shape[1] * 8
    for i in range(n):
        xor = np.bitwise_xor(packed[i], packed)
        hamming = np.unpackbits(xor, axis=1).sum(axis=1)
        sims[i] = 1.0 - (hamming / max_dist)
    return sims


# ============================================================================
# Database Loading
# ============================================================================

def load_existing_embeddings(db_path, limit=None):
    """Load existing embeddings from database."""
    print(f"\nLoading existing embeddings from database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT absolute_path, hash FROM images WHERE hash IS NOT NULL"
    if limit is not None:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    paths = []
    embeddings = []
    
    for path, embedding_blob in tqdm(results, desc="Loading embeddings"):
        if Path(path).exists():
            paths.append(path)
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings.append(embedding)
    
    embeddings_array = np.vstack(embeddings)
    print(f"Loaded {len(paths)} embeddings with dimension {embeddings_array.shape[1]}")
    
    return paths, embeddings_array


# ============================================================================
# Evaluation Metrics
# ============================================================================

def _rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def compute_similarity_preservation(original, compressed, method_name, compressed_mode="cosine"):
    """Measure how well compression preserves pairwise similarities."""
    num_samples = min(SIMILARITY_SAMPLE_SIZE, len(original))
    indices = RNG.choice(len(original), num_samples, replace=False)
    
    orig_sample = original[indices]
    comp_sample = compressed[indices]
    
    orig_sim = cosine_similarity_matrix(orig_sample, orig_sample)
    if compressed_mode == "hamming":
        comp_sim = hamming_similarity_matrix(comp_sample)
    else:
        comp_sim = cosine_similarity_matrix(comp_sample, comp_sample)
    
    orig_flat = orig_sim.flatten()
    comp_flat = comp_sim.flatten()
    
    errors = np.abs(orig_flat - comp_flat)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((orig_flat - comp_flat) ** 2))
    p95 = np.percentile(errors, 95)
    
    pearson = np.corrcoef(orig_flat, comp_flat)[0, 1]
    spearman = np.corrcoef(_rankdata(orig_flat), _rankdata(comp_flat))[0, 1]
    
    return {
        'method': method_name,
        'mean_absolute_error': float(mae),
        'rmse': float(rmse),
        'p95_error': float(p95),
        'pearson': float(pearson),
        'spearman': float(spearman),
        'max_error': float(np.max(errors))
    }


def compute_recall_at_k(original_embeddings, compressed_embeddings, k_values, method_name, compressed_mode="cosine"):
    """Ranking metrics for compressed embeddings vs original ordering."""
    num_queries = min(NUM_QUERY_SAMPLES, len(original_embeddings))
    query_indices = RNG.choice(len(original_embeddings), num_queries, replace=False)
    
    results = {
        'mrr': [],
        'recall': {k: [] for k in k_values},
        'jaccard': {k: [] for k in k_values},
        'ndcg': {k: [] for k in k_values},
        'overlap': {k: [] for k in k_values}
    }
    
    idcg = {k: sum(1 / np.log2(i + 2) for i in range(k)) for k in k_values}
    
    for query_idx in tqdm(query_indices, desc=f"Computing ranking metrics for {method_name}"):
        orig_sims = cosine_similarity_matrix(
            original_embeddings[query_idx:query_idx+1],
            original_embeddings
        )[0]
        if compressed_mode == "hamming":
            comp_sims = hamming_similarity_vector(
                compressed_embeddings[query_idx],
                compressed_embeddings
            )
        else:
            comp_sims = cosine_similarity_matrix(
                compressed_embeddings[query_idx:query_idx+1],
                compressed_embeddings
            )[0]
        
        orig_rank = np.argsort(orig_sims)[::-1]
        comp_rank = np.argsort(comp_sims)[::-1]
        
        orig_rank = orig_rank[orig_rank != query_idx]
        comp_rank = comp_rank[comp_rank != query_idx]
        
        target = orig_rank[0]
        target_pos = np.where(comp_rank == target)[0]
        if target_pos.size == 0:
            results['mrr'].append(0.0)
        else:
            results['mrr'].append(1.0 / (int(target_pos[0]) + 1))
        
        for k in k_values:
            orig_top_k = orig_rank[:k]
            comp_top_k = comp_rank[:k]
            orig_set = set(orig_top_k)
            comp_set = set(comp_top_k)
            overlap = len(orig_set & comp_set)
            
            results['overlap'][k].append(overlap)
            results['recall'][k].append(overlap / k)
            
            denom = (2 * k - overlap)
            results['jaccard'][k].append(overlap / denom if denom > 0 else 0.0)
            
            rel = [1 if idx in orig_set else 0 for idx in comp_top_k]
            dcg = sum(rel_i / np.log2(i + 2) for i, rel_i in enumerate(rel))
            results['ndcg'][k].append(dcg / idcg[k])
    
    summary = {'mrr': float(np.mean(results['mrr']))}
    for k in k_values:
        summary[f'recall@{k}'] = float(np.mean(results['recall'][k]))
        summary[f'jaccard@{k}'] = float(np.mean(results['jaccard'][k]))
        summary[f'ndcg@{k}'] = float(np.mean(results['ndcg'][k]))
        summary[f'overlap@{k}'] = float(np.mean(results['overlap'][k]))
    
    return summary


def measure_search_speed(embeddings, num_queries=SPEED_QUERY_SAMPLES):
    """Measure search speed on a capped candidate set for consistency."""
    candidate_count = min(SPEED_CANDIDATE_LIMIT, len(embeddings))
    candidate_indices = RNG.choice(len(embeddings), candidate_count, replace=False)
    candidates = embeddings[candidate_indices]
    
    query_count = min(num_queries, len(candidates))
    query_indices = RNG.choice(len(candidates), query_count, replace=False)
    queries = candidates[query_indices]
    
    start = time.time()
    _ = cosine_similarity_matrix(queries, candidates)
    elapsed = time.time() - start
    
    return {
        'total_time': float(elapsed),
        'time_per_query': float(elapsed / query_count),
        'queries_per_second': float(query_count / elapsed),
        'candidates': int(candidate_count),
        'queries': int(query_count)
    }


def measure_search_speed_hamming(packed_embeddings, num_queries=SPEED_QUERY_SAMPLES):
    """Measure Hamming search speed for packed binary embeddings."""
    candidate_count = min(SPEED_CANDIDATE_LIMIT, len(packed_embeddings))
    candidate_indices = RNG.choice(len(packed_embeddings), candidate_count, replace=False)
    candidates = packed_embeddings[candidate_indices]
    
    query_count = min(num_queries, len(candidates))
    query_indices = RNG.choice(len(candidates), query_count, replace=False)
    queries = candidates[query_indices]
    
    start = time.time()
    for query in queries:
        _ = hamming_similarity_vector(query, candidates)
    elapsed = time.time() - start
    
    return {
        'total_time': float(elapsed),
        'time_per_query': float(elapsed / query_count),
        'queries_per_second': float(query_count / elapsed),
        'candidates': int(candidate_count),
        'queries': int(query_count)
    }


def calculate_storage_size(embeddings, metadata=None):
    """Calculate storage size in bytes, including metadata overhead."""
    total = embeddings.nbytes
    if metadata:
        for value in metadata.values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
    return total


# ============================================================================
# Main Testing Function
# ============================================================================

def run_compression_tests():
    """Main test orchestration - test compression on existing embeddings"""
    results = {
        'test_config': {
            'database': str(DB_PATH),
            'num_test_images': NUM_TEST_IMAGES,
            'num_query_samples': NUM_QUERY_SAMPLES,
            'similarity_sample_size': SIMILARITY_SAMPLE_SIZE,
            'speed_query_samples': SPEED_QUERY_SAMPLES,
            'speed_candidate_limit': SPEED_CANDIDATE_LIMIT,
            'timestamp': datetime.now().isoformat()
        },
        'original_embeddings': {},
        'compression': {}
    }
    
    print("\n" + "="*80)
    print("LOADING EXISTING EMBEDDINGS FROM DATABASE")
    print("="*80)
    
    paths, original_embeddings = load_existing_embeddings(DB_PATH, limit=NUM_TEST_IMAGES)
    
    results['original_embeddings'] = {
        'model': 'DINOv3 ViT-L (from database)',
        'dimension': int(original_embeddings.shape[1]),
        'num_samples': int(len(original_embeddings)),
        'storage_bytes': calculate_storage_size(original_embeddings),
        'storage_mb': calculate_storage_size(original_embeddings) / (1024**2)
    }
    
    print(f"\nOriginal embeddings info:")
    print(f"  Model: DINOv3 ViT-L")
    print(f"  Dimension: {original_embeddings.shape[1]}")
    print(f"  Samples: {len(original_embeddings)}")
    print(f"  Storage: {results['original_embeddings']['storage_mb']:.2f} MB")
    
    # ========================================================================
    # Test compression methods
    # ========================================================================
    
    print("\n" + "="*80)
    print("TESTING COMPRESSION METHODS")
    print("="*80)
    
    compression_configs = [
        ('float16', lambda x: quantize_float16(x), lambda x: dequantize_float16(x)),
        ('int8_vec_minmax', lambda x: quantize_int8_per_vector_minmax(x), lambda x, m: dequantize_int8_per_vector_minmax(x, m)),
        ('int8_vec_maxabs', lambda x: quantize_int8_per_vector_maxabs(x), lambda x, m: dequantize_int8_per_vector_maxabs(x, m)),
        ('binary_sign', lambda x: quantize_binary_sign(x), lambda x, m: dequantize_binary_sign(x, m))
    ]
    
    for method_name, quantize_fn, dequantize_fn in compression_configs:
        print(f"\n{'='*80}")
        print(f"Testing {method_name.upper()} quantization")
        print(f"{'='*80}")
        
        # Quantize
        print(f"Quantizing to {method_name}...")
        quantize_result = quantize_fn(original_embeddings)
        
        # Handle methods that return metadata
        if isinstance(quantize_result, tuple):
            quantized, metadata = quantize_result
        else:
            quantized = quantize_result
            metadata = None
        
        compressed_mode = "hamming" if method_name == "binary_sign" else "cosine"
        dequantized = None
        if compressed_mode == "cosine":
            print("Dequantizing back to float32...")
            if metadata is not None:
                dequantized = dequantize_fn(quantized, metadata)
            else:
                dequantized = dequantize_fn(quantized)
        
        # Compute metrics
        print("Computing similarity preservation...")
        sim_source = quantized if compressed_mode == "hamming" else dequantized
        sim_preservation = compute_similarity_preservation(
            original_embeddings, sim_source, method_name, compressed_mode=compressed_mode
        )
        
        print("Computing recall@k...")
        recall_source = quantized if compressed_mode == "hamming" else dequantized
        recall_metrics = compute_recall_at_k(
            original_embeddings, recall_source, TOP_K_VALUES, method_name, compressed_mode=compressed_mode
        )
        
        print("Measuring search speed...")
        speed_original = measure_search_speed(original_embeddings)
        if compressed_mode == "hamming":
            speed_compressed = measure_search_speed_hamming(quantized)
        else:
            speed_compressed = measure_search_speed(dequantized)
        
        # Storage
        storage_original = calculate_storage_size(original_embeddings)
        storage_compressed = calculate_storage_size(quantized, metadata)
        compression_ratio = storage_original / storage_compressed
        
        # Store results
        results['compression'][method_name] = {
            'storage': {
                'original_bytes': int(storage_original),
                'compressed_bytes': int(storage_compressed),
                'original_mb': float(storage_original / (1024**2)),
                'compressed_mb': float(storage_compressed / (1024**2)),
                'compression_ratio': float(compression_ratio),
                'space_saved_percent': float((1 - 1/compression_ratio) * 100)
            },
            'similarity_preservation': sim_preservation,
            'recall': recall_metrics,
            'speed': {
                'original': speed_original,
                'compressed': speed_compressed
            }
        }
        
        print(f"\n{method_name.upper()} Results:")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Similarity (Pearson): {sim_preservation['pearson']:.4f}")
        print(f"  Recall@10: {recall_metrics['recall@10']:.4f}")
        print(f"  NDCG@10: {recall_metrics['ndcg@10']:.4f}")
        print(f"  MRR: {recall_metrics['mrr']:.4f}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    results_file = OUTPUT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")
    
    return results


def generate_report(results):
    """Generate a comprehensive markdown report"""
    
    report = f"""# Embedding Compression Test Report

**Generated:** {results['test_config']['timestamp']}  
**Database:** {results['test_config']['database']}  
**Test Images:** {results['test_config']['num_test_images']}

## Summary

### Original Embeddings

- **Model:** {results['original_embeddings']['model']}
- **Dimension:** {results['original_embeddings']['dimension']}
- **Samples:** {results['original_embeddings']['num_samples']}
- **Storage:** {results['original_embeddings']['storage_mb']:.2f} MB

### Compression Methods

| Method | Compression Ratio | Storage Saved | Pearson | Spearman | Recall@10 | NDCG@10 | MRR |
|--------|-------------------|---------------|---------|----------|-----------|---------|-----|
"""
    
    for method in sorted(results['compression'].keys()):
        comp = results['compression'][method]
        report += (
            f"| {method.upper()} | {comp['storage']['compression_ratio']:.2f}x | "
            f"{comp['storage']['space_saved_percent']:.1f}% | "
            f"{comp['similarity_preservation']['pearson']:.4f} | "
            f"{comp['similarity_preservation']['spearman']:.4f} | "
            f"{comp['recall']['recall@10']:.4f} | "
            f"{comp['recall']['ndcg@10']:.4f} | "
            f"{comp['recall']['mrr']:.4f} |\n"
        )
    
    report += f"""

## Detailed Results

### Baseline: {results['original_embeddings']['model']}

- **Dimension:** {results['original_embeddings']['dimension']}
- **Storage:** {results['original_embeddings']['storage_mb']:.2f} MB
- **Samples:** {results['original_embeddings']['num_samples']}

### Compression Analysis

"""
    
    for method in sorted(results['compression'].keys()):
        comp = results['compression'][method]
        report += f"""#### {method.upper()} Quantization

**Storage:**
- Compression Ratio: {comp['storage']['compression_ratio']:.2f}x
- Original Size: {comp['storage']['original_mb']:.2f} MB
- Compressed Size: {comp['storage']['compressed_mb']:.2f} MB
- Space Saved: {comp['storage']['space_saved_percent']:.1f}%

**Similarity Preservation:**
- Pearson: {comp['similarity_preservation']['pearson']:.4f}
- Spearman: {comp['similarity_preservation']['spearman']:.4f}
- RMSE: {comp['similarity_preservation']['rmse']:.6f}
- Mean Absolute Error: {comp['similarity_preservation']['mean_absolute_error']:.6f}
- P95 Error: {comp['similarity_preservation']['p95_error']:.6f}
- Max Error: {comp['similarity_preservation']['max_error']:.6f}

**Ranking Metrics:**
- MRR: {comp['recall']['mrr']:.4f}
"""
        for k in [1, 5, 10, 20, 50, 100]:
            report += (
                f"- Recall@{k}: {comp['recall'][f'recall@{k}']:.4f} "
                f"| NDCG@{k}: {comp['recall'][f'ndcg@{k}']:.4f} "
                f"| Jaccard@{k}: {comp['recall'][f'jaccard@{k}']:.4f}\n"
            )
        
        report += "\n"
    
    report += """
## Recommendations

Based on the test results:

### For Maximum Quality (minimal loss):
- **Use Float16**: Nearly identical quality with 2x compression
- Best for production use where accuracy is critical

### For Balanced Performance:
- **Use Int8**: Good compression (4x) with typically <5% quality loss
- Best for most real-world applications

### For Maximum Speed/Compression:
- **Use Binary**: 32x compression, but significant quality loss
- Best for initial filtering with float32/int8 rescoring on top candidates
- Not recommended as sole method

### Practical Recommendations:

1. **If storage is not an issue**: Keep float32 or use float16 (minimal risk)
2. **If storage is tight**: Use int8 (good balance, test recall first)
3. **For massive scale**: Two-stage retrieval with binary + int8 rescoring
"""
    
    report_file = OUTPUT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
    
    # Also print to console
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    return report_file


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EMBEDDING COMPRESSION TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Database: {DB_PATH}")
    print(f"  Test images: {NUM_TEST_IMAGES}")
    print(f"  Query samples: {NUM_QUERY_SAMPLES}")
    print("="*80)
    
    # Run tests
    results = run_compression_tests()
    
    # Generate report
    generate_report(results)
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
