# Quick Start Guide: Using DINOv3 for Duplicate Detection

## Installation

First, install the updated dependencies:

```powershell
pip install -r requirements.txt
```

This will install the new `transformers` library needed for DINOv3.

## First-Time Setup with DINOv3

1. **Run the detector**:
   ```powershell
   python detector.py
   ```

2. **Enter your image folder path** when prompted.

3. **Select mode** - Choose option [2] or [4] to hash images:
   ```
   [2] Generate/Update Image Hashes
   [4] Full Workflow (Hash then Remove)
   ```

4. **Model Selection** - Choose DINOv3:
   ```
   --- Model Selection ---
   Available models:
     [1] CLIP ViT-Large-14 (768-dim, higher quality, slower) [DEFAULT]
     [2] CLIP ViT-Base-32 (512-dim, faster, lower quality)
     [3] DINOv3 ViT-Large (1024-dim, excellent dense features)
   
   Select model (1-3) [1]: 3
   ```

5. **No embedding selection needed** - DINOv3 automatically uses pooler output:
   ```
   Selected: facebook/dinov3-vitl16-pretrain-lvd1689m (1024-dimensional)
   
   DINOv3 will use pooler output (no embedding type selection needed)
   ```

6. **Hash your images**:
   - Choose whether to re-hash all images or only new ones
   - Wait for the hashing process to complete

7. **Find duplicates**:
   - If you chose mode [4], duplicate detection starts automatically
   - Otherwise, return to the menu and select option [3]

## Switching Between Models

### From CLIP to DINOv3

If you have an existing database with CLIP embeddings:

1. Run the detector as usual
2. Select a hashing operation (mode 2 or 4)
3. Choose DINOv3 when prompted
4. The system will detect the dimension mismatch and ask:
   ```
   ⚠️  WARNING: Database dimension mismatch!
   Database contains 768-dimensional features
   Selected model produces 1024-dimensional features
   
   Do you want to re-hash all images with the new configuration? (y/n):
   ```
5. Enter `y` to re-hash with DINOv3

### From DINOv3 to CLIP

Same process - the system will detect the mismatch and prompt you to re-hash.

## Performance Tips

### GPU Memory Management

DINOv3 ViT-L is configured with a batch size of 400, which should work well on most modern GPUs. If you encounter out-of-memory errors:

1. Edit `detector.py` line ~42:
   ```python
   DINOV3_MODELS = {
       'vitl': {'name': 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'feature_dim': 1024, 'batch_size': 300, 'type': 'dinov3'}
   }
   ```
2. Reduce `batch_size` from 400 to 300 or lower

### Speed Optimization

- **CUDA**: DINOv3 automatically uses FP16 on CUDA devices for faster processing
- **CPU**: If running on CPU, expect significantly slower performance (uses FP32)
- **Workers**: The system uses `cpu_count - 3` workers for data loading

### Similarity Threshold

DINOv3 may perform differently than CLIP with the same threshold. Consider:

- **Default (0.96)**: Good starting point
- **Stricter (0.97-0.98)**: If you're getting too many false positives
- **Looser (0.93-0.95)**: If you're missing duplicates

To use a custom threshold, select it when running duplicate detection:
```
Use default (0.96) or enter custom value (0.0-1.0) [default]: 0.95
```

## Model Comparison Example

Run this experiment to compare CLIP vs DINOv3:

1. **Test with CLIP Large**:
   - Hash images with CLIP ViT-L-14
   - Note the number of duplicate groups found
   - Note processing time

2. **Test with DINOv3**:
   - Re-hash same images with DINOv3
   - Compare duplicate groups found
   - Compare processing time

3. **Analyze results**:
   - DINOv3 often finds more subtle duplicates due to superior dense features
   - Processing speed should be comparable
   - Memory usage may be slightly higher (1024-dim vs 768-dim)

## Troubleshooting

### Error: "Hugging Face token required"

DINOv3 models may require accepting the license on Hugging Face:

1. Go to: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
2. Click "Agree and access repository"
3. Create a Hugging Face token if needed
4. Set environment variable:
   ```powershell
   $env:HF_TOKEN = "your_token_here"
   ```

### Error: "Out of memory"

Reduce batch size in the `DINOV3_MODELS` configuration (see Performance Tips above).

### Warning: "Image size not multiple of 16"

DINOv3 works best with images that are multiples of 16x16 (patch size). The model will automatically crop if needed, but this may affect features slightly. Not typically a problem for duplicate detection.

## Advanced: Mixed Model Usage

You can maintain separate databases for different model types:

1. **Keep CLIP database**: Rename `image_database.db` to `image_database_clip.db`
2. **Create DINOv3 database**: Run hashing with DINOv3, creates new `image_database.db`
3. **Switch between them**: Rename databases as needed

This allows you to compare results without re-hashing each time.

## Expected First Run Output

```
--- Welcome to the GPU-Accelerated Image Deduplicator ---
Please enter the full path to your image folder: C:\Images

Please select a mode to run:
  [1] Quick Filename Cleanup
  [2] Generate/Update Image Hashes (will ask for model selection)
  [3] Find & Remove Duplicates (with Visual Reviewer)
  [4] Full Workflow (Hash then Remove - will ask for model selection)
  [5] Clean/Verify Database (removes entries for deleted files)
  [6] Add Another Folder Path (for cross-database duplicate detection)
  [7] Exit

Enter your choice (1-7): 4

--- Running Full Workflow ---
Re-hash ALL images (y) or only process NEW images (n)? [n]: y

Step 1: Hashing images...

--- Model Selection ---
Available models:
  [1] CLIP ViT-Large-14 (768-dim, higher quality, slower) [DEFAULT]
  [2] CLIP ViT-Base-32 (512-dim, faster, lower quality)
  [3] DINOv3 ViT-Large (1024-dim, excellent dense features)

Select model (1-3) [1]: 3
Selected: facebook/dinov3-vitl16-pretrain-lvd1689m (1024-dimensional)

DINOv3 will use pooler output (no embedding type selection needed)

Loading pre-trained model (facebook/dinov3-vitl16-pretrain-lvd1689m)...
Model loaded.
Scanning for image files...
Found 1523 total image files.
Re-hashing all images as requested.

Current configuration:
  Model: facebook/dinov3-vitl16-pretrain-lvd1689m
  Embedding: pooler output
  Dimensions: 1024
  Batch size: 400

Starting hashing with 8 parallel workers...
Hashing Images: 100%|████████████████| 4/4 [00:12<00:00,  3.14s/it]
Hashing complete.

Step 2: Finding and removing duplicates...
[... duplicate detection continues ...]
```

## Next Steps

- Review the [full integration documentation](DINOV3_INTEGRATION.md) for technical details
- Experiment with different similarity thresholds
- Compare results with CLIP models
- Report any issues or feedback
