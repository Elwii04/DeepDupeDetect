# DINOv3 Integration Summary

## Overview
DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) has been successfully added as a model option to the DeepDupeDetect service. This provides an alternative to CLIP models for image duplicate detection.

## Changes Made

### 1. Model Configuration
- Added `DINOV3_MODELS` dictionary with ViT-L configuration:
  - Model: `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - Feature dimension: 1024 (pooler output)
  - Batch size: 400
  - Model type: 'dinov3'

### 2. Model Selection
Updated `_get_model_choice()` to include DINOv3:
- Option [1]: CLIP ViT-Large-14 (768-dim)
- Option [2]: CLIP ViT-Base-32 (512-dim)
- Option [3]: **DINOv3 ViT-Large (1024-dim)** ⬅️ NEW

### 3. Embedding Type Handling
- DINOv3 only supports **pooler output** (no embedding type selection needed)
- CLIP models continue to support:
  - Final embedding
  - Last hidden layer
  - Both embeddings (dual hash)

### 4. Feature Extraction
Updated `_extract_features()` to handle DINOv3:
```python
if self.model_config['type'] == 'dinov3':
    outputs = self.model(pixel_values=image_batch)
    features = outputs.pooler_output
    features = features / features.norm(dim=-1, keepdim=True)
    return features
```

### 5. Model Initialization
Updated `_initialize_model()` to load DINOv3 using transformers:
```python
if self.model_config['type'] == 'dinov3':
    self.preprocess = AutoImageProcessor.from_pretrained(self.model_config['name'])
    self.model = AutoModel.from_pretrained(
        self.model_config['name'],
        device_map=self.device,
        torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32
    )
```

### 6. Data Loading
- Added `model_type` parameter to `ImageDataset` class
- Created separate collate functions:
  - `collate_fn_clip` for CLIP models
  - `collate_fn_dinov3` for DINOv3 models (handles dict output from processor)

### 7. Dependencies
Updated `requirements.txt` to include:
```
transformers>=4.30.0
```

## Key Differences: DINOv3 vs CLIP

| Feature | CLIP | DINOv3 |
|---------|------|---------|
| **Library** | open-clip-torch | transformers |
| **Output Type** | Tensor | ModelOutput with pooler_output |
| **Preprocessing** | Transform function | AutoImageProcessor |
| **Embedding Options** | Final, Last Hidden, Both | Pooler output only |
| **Feature Dimension** | 512 (Base) / 768 (Large) | 1024 (ViT-L) |
| **Use Case** | General vision tasks | Dense features, superior duplicate detection |

## Usage

When running the detector, you'll now see:

```
--- Model Selection ---
Available models:
  [1] CLIP ViT-Large-14 (768-dim, higher quality, slower) [DEFAULT]
  [2] CLIP ViT-Base-32 (512-dim, faster, lower quality)
  [3] DINOv3 ViT-Large (1024-dim, excellent dense features)

Select model (1-3) [1]:
```

If you select DINOv3 (option 3):
```
Selected: facebook/dinov3-vitl16-pretrain-lvd1689m (1024-dimensional)

DINOv3 will use pooler output (no embedding type selection needed)
```

## Technical Notes

### DINOv3 Model Details
- **Full name**: facebook/dinov3-vitl16-pretrain-lvd1689m
- **Architecture**: Vision Transformer Large (ViT-L)
- **Patch size**: 16x16
- **Input size**: 224x224 (scalable to multiples of 16)
- **Register tokens**: 4 (for cleaner attention maps)
- **Training data**: LVD-1689M (1.689 billion curated web images)

### Pooler Output
DINOv3's `pooler_output` is the [CLS] token from the last layer, which:
- Captures global image representation
- Is excellent for duplicate detection
- Is normalized for cosine similarity comparison
- Doesn't require additional projection (unlike CLIP's hidden layers)

### Performance Considerations
- DINOv3 ViT-L produces 1024-dim embeddings (vs CLIP's 768-dim)
- Batch size adjusted to 400 for optimal GPU memory usage
- Compatible with existing database structure (dynamic dimension detection)

## Database Compatibility

The system automatically handles different feature dimensions:
- Detects dimension mismatch on first run
- Prompts user to re-hash if switching models
- Stores model type implicitly through feature dimension

## Testing Recommendations

1. **First run with DINOv3**: Hash a small subset of images
2. **Compare quality**: Run duplicate detection with both CLIP and DINOv3
3. **Benchmark**: Note processing speed and accuracy differences
4. **Threshold tuning**: DINOv3 may require different similarity thresholds

## Future Enhancements

Possible additions:
- DINOv3 ConvNeXt variants
- Smaller DINOv3 models (ViT-S, ViT-B)
- Satellite-trained DINOv3 (SAT-493M) for aerial imagery
- Mixed model databases (store model type in DB)
