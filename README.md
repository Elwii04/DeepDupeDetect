# GPU-Accelerated AI Duplicate Image Finder

A high-performance, AI-powered tool designed to find duplicate and near-duplicate images within massive collections. This script is capable of processing over 1 million images, leveraging state-of-the-art deep learning models to identify visually similar images, even with variations in resolution, cropping, filters, and color.

![image](https://user-images.githubusercontent.com/12345/your-reviewer-screenshot.png) <!-- It's highly recommended to replace this with a real screenshot of the reviewer GUI! -->

## ✨ Key Features

- **State-of-the-Art Accuracy:** Choose between OpenAI's powerful **CLIP** model or Meta's **DINOv3** foundation model for understanding image content on a semantic level, making it incredibly robust against complex transformations.
  - **CLIP**: Excellent general-purpose vision model with text-image understanding
  - **DINOv3**: Superior dense features for visual similarity, outperforming specialized models
- **Massive Scalability:** Engineered to handle huge datasets (tested on 700k+, capable of 1M+ images) by using memory-efficient, batched comparison algorithms.
- **Blazing Fast Hashing:** Leverages your NVIDIA GPU and parallel CPU cores with a PyTorch `DataLoader` to rapidly generate image "hashes" (feature vectors).
- **Interactive Visual Reviewer:** Instead of endless terminal lists, a GUI pops up allowing you to visually review duplicate sets side-by-side before confirming deletion.
- **Smart "Keep" Logic:** Automatically marks the largest file in a duplicate set to be kept, assuming higher file size equals higher quality.
- **Intelligent Database:** Uses a persistent SQLite database to store hashes. You only need to hash your collection once; future runs only process new files.
- **Database Maintenance Tools:** Includes built-in utilities to clean up the database by removing entries for files that have been deleted manually.
- **Cross-Database Duplicate Detection:** NEW! Add a second folder to find duplicates between two separate collections (e.g., different download sources).
- **Configurable Similarity Threshold:** Customize the similarity threshold at runtime without editing code. Choose stricter or looser matching based on your needs.

## 📋 System Requirements

- **Python:** 3.9 or later.
- **GPU:** An NVIDIA GPU with CUDA support is required for acceleration. The more powerful the GPU, the faster the hashing process.
- **Storage:** **An SSD is strongly recommended.** The primary bottleneck for hashing millions of small files is disk I/O speed. Using an HDD will be significantly slower.
- **RAM:** 16 GB or more is recommended for smooth operation with large datasets.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install PyTorch with CUDA support:** This is the most critical step. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and select the correct options for your system (e.g., Pip, Windows/Linux, CUDA). Run the generated command. For example:
    ```bash
    # Example command for PyTorch with CUDA 12.1 - get the latest from the website!
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install all other required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ How to Use

Simply run the script from your terminal. It will guide you through an interactive menu.

```bash
python detector.py
```

First, you will be asked for the full path to your main image folder. The script will then present the following menu:

### Single-Folder Mode (Standard Operation)

When working with one folder, you'll see these options:

1.  **[1] Quick Filename Cleanup:** A fast first pass that finds duplicates based on common naming patterns (e.g., `image.jpg` and `image-001.jpg`).
2.  **[2] Generate/Update Image Hashes:** The core function. It scans your folders, finds new images not yet in the database, and uses the GPU to compute their AI-based hashes. 
    - **Model Selection:** Choose between:
      - **CLIP ViT-Large-14** (768-dim) - Higher quality, general-purpose [DEFAULT]
      - **CLIP ViT-Base-32** (512-dim) - Faster, lower quality
      - **DINOv3 ViT-Large** (1024-dim) - Excellent dense features, superior duplicate detection
    - **Embedding Type (CLIP only):** Choose between final embedding, last hidden layer, or both
    - **DINOv3:** Automatically uses pooler output (no embedding selection needed)
    - Run this first on your collection. The script is interruptible and can be resumed later.
3.  **[3] Find & Remove Duplicates (with Visual Reviewer):** After hashing, use this mode to find all duplicate sets within the folder. A GUI window will launch, allowing you to review each set before making a final deletion confirmation in the terminal.
    - **New:** You'll be prompted to use the default similarity threshold (0.96) or enter a custom value
    - Range: 0.0 (very loose) to 1.0 (identical images only)
    - Recommended: 0.95-0.98 for most use cases
4.  **[4] Full Workflow (Hash then Remove):** A convenient option that runs Mode 2 and then Mode 3 sequentially.
5.  **[5] Clean/Verify Database:** An essential maintenance tool. This scans the database and removes "ghost" entries for files that you have deleted outside of this script, keeping your database in sync with your file system.
6.  **[6] Add Another Folder Path:** Switch to multi-folder mode to find duplicates between two separate collections.
7.  **[7] Exit:** Closes the application.

### Multi-Folder Mode (Cross-Database Detection)

NEW! You can now detect duplicates **between** two separate folders. This is perfect when you have images from different sources and want to find duplicates across them.

#### How It Works:

1. **Initial Setup:**
   - Start by processing each folder separately (hash and clean duplicates within each)
   - Each folder will have its own `image_database.db` file

2. **Adding a Second Folder:**
   - Select option **[6] Add Another Folder Path**
   - Enter the path to your second folder
   - **Important:** The second folder MUST already have its own `image_database.db` file
   - If it doesn't exist, you need to hash that folder first (in a separate session)

3. **Multi-Folder Menu:**
   Once a second folder is added, the menu simplifies to:
   - **[1] Quick Filename Cleanup**
   - **[2] Find & Remove Duplicates** - Now finds duplicates ONLY between the two databases
   - **[3] Exit**
   
   Note: Hashing options are hidden since both databases should already be populated.

4. **Cross-Database Detection (Important!):**
   - In multi-folder mode, the system **ONLY looks for duplicates BETWEEN the two databases**
   - It will NOT re-scan for duplicates within Database 1 or within Database 2
   - This is by design: you should have already cleaned each folder separately
   - Saves processing time by only finding cross-database duplicates

5. **Deletion Strategies:**
   When duplicates are found between two folders, you'll choose how to handle them:
   
   - **[1] Keep Largest Files (Default):** Keeps the largest file from either database, deletes smaller duplicates from both
   - **[2] Keep Files from Database 1:** Keeps ALL files from the first folder, only deletes duplicates from the second folder
   - **[3] Keep Files from Database 2:** Keeps ALL files from the second folder, only deletes duplicates from the first folder

#### Example Workflow:

```
Scenario: Two download folders with potential duplicates
  Folder A: E:\Downloads\Source1 (5000 images)
  Folder B: E:\Downloads\Source2 (3000 images)

Step 1: Hash and clean Folder A
  > python detector.py
  > Enter path: E:\Downloads\Source1
  > [2] Generate/Update Image Hashes
  > [3] Find & Remove Duplicates (finds duplicates within Source1)
  > [7] Exit

Step 2: Hash and clean Folder B
  > python detector.py
  > Enter path: E:\Downloads\Source2
  > [2] Generate/Update Image Hashes
  > [3] Find & Remove Duplicates (finds duplicates within Source2)
  > [7] Exit

Step 3: Find duplicates between A and B
  > python detector.py
  > Enter path: E:\Downloads\Source1
  > [6] Add Another Folder Path
  > Enter second path: E:\Downloads\Source2
  > [2] Find & Remove Duplicates
  > System now ONLY compares between Source1 and Source2
  > Choose deletion strategy:
      [1] Keep largest files from either folder
      [2] Keep all from Source1, delete duplicates from Source2
      [3] Keep all from Source2, delete duplicates from Source1
  > Review in visual GUI
  > Confirm deletion
```

### The Visual Reviewer

-   **Navigate:** Use the `<- Previous` and `Next ->` buttons or your keyboard's **Left and Right Arrow Keys**.
-   **Colors:**
    -   **Green Border:** The file to be **KEPT** (automatically the largest file in the set).
    -   **Red Border:** The file(s) to be **DELETED**.
-   **Action:** Close the window when you are done reviewing. The final deletion confirmation will happen in the terminal.

## ⚙️ Performance & Tuning

If you run into issues or want to maximize speed, you can adjust the configuration variables at the top of the script:

-   `BATCH_SIZE`: The number of images processed by the GPU at once.
    -   **Increase** if you have a high-end GPU with lots of VRAM to potentially improve speed.
    -   **Decrease** if you get "Out of Memory" errors.
-   `NUM_WORKERS`: The number of parallel CPU cores used for loading and preparing data. It's set to a safe default. Increasing it may speed up hashing if your CPU is the bottleneck.
-   `SIMILARITY_THRESHOLD`: The default threshold for duplicate detection (0.96).
    -   **Note:** You can now override this at runtime without editing the code!
    -   Lower values (0.90-0.95) = More matches, but potentially more false positives
    -   Higher values (0.97-0.99) = Fewer matches, only very similar images
    -   1.0 = Identical images only
-   **Optional Speed Boost:** For an extra performance gain during the CPU-bound image loading phase, you can install the `Pillow-SIMD` library, a much faster, drop-in replacement for the standard `Pillow`.
    ```bash
    pip uninstall Pillow
    pip install Pillow-SIMD
    ```

## 🛠️ How It Works (Technical Details)

-   **Hashing:** Instead of simple perceptual hashes, this tool uses advanced deep learning models to generate high-dimensional feature vectors for each image:
    - **CLIP** (OpenAI): 512-dim (Base) or 768-dim (Large) vectors with text-image understanding
    - **DINOv3** (Meta): 1024-dim vectors with superior dense features for visual similarity
    - These vectors represent a deep, semantic understanding of the image's content, making them highly robust to visual changes.
-   **Database:** A simple and portable **SQLite** database (`image_database.db`) is created in the root image folder to store file paths and their corresponding feature vectors.
-   **Comparison Logic:**
    - **Single-Folder Mode:** Compares all images within the database to find all duplicates
    - **Multi-Folder Mode:** ONLY compares images between Database 1 and Database 2 (skips within-database comparisons)
-   **Memory Efficiency:** To handle hundreds of thousands of images without running out of memory, the script performs a **batched (chunked) similarity search**. It calculates the Cosine Similarity between chunks of images, never loading the entire dataset into VRAM at once.
-   **Similarity Search:** The comparison logic uses GPU-accelerated matrix operations with PyTorch for fast similarity calculations.

### Comparison Logic Visualization:

**Single-Folder Mode:**
```
┌─────────────┐
│ Database A  │
│             │
│  ●───●───●  │ ← Finds all duplicates
│  │ ╲ │ ╱ │  │   within this database
│  ●───●───●  │
└─────────────┘
```

**Multi-Folder Mode:**
```
┌─────────────┐     ┌─────────────┐
│ Database A  │     │ Database B  │
│             │     │             │
│  ●───●───●  │─────│  ●───●───●  │ ← Only compares
│  │ ✗ │ ✗ │  │     │  │ ✗ │ ✗ │  │   BETWEEN databases
│  ●───●───●  │─────│  ●───●───●  │
└─────────────┘     └─────────────┘
     ✗ = Skipped        ✗ = Skipped
```

This design ensures you don't waste time re-scanning for duplicates you've already found within each folder.

## 🤖 Choosing Between CLIP and DINOv3

Both models are excellent for duplicate detection, but they have different strengths:

### CLIP (OpenAI)
**Best for:**
- General-purpose duplicate detection
- When you need text-image understanding
- Multiple embedding options (final, last hidden layer, both)
- Lower VRAM requirements (512-768 dim)

**Models Available:**
- ViT-Base-32: 512-dim, fastest, good quality
- ViT-Large-14: 768-dim, higher quality, slower

### DINOv3 (Meta)
**Best for:**
- **Highest quality duplicate detection** with superior dense features
- Finding subtle duplicates that CLIP might miss
- Outperforms specialized models on visual similarity tasks
- Simple setup (only pooler output, no embedding selection)

**Models Available:**
- ViT-Large: 1024-dim, excellent quality, trained on 1.689 billion images

### Performance Comparison

| Feature | CLIP Base | CLIP Large | DINOv3 Large |
|---------|-----------|------------|--------------|
| Feature Dim | 512 | 768 | 1024 |
| Batch Size | 512 | 450 | 400 |
| Quality | Good | High | Excellent |
| Speed | Fastest | Fast | Fast |
| VRAM Usage | Low | Medium | Medium-High |
| Training Data | Web images | Web images | 1.689B curated images |

### Quick Start with DINOv3

See [QUICKSTART_DINOV3.md](QUICKSTART_DINOV3.md) for detailed instructions on using DINOv3.

**First-time setup:**
```bash
python detector.py
# Select mode [2] or [4] for hashing
# Choose model [3] for DINOv3
# System automatically uses pooler output
# Hash your images and find duplicates!
```

**Switching models:**
The system will detect dimension mismatches and prompt you to re-hash when switching between models.

---
Happy de-duplicating!