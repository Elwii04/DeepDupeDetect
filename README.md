# GPU-Accelerated AI Duplicate Image Finder

A high-performance, AI-powered tool designed to find duplicate and near-duplicate images within massive collections. This script is capable of processing over 1 million images, leveraging state-of-the-art deep learning models to identify visually similar images, even with variations in resolution, cropping, filters, and color.

![image](https://user-images.githubusercontent.com/12345/your-reviewer-screenshot.png) <!-- It's highly recommended to replace this with a real screenshot of the reviewer GUI! -->

## ✨ Key Features

- **State-of-the-Art Accuracy:** Utilizes OpenAI's powerful **CLIP** model to understand image content on a semantic level, making it incredibly robust against complex transformations.
- **Massive Scalability:** Engineered to handle huge datasets (tested on 700k+, capable of 1M+ images) by using memory-efficient, batched comparison algorithms.
- **Blazing Fast Hashing:** Leverages your NVIDIA GPU and parallel CPU cores with a PyTorch `DataLoader` to rapidly generate image "hashes" (feature vectors).
- **Interactive Visual Reviewer:** Instead of endless terminal lists, a GUI pops up allowing you to visually review duplicate sets side-by-side before confirming deletion.
- **Smart "Keep" Logic:** Automatically marks the largest file in a duplicate set to be kept, assuming higher file size equals higher quality.
- **Intelligent Database:** Uses a persistent SQLite database to store hashes. You only need to hash your collection once; future runs only process new files.
- **Database Maintenance Tools:** Includes built-in utilities to clean up the database by removing entries for files that have been deleted manually.

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

#### Menu Options:

1.  **[1] Quick Filename Cleanup:** A fast first pass that finds duplicates based on common naming patterns (e.g., `image.jpg` and `image-001.jpg`).
2.  **[2] Generate/Update Image Hashes:** The core function. It scans your folders, finds new images not yet in the database, and uses the GPU to compute their AI-based hashes. Run this first on your collection. The script is interruptible and can be resumed later.
3.  **[3] Find & Remove Duplicates (with Visual Reviewer):** After hashing, use this mode to find all duplicate sets. A GUI window will launch, allowing you to review each set before making a final deletion confirmation in the terminal.
4.  **[4] Full Workflow (Hash then Remove):** A convenient option that runs Mode 2 and then Mode 3 sequentially.
5.  **[5] Clean/Verify Database:** An essential maintenance tool. This scans the database and removes "ghost" entries for files that you have deleted outside of this script, keeping your database in sync with your file system.
6.  **[6] Exit:** Closes the application.

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
-   **Optional Speed Boost:** For an extra performance gain during the CPU-bound image loading phase, you can install the `Pillow-SIMD` library, a much faster, drop-in replacement for the standard `Pillow`.
    ```bash
    pip uninstall Pillow
    pip install Pillow-SIMD
    ```

## 🛠️ How It Works (Technical Details)

-   **Hashing:** Instead of simple perceptual hashes, this tool uses OpenAI's **CLIP (ViT-B/32)** model to generate a 512-dimension feature vector for each image. This vector represents a deep, semantic understanding of the image's content, making it highly robust to visual changes.
-   **Database:** A simple and portable **SQLite** database (`image_database.db`) is created in the root image folder to store file paths and their corresponding feature vectors.
-   **Comparison:** To handle hundreds of thousands of images without running out of memory, the script performs a **batched (chunked) similarity search**. It calculates the Cosine Similarity between a small chunk of images and the entire dataset, repeating until all images have been compared.
-   **Similarity Search:** The comparison logic is handled by **scikit-learn**, a robust and standard data science library. *This tool does not use FAISS.*

---
Happy de-duplicating!