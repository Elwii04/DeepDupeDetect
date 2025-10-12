import torch
import numpy as np
import sqlite3
from pathlib import Path
from tqdm import tqdm
import re
import os
import sys
import textwrap
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='timm.models.layers')

# --- ALGORITHM UPGRADE: Using OpenAI CLIP ---
import open_clip

# --- UI UPGRADE: Using Matplotlib for the viewer ---
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# --- Configuration ---
DB_NAME = "image_database.db"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# --- ALGORITHM UPGRADE: New strict threshold for CLIP model ---
# 1.0 is identical. 0.98 is very strict to avoid false positives.
# You can try 0.95 or 0.96 if you find it's missing some duplicates.
SIMILARITY_THRESHOLD = 0.96

# Model configurations
CLIP_MODELS = {
    'large': {'name': 'ViT-L-14', 'pretrained': 'openai', 'feature_dim': 768, 'batch_size': 450},
    'base': {'name': 'ViT-B-32', 'pretrained': 'openai', 'feature_dim': 512, 'batch_size': 512}
}

# Default settings (will be overridden by user choice)
CURRENT_MODEL_CONFIG = CLIP_MODELS['large']  # Default to large model
FEATURE_DIMENSION = CURRENT_MODEL_CONFIG['feature_dim']
USE_LAST_HIDDEN_LAYER = False  # Default to final embedding

NUM_WORKERS = os.cpu_count() - 3 if os.cpu_count() > 1 else 0


# --- UI/UX BUG FIX ---
class ReviewerGUI:
    def __init__(self, duplicate_groups):
        self.groups = duplicate_groups
        self.current_group_index = 0
        
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.fig.canvas.manager.set_window_title('Duplicate Viewer - Navigate with Arrow Keys')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def run(self):
        if not self.groups:
            print("No duplicate groups to review.")
            return
        self._display_group()
        plt.show()

    def _on_key_press(self, event):
        if event.key == 'right': self._next_group()
        elif event.key == 'left': self._prev_group()

    def _display_group(self):
        self.fig.clf()
        group = self.groups[self.current_group_index]
        num_images = len(group)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        self.fig.suptitle(f'Duplicate Set {self.current_group_index + 1} of {len(self.groups)}', fontsize=16)
        try:
            files_with_sizes = [(p, Path(p).stat().st_size) for p in group]
            files_with_sizes.sort(key=lambda x: x[1], reverse=True)
            file_to_keep = files_with_sizes[0][0]
        except FileNotFoundError:
            print(f"Warning: A file in group {self.current_group_index+1} was not found. Skipping.")
            return
        for i, (path, size) in enumerate(files_with_sizes):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            try:
                img = Image.open(Path(path)).convert("RGB")
                img.thumbnail((400, 400))
                ax.imshow(img)
            except Exception: ax.text(0.5, 0.5, "Error loading image", ha='center', va='center')
            title = textwrap.fill(Path(path).name, width=40)
            color = 'green' if path == file_to_keep else 'red'
            ax.set_title(title, color=color, fontsize=8)
            plt.setp(ax.spines.values(), color=color, linewidth=4)
            ax.set_xticks([]); ax.set_yticks([])
        self._add_buttons()
        # --- FIX: Wrap tight_layout in a try...except to prevent crashes from warnings ---
        try:
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        except Exception:
            pass # Ignore tight_layout errors on complex figures
        self.fig.canvas.draw_idle()

    def _add_buttons(self):
        ax_prev = plt.axes([0.2, 0.05, 0.25, 0.075]); self.btn_prev = Button(ax_prev, '<- Previous')
        self.btn_prev.on_clicked(lambda e: self._prev_group())
        ax_next = plt.axes([0.55, 0.05, 0.25, 0.075]); self.btn_next = Button(ax_next, 'Next ->')
        self.btn_next.on_clicked(lambda e: self._next_group())

    def _next_group(self):
        if self.current_group_index < len(self.groups) - 1:
            self.current_group_index += 1
            self._display_group()

    def _prev_group(self):
        if self.current_group_index > 0:
            self.current_group_index -= 1
            # BUG FIX: Corrected method name from display_group to _display_group
            self._display_group()


# --- HIGHER PERFORMANCE: Custom Dataset Class for DataLoader ---
class ImageDataset(Dataset):
    def __init__(self, root_path, image_paths_absolute, transform):
        self.root_path = root_path
        self.image_paths = image_paths_absolute
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        abs_path = self.image_paths[idx]
        full_path = Path(abs_path)
        try:
            img = Image.open(full_path).convert('RGB')
            # The transform (including converting to tensor) happens here
            return self.transform(img), abs_path
        except Exception:
            # If an image is corrupt, return None so it can be filtered out
            return None, None

def collate_fn(batch):
    # Custom collate function to filter out failed images (None)
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    images, paths = zip(*batch)
    return torch.stack(images, 0), paths



class ImageDeduplicator:
    # All other methods (init, initialize_model, find_and_remove, etc.) are the same.
    # Only generate_hashes and _get_image_files are updated.
    def __init__(self, root_folder, secondary_folder=None):
        self.root_path = Path(root_folder).resolve()
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Error: The specified folder does not exist: {self.root_path}")
        
        # Multi-folder support
        self.secondary_path = Path(secondary_folder).resolve() if secondary_folder else None
        if self.secondary_path and not self.secondary_path.is_dir():
            raise FileNotFoundError(f"Error: The specified secondary folder does not exist: {self.secondary_path}")
        
        self.is_multi_folder_mode = self.secondary_path is not None
        
        # Primary database connection
        self.db_path = self.root_path / DB_NAME
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._initialize_database()
        
        # Secondary database connection (if in multi-folder mode)
        self.secondary_conn = None
        self.secondary_cursor = None
        if self.is_multi_folder_mode:
            self.secondary_db_path = self.secondary_path / DB_NAME
            if not self.secondary_db_path.exists():
                raise FileNotFoundError(f"Error: Database not found in secondary folder: {self.secondary_db_path}")
            self.secondary_conn = sqlite3.connect(self.secondary_db_path)
            self.secondary_cursor = self.secondary_conn.cursor()
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device.upper()}")
        if 'cuda' not in self.device: print("Warning: CUDA not found. Processing will be significantly slower.")
        self.model = None; self.preprocess = None
        # Model configuration will be set when user chooses
        self.model_config = None
        self.use_last_hidden_layer = None
        self.save_both_embeddings = False
        self.last_hidden_dim = None
        self.final_embedding_dim = None

    def _get_model_choice(self):
        """Get user's choice for CLIP model and embedding type."""
        print("\n--- CLIP Model Selection ---")
        print("Available models:")
        print("  [1] ViT-Large-14 (768-dim, higher quality, slower) [DEFAULT]")
        print("  [2] ViT-Base-32 (512-dim, faster, lower quality)")
        
        while True:
            try:
                choice = input("Select model (1-2) [1]: ").strip()
                if choice == '' or choice == '1':
                    self.model_config = CLIP_MODELS['large']
                    break
                elif choice == '2':
                    self.model_config = CLIP_MODELS['base']
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")
        
        print(f"Selected: {self.model_config['name']} ({self.model_config['feature_dim']}-dimensional)")
        
        print("\n--- Embedding Type Selection ---")
        print("Embedding types:")
        print("  [1] Final embedding only (for duplicate detection) [DEFAULT]")
        print("  [2] Last hidden layer only (for training/classification)")
        print("  [3] Save both embeddings (dual hash columns)")
        
        while True:
            try:
                choice = input("Select embedding type (1-3) [1]: ").strip()
                if choice == '' or choice == '1':
                    self.use_last_hidden_layer = False
                    self.save_both_embeddings = False
                    break
                elif choice == '2':
                    self.use_last_hidden_layer = True
                    self.save_both_embeddings = False
                    break
                elif choice == '3':
                    self.use_last_hidden_layer = False  # Will extract both
                    self.save_both_embeddings = True
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter 1, 2, or 3.")
        
        if self.save_both_embeddings:
            embedding_type = "both embeddings (dual hash columns)"
        else:
            embedding_type = "last hidden layer" if self.use_last_hidden_layer else "final embedding"
        print(f"Selected: {embedding_type}")
        
        # --- START OF FIX ---
        # Dynamically set the feature dimension based on model and embedding type
        if self.model_config['name'] == 'ViT-L-14' and self.use_last_hidden_layer:
            # The last hidden layer of ViT-L is 1024-dimensional
            self.model_config['feature_dim'] = 1024
        elif self.model_config['name'] == 'ViT-L-14' and not self.use_last_hidden_layer:
            # The final projected embedding is 768-dimensional
            self.model_config['feature_dim'] = 768
        # Add other conditions here if you use more models, e.g., ViT-B/32
        elif self.model_config['name'] == 'ViT-B-32': # B/32 has a hidden dim of 768 and final dim of 512
             self.model_config['feature_dim'] = 768 if self.use_last_hidden_layer else 512

        # Store dimensions for both embeddings when saving both
        if self.save_both_embeddings:
            if self.model_config['name'] == 'ViT-L-14':
                self.last_hidden_dim = 1024
                self.final_embedding_dim = 768
            elif self.model_config['name'] == 'ViT-B-32':
                self.last_hidden_dim = 768
                self.final_embedding_dim = 512

        # Update global variables for database compatibility
        global FEATURE_DIMENSION, BATCH_SIZE, CURRENT_MODEL_CONFIG
        CURRENT_MODEL_CONFIG = self.model_config
        FEATURE_DIMENSION = self.model_config['feature_dim'] # This will now be correct (1024 or 768)
        BATCH_SIZE = self.model_config['batch_size']
        
        print(f"Configuration: {self.model_config['name']} with {embedding_type}")
        if self.save_both_embeddings:
            print(f"  Last hidden layer: {self.last_hidden_dim}-dim")
            print(f"  Final embedding: {self.final_embedding_dim}-dim")
        else:
            print(f"  Dimensions: {FEATURE_DIMENSION}")
        # --- END OF FIX ---

    def _initialize_database(self):
        # Check if we need to update the database schema
        self.cursor.execute("PRAGMA table_info(images)")
        columns = [row[1] for row in self.cursor.fetchall()]
        
        if not columns:
            # Create new table with dual hash support
            self.cursor.execute("""
                CREATE TABLE images (
                    absolute_path TEXT PRIMARY KEY, 
                    hash BLOB NOT NULL,
                    final_embedding_hash BLOB
                )
            """)
        elif 'final_embedding_hash' not in columns:
            # Add the new column to existing table
            self.cursor.execute("ALTER TABLE images ADD COLUMN final_embedding_hash BLOB")
        
        self.conn.commit()

    def _initialize_model(self):
        if self.model is None:
            if self.model_config is None:
                self._get_model_choice()
            
            print(f"Loading pre-trained model ({self.model_config['name']})...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_config['name'], 
                pretrained=self.model_config['pretrained']
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded.")

    def _extract_features(self, image_batch):
        """Extract features using either final embedding, last hidden layer, or both."""
        with torch.no_grad():
            if self.save_both_embeddings:
                # Extract both embeddings
                # First get the last hidden layer
                x = self.model.visual.conv1(image_batch)  # patch embedding
                x = x.reshape(x.shape[0], x.shape[1], -1)  # flatten patches
                x = x.permute(0, 2, 1)  # change to (batch, seq_len, embed_dim)
                
                # Add class token and positional embeddings
                x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
                x = x + self.model.visual.positional_embedding.to(x.dtype)
                
                # Apply layer norm before transformer
                x = self.model.visual.ln_pre(x)
                
                # Forward through transformer layers
                x = x.permute(1, 0, 2)  # change to (seq_len, batch, embed_dim) for transformer
                x = self.model.visual.transformer(x)
                x = x.permute(1, 0, 2)  # change back to (batch, seq_len, embed_dim)
                
                # Get CLS token (first token) from last layer - this is the "last hidden layer"
                last_hidden_features = x[:, 0, :]  # Shape: (batch, hidden_dim)
                
                # Get final embedding through the projection head
                final_features = self.model.encode_image(image_batch)
                
                # Normalize both features
                last_hidden_features = last_hidden_features / last_hidden_features.norm(dim=-1, keepdim=True)
                final_features = final_features / final_features.norm(dim=-1, keepdim=True)
                
                return last_hidden_features, final_features
                
            elif self.use_last_hidden_layer:
                # Extract last hidden layer (for training/classification)
                # We need to get the raw transformer output before the final projection
                
                # Forward through the visual encoder to get the transformer output
                x = self.model.visual.conv1(image_batch)  # patch embedding
                x = x.reshape(x.shape[0], x.shape[1], -1)  # flatten patches
                x = x.permute(0, 2, 1)  # change to (batch, seq_len, embed_dim)
                
                # Add class token and positional embeddings
                x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
                x = x + self.model.visual.positional_embedding.to(x.dtype)
                
                # Apply layer norm before transformer
                x = self.model.visual.ln_pre(x)
                
                # Forward through transformer layers
                x = x.permute(1, 0, 2)  # change to (seq_len, batch, embed_dim) for transformer
                x = self.model.visual.transformer(x)
                x = x.permute(1, 0, 2)  # change back to (batch, seq_len, embed_dim)
                
                # Get CLS token (first token) from last layer - this is the "last hidden layer"
                features = x[:, 0, :]  # Shape: (batch, hidden_dim)
                
                print(f"DEBUG: Last hidden layer shape: {features.shape}")
                
            else:
                # Standard final embedding (for duplicate detection)
                features = self.model.encode_image(image_batch)
            
            # Normalize features (for single embedding case)
            if not self.save_both_embeddings:
                features = features / features.norm(dim=-1, keepdim=True)
                return features

    def _check_database_compatibility(self):
        """Check if the database is compatible with the current model configuration."""
        self.cursor.execute("SELECT COUNT(*) FROM images")
        count = self.cursor.fetchone()[0]
        
        if count == 0:
            return True  # Empty database is always compatible
        
        # Check for dual hash columns
        self.cursor.execute("PRAGMA table_info(images)")
        columns = [row[1] for row in self.cursor.fetchall()]
        has_dual_columns = 'final_embedding_hash' in columns
        
        if self.save_both_embeddings and not has_dual_columns:
            print(f"\n⚠️  WARNING: Database structure mismatch!")
            print("You selected dual hash mode but the database doesn't have the required columns.")
            print("The database needs to be updated to support dual embeddings.")
            
            user_choice = input("Do you want to re-hash all images with dual embeddings? (y/n): ").lower().strip()
            if user_choice == 'y':
                # Clear the database and update schema
                self.cursor.execute("DELETE FROM images")
                self.conn.commit()
                print("Database cleared. All images will be re-hashed with dual embeddings.")
                return True
            else:
                print("Cannot proceed with dual embeddings on incompatible database.")
                return False
        
        # Check a sample hash to determine the feature dimension (for single embedding modes)
        if not self.save_both_embeddings:
            self.cursor.execute("SELECT hash FROM images LIMIT 1")
            sample_hash = self.cursor.fetchone()[0]
            stored_dim = len(np.frombuffer(sample_hash, dtype=np.float32))
            
            if stored_dim != self.model_config['feature_dim']:
                print(f"\n⚠️  WARNING: Database dimension mismatch!")
                print(f"Database contains {stored_dim}-dimensional features")
                print(f"Selected model produces {self.model_config['feature_dim']}-dimensional features")
                print("This means the database was created with a different model/embedding type.")
                
                user_choice = input("Do you want to re-hash all images with the new configuration? (y/n): ").lower().strip()
                if user_choice == 'y':
                    # Clear the database
                    self.cursor.execute("DELETE FROM images")
                    self.conn.commit()
                    print("Database cleared. All images will be re-hashed.")
                    return True
                else:
                    print("Keeping existing database. Note: Results may be inconsistent.")
                    # Update the global dimension to match the database
                    global FEATURE_DIMENSION
                    FEATURE_DIMENSION = stored_dim
                    self.model_config['feature_dim'] = stored_dim
                    return False
        
        return True

    def _display_current_config(self):
        """Display current model configuration."""
        if self.model_config:
            if self.save_both_embeddings:
                embedding_type = "both embeddings (dual hash columns)"
            else:
                embedding_type = "last hidden layer" if self.use_last_hidden_layer else "final embedding"
            print(f"\nCurrent configuration:")
            print(f"  Model: {self.model_config['name']}")
            print(f"  Embedding: {embedding_type}")
            if self.save_both_embeddings:
                print(f"  Last hidden layer: {self.last_hidden_dim}-dim")
                print(f"  Final embedding: {self.final_embedding_dim}-dim")
            else:
                print(f"  Dimensions: {self.model_config['feature_dim']}")
            print(f"  Batch size: {self.model_config['batch_size']}")
        return True

    def _get_image_files(self, rescan_all):
        print("Scanning for image files...")
        all_files = {p.as_posix() for p in self.root_path.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS}
        print(f"Found {len(all_files)} total image files.")
        if rescan_all:
            print("Re-hashing all images as requested."); return list(all_files)
        self.cursor.execute("SELECT absolute_path FROM images")
        db_files = {row[0] for row in self.cursor.fetchall()}
        new_files = list(all_files - db_files)
        print(f"{len(new_files)} new or changed files to process."); return new_files

    # --- HIGHER PERFORMANCE: This function is now completely rebuilt ---
    def generate_hashes(self, rescan_all):
        self._initialize_model()
        
        # Check database compatibility with current model configuration
        if not rescan_all:
            rescan_all = not self._check_database_compatibility()
        
        image_files_absolute = self._get_image_files(rescan_all)
        
        if not image_files_absolute:
            print("No new images to hash. Database is up to date.")
            return

        # Display current configuration
        self._display_current_config()

        # 1. Create the custom dataset
        dataset = ImageDataset(self.root_path, image_files_absolute, self.preprocess)
        
        # 2. Create the DataLoader to run in parallel
        # Use the batch size from the selected model configuration
        data_loader = DataLoader(
            dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        print(f"Starting hashing with {NUM_WORKERS} parallel workers...")
        embedding_type = "last hidden layer" if self.use_last_hidden_layer else "final embedding"
        print(f"Using {self.model_config['name']} with {embedding_type}")
        
        # 3. Iterate over the DataLoader, which provides batches
        for image_batch, paths in tqdm(data_loader, desc="Hashing Images"):
            if image_batch is None: continue # Skip corrupt batches

            # Batches are already prepared, just move to GPU
            image_batch = image_batch.to(self.device)
            
            # Use the new feature extraction method
            if self.save_both_embeddings:
                last_hidden_features, final_features = self._extract_features(image_batch)
                
                # Save both embeddings to the database
                cpu_last_hidden = last_hidden_features.cpu().numpy()
                cpu_final = final_features.cpu().numpy()
                
                db_entries = [
                    (path, last_hidden_vec.tobytes(), final_vec.tobytes()) 
                    for path, last_hidden_vec, final_vec in zip(paths, cpu_last_hidden, cpu_final)
                ]
                self.cursor.executemany(
                    "INSERT OR REPLACE INTO images (absolute_path, hash, final_embedding_hash) VALUES (?, ?, ?)", 
                    db_entries
                )
            else:
                features = self._extract_features(image_batch)
                
                # Save the processed batch to the database
                cpu_features = features.cpu().numpy()
                db_entries = [(path, vec.tobytes()) for path, vec in zip(paths, cpu_features)]
                self.cursor.executemany("INSERT OR REPLACE INTO images (absolute_path, hash) VALUES (?, ?)", db_entries)
            
            self.conn.commit()

        print("Hashing complete.")


    def clean_database(self):
        """
        Scans the database and removes entries for files that no longer exist on disk.
        """
        print("Verifying database against the file system...", flush=True)
        self.cursor.execute("SELECT absolute_path FROM images")
        # Fetch all paths as a flat list of strings
        all_db_paths = [row[0] for row in self.cursor.fetchall()]

        if not all_db_paths:
            print("Database is empty. Nothing to clean.")
            return

        missing_files = []
        # Use tqdm for a progress bar during the check, as it can be slow on HDDs
        for abs_path in tqdm(all_db_paths, desc="Checking files"):
            full_path = Path(abs_path)
            if not full_path.exists():
                missing_files.append(abs_path)

        if not missing_files:
            print("Database is clean. All file entries are valid.")
            return

        print(f"\nFound {len(missing_files)} entries in the database pointing to deleted files.")
        
        user_input = input("Do you want to permanently remove these ghost entries from the database? (y/n): ").lower().strip()

        if user_input == 'y':
            # Use executemany for efficient bulk deletion
            # The data needs to be a list of tuples, e.g., [('path1',), ('path2',)]
            paths_to_delete = [(path,) for path in missing_files]
            
            self.cursor.executemany("DELETE FROM images WHERE absolute_path = ?", paths_to_delete)
            self.conn.commit()
            
            print(f"Successfully removed {len(missing_files)} ghost entries from the database.")
        else:
            print("Database cleaning aborted.")



    def _get_duplicate_detection_embedding_choice(self):
        """Get user's choice for which embedding to use for duplicate detection when both are available."""
        # Check if both embeddings are available in the database
        self.cursor.execute("PRAGMA table_info(images)")
        columns = [row[1] for row in self.cursor.fetchall()]
        has_dual_columns = 'final_embedding_hash' in columns
        
        if not has_dual_columns:
            # Only single embedding available, use the hash column
            return 'hash', 'single embedding'
        
        # Check if we actually have data in both columns
        self.cursor.execute("SELECT COUNT(*) FROM images WHERE final_embedding_hash IS NOT NULL")
        dual_count = self.cursor.fetchone()[0]
        
        if dual_count == 0:
            # No dual embeddings stored, use the hash column
            return 'hash', 'single embedding'
        
        print("\n--- Duplicate Detection Embedding Selection ---")
        print("Available embeddings for duplicate detection:")
        print("  [1] Last hidden layer embedding (from 'hash' column) [DEFAULT]")
        print("  [2] Final embedding (from 'final_embedding_hash' column)")
        
        while True:
            try:
                choice = input("Select embedding for duplicate detection (1-2) [1]: ").strip()
                if choice == '' or choice == '1':
                    return 'hash', 'last hidden layer'
                elif choice == '2':
                    return 'final_embedding_hash', 'final embedding'
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")

    def find_and_remove_duplicates(self, custom_threshold=None):
        """
        Finds duplicates using a memory-efficient, chunk-vs-chunk GPU comparison.
        This algorithm is designed to handle massive datasets without causing OOM errors
        by never loading the entire dataset into VRAM at once.
        Supports both single-folder and multi-folder (cross-database) duplicate detection.
        
        Args:
            custom_threshold: Optional custom similarity threshold (0.0-1.0). If None, uses the default.
        """
        # --- CONFIGURATION: VRAM USAGE CONTROL ---
        # This is the most important setting. It controls how many hashes are loaded into VRAM at once.
        # A smaller number uses less VRAM but may be slightly slower due to more DB reads.
        # A larger number uses more VRAM but can be faster.
        # For your 32GB GPU, 65536 is a very safe value.
        # 65536 * 65536 * 2 bytes (float16) = ~8.5 GB, which fits easily.
        # You could likely increase this to 90000 or even 120000 for more speed if you wish.
        MEGA_CHUNK_SIZE = 70000

        # Determine the similarity threshold to use
        if custom_threshold is not None:
            similarity_threshold = custom_threshold
            print(f"Using custom similarity threshold: {similarity_threshold}")
        else:
            similarity_threshold = SIMILARITY_THRESHOLD
            print(f"Using default similarity threshold: {similarity_threshold}")

        # Get user's choice of embedding for the operation
        hash_column, embedding_description = self._get_duplicate_detection_embedding_choice()
        print(f"Using {embedding_description} embeddings for duplicate detection.")

        # --- STEP 1: Fast initial queries ---
        # Get the total number of images to process. This is a very fast query.
        print("Getting image count from database(s)...")
        
        # Load data from primary database
        if hash_column == 'final_embedding_hash':
            self.cursor.execute("SELECT COUNT(*) FROM images WHERE final_embedding_hash IS NOT NULL")
        else:
            self.cursor.execute("SELECT COUNT(*) FROM images")
        num_images_primary = self.cursor.fetchone()[0]
        
        # Load data from secondary database if in multi-folder mode
        num_images_secondary = 0
        if self.is_multi_folder_mode:
            if hash_column == 'final_embedding_hash':
                self.secondary_cursor.execute("SELECT COUNT(*) FROM images WHERE final_embedding_hash IS NOT NULL")
            else:
                self.secondary_cursor.execute("SELECT COUNT(*) FROM images")
            num_images_secondary = self.secondary_cursor.fetchone()[0]
        
        num_images = num_images_primary + num_images_secondary

        if num_images < 2:
            print("Not enough images in the database(s) to find duplicates.")
            return

        # Get all file paths. Loading millions of strings is fine for RAM.
        print("Loading all file paths into memory...")
        if hash_column == 'final_embedding_hash':
            self.cursor.execute("SELECT absolute_path FROM images WHERE final_embedding_hash IS NOT NULL")
        else:
            self.cursor.execute("SELECT absolute_path FROM images")
        paths = [row[0] for row in self.cursor.fetchall()]
        
        # Add secondary database paths if in multi-folder mode
        if self.is_multi_folder_mode:
            if hash_column == 'final_embedding_hash':
                self.secondary_cursor.execute("SELECT absolute_path FROM images WHERE final_embedding_hash IS NOT NULL")
            else:
                self.secondary_cursor.execute("SELECT absolute_path FROM images")
            secondary_paths = [row[0] for row in self.secondary_cursor.fetchall()]
            paths.extend(secondary_paths)
        
        # Track which database each image belongs to (for deletion strategy)
        db_source = [1] * num_images_primary + [2] * num_images_secondary if self.is_multi_folder_mode else [1] * num_images
        
        # Determine the number of chunks we'll need to iterate through
        num_mega_chunks = (num_images + MEGA_CHUNK_SIZE - 1) // MEGA_CHUNK_SIZE
        print(f"Dataset of {num_images} images will be processed in {num_mega_chunks} mega-chunks.")
        if self.is_multi_folder_mode:
            print(f"  Database 1: {num_images_primary} images from {self.root_path}")
            print(f"  Database 2: {num_images_secondary} images from {self.secondary_path}")

        # This will store the graph of connections between duplicate images
        adj = [set() for _ in range(num_images)]

        # --- HELPER FUNCTION for loading data in chunks ---
        def _load_chunk_to_gpu(offset, size):
            """Load a chunk of hashes from both databases."""
            hashes_blob = []
            
            # Determine which database(s) to read from based on offset
            if offset < num_images_primary:
                # Read from primary database
                primary_start = offset
                primary_size = min(size, num_images_primary - offset)
                
                query = f"SELECT {hash_column} FROM images WHERE {hash_column} IS NOT NULL LIMIT ? OFFSET ?"
                self.cursor.execute(query, (primary_size, primary_start))
                hashes_blob.extend([row[0] for row in self.cursor.fetchall()])
                
                # If we need more, read from secondary database
                if self.is_multi_folder_mode and len(hashes_blob) < size and (offset + size) > num_images_primary:
                    secondary_start = 0
                    secondary_size = (offset + size) - num_images_primary
                    
                    query = f"SELECT {hash_column} FROM images WHERE {hash_column} IS NOT NULL LIMIT ? OFFSET ?"
                    self.secondary_cursor.execute(query, (secondary_size, secondary_start))
                    hashes_blob.extend([row[0] for row in self.secondary_cursor.fetchall()])
            else:
                # Read only from secondary database
                secondary_start = offset - num_images_primary
                secondary_size = size
                
                query = f"SELECT {hash_column} FROM images WHERE {hash_column} IS NOT NULL LIMIT ? OFFSET ?"
                self.secondary_cursor.execute(query, (secondary_size, secondary_start))
                hashes_blob.extend([row[0] for row in self.secondary_cursor.fetchall()])
            
            if not hashes_blob:
                return None

            # Determine feature dimension from the first blob
            feature_dim = len(np.frombuffer(hashes_blob[0], dtype=np.float32))
            
            # Convert raw bytes to a NumPy array, then to a GPU tensor with float16
            hashes_np = np.frombuffer(b''.join(hashes_blob), dtype=np.float32).reshape(-1, feature_dim)
            return torch.from_numpy(hashes_np).to(self.device, non_blocking=True).half()

        # --- STEP 2: The Core Double-Loop Chunk-vs-Chunk Algorithm ---
        # In multi-folder mode, calculate chunks separately for each database
        if self.is_multi_folder_mode:
            num_chunks_primary = (num_images_primary + MEGA_CHUNK_SIZE - 1) // MEGA_CHUNK_SIZE
            num_chunks_secondary = (num_images_secondary + MEGA_CHUNK_SIZE - 1) // MEGA_CHUNK_SIZE
            total_comparisons = num_chunks_primary * num_chunks_secondary
            print(f"Multi-folder mode: Only comparing BETWEEN databases (skipping within-database duplicates)")
            print(f"  Database 1: {num_chunks_primary} chunks")
            print(f"  Database 2: {num_chunks_secondary} chunks")
            print(f"  Total cross-database comparisons: {total_comparisons}")
        else:
            total_comparisons = num_mega_chunks * (num_mega_chunks + 1) // 2
        
        pbar = tqdm(total=total_comparisons, desc="Comparing Chunks")

        for i in range(num_mega_chunks):
            offset_i = i * MEGA_CHUNK_SIZE
            size_i = min(MEGA_CHUNK_SIZE, num_images - offset_i)
            
            # Load the first chunk (chunk_i)
            tensor_i = _load_chunk_to_gpu(offset_i, size_i)

            for j in range(i, num_mega_chunks):
                pbar.set_description(f"Comparing chunk {i+1}/{num_mega_chunks} vs {j+1}/{num_mega_chunks}")
                
                offset_j = j * MEGA_CHUNK_SIZE
                size_j = min(MEGA_CHUNK_SIZE, num_images - offset_j)

                # --- MULTI-FOLDER MODE: Skip comparisons within same database ---
                if self.is_multi_folder_mode:
                    # Determine which database each chunk belongs to
                    chunk_i_db = 1 if offset_i < num_images_primary else 2
                    chunk_j_db = 1 if offset_j < num_images_primary else 2
                    
                    # Skip if both chunks are from the same database
                    if chunk_i_db == chunk_j_db:
                        pbar.update(1)
                        continue

                if i == j:
                    # Case 1: Comparing a chunk against itself (finds intra-chunk duplicates)
                    # This case is skipped in multi-folder mode due to the check above
                    tensor_j = tensor_i
                    # Create the similarity matrix on the GPU
                    similarity = tensor_j @ tensor_j.T
                    # Use torch.triu to get the upper triangle, ignoring the diagonal.
                    # This avoids self-comparisons (imgA vs imgA) and duplicate pairs (A-B vs B-A).
                    rows, cols = torch.where(torch.triu(similarity, diagonal=1) >= similarity_threshold)
                else:
                    # Case 2: Comparing chunk_i against a different chunk_j
                    tensor_j = _load_chunk_to_gpu(offset_j, size_j)
                    if tensor_j is None: continue
                    
                    # Create the similarity matrix for the two different chunks
                    similarity = tensor_i @ tensor_j.T
                    rows, cols = torch.where(similarity >= similarity_threshold)

                # Move results to CPU and update the adjacency list with ABSOLUTE indices
                rows_cpu = rows.cpu().numpy()
                cols_cpu = cols.cpu().numpy()

                for r, c in zip(rows_cpu, cols_cpu):
                    abs_idx1 = offset_i + r
                    abs_idx2 = offset_j + c
                    adj[abs_idx1].add(abs_idx2)
                    adj[abs_idx2].add(abs_idx1)
                
                # --- CRITICAL MEMORY MANAGEMENT ---
                # Explicitly delete tensors to free up VRAM before the next iteration
                del similarity, rows, cols
                if i != j and tensor_j is not None:
                    del tensor_j
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()
                
                pbar.update(1)

            # We are done comparing tensor_i against all subsequent tensors
            del tensor_i
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
        
        pbar.close()

        # --- STEP 3: Find connected groups (same as before) ---
        # The rest of the function operates on the `adj` list, which is now fully populated.
        # This part of the code does not consume significant memory.
        print("Connecting duplicate groups...")
        seen = set()
        duplicate_groups = []
        for i in range(num_images):
            if i in seen: continue
            
            component = []
            q = [i]
            seen.add(i)
            
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                component.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
            
            if len(component) > 1:
                group_paths = [paths[j] for j in component]
                duplicate_groups.append(group_paths)

        # --- The rest of the function for reviewing and deleting remains unchanged ---
        if not duplicate_groups:
            print("No duplicate images found with the current threshold.")
            return

        # --- MULTI-FOLDER MODE: Get deletion strategy ---
        deletion_strategy = 'largest'  # Default
        preferred_db = None
        
        if self.is_multi_folder_mode:
            print("\n--- Deletion Strategy Selection ---")
            print("How would you like to handle duplicates across the two folders?")
            print("  [1] Keep largest files (delete smaller duplicates from either folder) [DEFAULT]")
            print(f"  [2] Keep files from Database 1 ({self.root_path})")
            print(f"  [3] Keep files from Database 2 ({self.secondary_path})")
            
            while True:
                try:
                    choice = input("Select deletion strategy (1-3) [1]: ").strip()
                    if choice == '' or choice == '1':
                        deletion_strategy = 'largest'
                        break
                    elif choice == '2':
                        deletion_strategy = 'prefer_db'
                        preferred_db = 1
                        break
                    elif choice == '3':
                        deletion_strategy = 'prefer_db'
                        preferred_db = 2
                        break
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")
                except ValueError:
                    print("Invalid input. Please enter 1, 2, or 3.")
            
            if deletion_strategy == 'largest':
                print("Selected: Keep largest files (current behavior)")
            else:
                print(f"Selected: Keep all files from Database {preferred_db}, delete duplicates from the other")

        files_to_delete_absolute = set()
        for group in duplicate_groups:
            try:
                if deletion_strategy == 'largest':
                    # Original behavior: keep the largest file
                    files_with_sizes = [(p, Path(p).stat().st_size) for p in group]
                    files_with_sizes.sort(key=lambda x: x[1], reverse=True)
                    for path_to_del, size in files_with_sizes[1:]:
                        files_to_delete_absolute.add(path_to_del)
                else:
                    # New behavior: keep files from preferred database
                    files_with_db = []
                    for p in group:
                        path_idx = paths.index(p)
                        files_with_db.append((p, db_source[path_idx]))
                    
                    # Check if preferred database has any files in this group
                    has_preferred = any(db == preferred_db for _, db in files_with_db)
                    
                    if has_preferred:
                        # Keep all files from preferred DB, delete from other DB
                        for path, db in files_with_db:
                            if db != preferred_db:
                                files_to_delete_absolute.add(path)
                    else:
                        # If preferred DB has no files in this group, fall back to keeping largest
                        files_with_sizes = [(p, Path(p).stat().st_size) for p, _ in files_with_db]
                        files_with_sizes.sort(key=lambda x: x[1], reverse=True)
                        for path_to_del, size in files_with_sizes[1:]:
                            files_to_delete_absolute.add(path_to_del)
            except FileNotFoundError:
                print(f"Warning: A file was not found while analyzing a group. It will be skipped.")
                continue

        print(f"\nFound {len(duplicate_groups)} sets of duplicates using {embedding_description}.")
        print(f"A total of {len(files_to_delete_absolute)} files are marked for deletion.")
        print("Launching visual reviewer (close window to continue)...")

        gui = ReviewerGUI(duplicate_groups)
        gui.run()

        if not files_to_delete_absolute:
            print("No files marked for deletion.")
            return

        print("\n--- Deletion Confirmation ---")
        confirm_input = input(f"CONFIRM: You are about to permanently delete {len(files_to_delete_absolute)} marked files. This cannot be undone. Are you sure? (y/n): ").lower().strip()
        
        if confirm_input == 'y':
            deleted_count = 0
            for abs_path in tqdm(list(files_to_delete_absolute), desc="Deleting Files"):
                try:
                    # Determine which database this file belongs to
                    path_idx = paths.index(abs_path)
                    file_db = db_source[path_idx]
                    
                    # Delete from the appropriate database
                    if file_db == 1:
                        self.cursor.execute("DELETE FROM images WHERE absolute_path = ?", (abs_path,))
                        self.conn.commit()
                    else:
                        self.secondary_cursor.execute("DELETE FROM images WHERE absolute_path = ?", (abs_path,))
                        self.secondary_conn.commit()
                    
                    # Delete the actual file
                    Path(abs_path).unlink()
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {abs_path}: {e}")
            
            print(f"Successfully deleted {deleted_count} files.")
            self._generate_summary(duplicate_groups, list(files_to_delete_absolute))
        else:
            print("Deletion cancelled by user.")

    def _generate_summary(self, initial_groups, deleted_files_absolute):
        # This function is unchanged
        print("\n" + "="*25 + " Deletion Summary " + "="*25)
        print(f"Total duplicate sets found: {len(initial_groups)}")
        cross_folder_sets = 0
        for group in initial_groups:
            parent_folders = {Path(p).parent for p in group}
            if len(parent_folders) > 1: cross_folder_sets += 1
        print(f"  - Sets within a single folder: {len(initial_groups) - cross_folder_sets}")
        print(f"  - Sets spanning multiple folders: {cross_folder_sets}")
        print("-" * 68)
        print(f"Total files deleted: {len(deleted_files_absolute)}")
        deletions_by_folder = defaultdict(int)
        for abs_path in deleted_files_absolute:
            parent_folder = Path(abs_path).parent.as_posix()
            deletions_by_folder[parent_folder] += 1
        if deletions_by_folder:
            print("\nDeletions by subfolder:")
            for folder, count in sorted(deletions_by_folder.items()):
                print(f"  - {folder}: {count} file(s)")
        print("="*68)
    
    def run_filename_pass(self):
        print("Starting filename-based duplicate scan...", flush=True)
        pattern = re.compile(r"^(.*?)(\s*-\s*\d+)(\..+)$")
        potential_dupes = {}
        
        # Collect all files (not just images) from the root path
        # It's important to iterate over the generator explicitly to ensure it's not empty before tqdm
        all_files_list = []
        for p in self.root_path.rglob('*'):
            if p.is_file():
                all_files_list.append(p)
        
        print(f"DEBUG: Finished collecting file paths. Found {len(all_files_list)} files.", flush=True)
        
        if not all_files_list: # If no files found by rglob and is_file() filter
            print("No files found by `rglob('*').is_file()`. Skipping filename scan.", flush=True)
            return

        # Use the collected list for tqdm
        for f in tqdm(all_files_list, desc="Scanning filenames", leave=True): # leave=True keeps the bar on screen
            match = pattern.match(f.name)
            if match:
                base_name, _, ext = match.groups()
                original_file_name = f"{base_name}{ext}"
                original_path = f.parent / original_file_name
                
                if original_path.exists():
                    if original_path not in potential_dupes: potential_dupes[original_path] = []
                    potential_dupes[original_path].append(f)
        
        print("DEBUG: Filename scanning loop completed.", flush=True)

        if not potential_dupes:
            print("No duplicates found based on filename patterns. Returning to main menu.", flush=True); return
        
        print(f"\nFound {len(potential_dupes)} sets of potential duplicates by filename.")
        user_input = input("Do you want to review and delete these files? (y/n): ").lower().strip()
        if user_input != 'y':
            print("Aborting deletion process.", flush=True); return
            
        files_to_delete = []
        print("\n--- Reviewing Files to Delete ---", flush=True)
        for original, dupes in potential_dupes.items():
            print("-" * 30, flush=True); print(f"[KEEPING] -> {original}", flush=True)
            for d in dupes:
                print(f"[MARKING] -> {d}", flush=True); files_to_delete.append(d)
        
        if not files_to_delete:
            print("No files were marked for deletion.", flush=True); return

        print("-" * 30, flush=True)
        confirm_input = input(f"CONFIRM: Delete {len(files_to_delete)} marked files? (y/n): ").lower().strip()
        if confirm_input == 'y':
            deleted_count = 0
            for f in tqdm(files_to_delete, desc="Deleting Files", leave=True):
                try: os.remove(f); deleted_count += 1
                except OSError as e: print(f"Error deleting {f}: {e}", flush=True)
            print(f"Deleted {deleted_count} files.", flush=True)
        else:
            print("Deletion cancelled.", flush=True)

    def close(self):
        self.conn.close()
        if self.secondary_conn:
            self.secondary_conn.close()

# --- Simplified main loop with clearer prompts ---
def get_folder_path(prompt="Please enter the full path to your image folder: "):
    while True:
        folder_path = input(prompt).strip()
        if Path(folder_path).is_dir(): return folder_path
        else: print("Error: The path you entered is not a valid directory. Please try again.")

def get_run_mode(is_multi_folder=False):
    print("\nPlease select a mode to run:")
    
    if is_multi_folder:
        # Simplified menu for multi-folder mode
        print("  [1] Quick Filename Cleanup")
        print("  [2] Find & Remove Duplicates (with Visual Reviewer)")
        print("  [3] Exit")
        
        while True:
            try:
                choice = int(input("Enter your choice (1-3): ").strip())
                if choice == 1: return 1
                elif choice == 2: return 3  # Map to option 3 (find duplicates)
                elif choice == 3: return 6  # Map to option 6 (exit)
                else: print("Invalid choice. Please enter a number between 1 and 3.")
            except ValueError: print("Invalid input. Please enter a number.")
    else:
        # Full menu for single-folder mode
        print("  [1] Quick Filename Cleanup")
        print("  [2] Generate/Update Image Hashes (will ask for model selection)")
        print("  [3] Find & Remove Duplicates (with Visual Reviewer)")
        print("  [4] Full Workflow (Hash then Remove - will ask for model selection)")
        print("  [5] Clean/Verify Database (removes entries for deleted files)")
        print("  [6] Add Another Folder Path (for cross-database duplicate detection)")
        print("  [7] Exit")
        
        while True:
            try:
                choice = int(input("Enter your choice (1-7): ").strip())
                if 1 <= choice <= 7: return choice
                else: print("Invalid choice. Please enter a number between 1 and 7.")
            except ValueError: print("Invalid input. Please enter a number.")

def get_custom_threshold():
    """Prompt user for custom similarity threshold."""
    print(f"\n--- Similarity Threshold Configuration ---")
    print(f"Current default threshold: {SIMILARITY_THRESHOLD}")
    print("Range: 0.0 (very loose) to 1.0 (identical)")
    print("Recommended values:")
    print("  0.98-1.0  = Very strict (nearly identical)")
    print("  0.95-0.97 = Strict (very similar)")
    print("  0.90-0.94 = Moderate (similar)")
    print("  0.85-0.89 = Loose (somewhat similar)")
    
    while True:
        choice = input(f"\nUse default ({SIMILARITY_THRESHOLD}) or enter custom value (0.0-1.0) [default]: ").strip()
        
        if choice == '':
            return None  # Use default
        
        try:
            threshold = float(choice)
            if 0.0 <= threshold <= 1.0:
                return threshold
            else:
                print("Error: Threshold must be between 0.0 and 1.0")
        except ValueError:
            print("Error: Invalid number. Please enter a decimal value between 0.0 and 1.0")

def main():
    print("--- Welcome to the GPU-Accelerated Image Deduplicator ---")
    folder = get_folder_path()
    
    # Initialize with single folder
    deduplicator = ImageDeduplicator(folder)
    is_multi_folder = False
    
    try:
        while True:
            mode = get_run_mode(is_multi_folder)
            
            if mode == 1: 
                deduplicator.run_filename_pass()
            elif mode == 2:
                rescan_choice = input("Re-hash ALL images (y) or only process NEW images (n)? [n]: ").lower().strip()
                deduplicator.generate_hashes(rescan_all=(rescan_choice == 'y'))
            elif mode == 3: 
                # Show current config if available
                if deduplicator.model_config:
                    deduplicator._display_current_config()
                # Ask for custom threshold
                custom_threshold = get_custom_threshold()
                deduplicator.find_and_remove_duplicates(custom_threshold=custom_threshold)
            elif mode == 4:
                print("\n--- Running Full Workflow ---")
                rescan_choice = input("Re-hash ALL images (y) or only process NEW images (n)? [n]: ").lower().strip()
                print("\nStep 1: Hashing images...")
                deduplicator.generate_hashes(rescan_all=(rescan_choice == 'y'))
                print("\nStep 2: Finding and removing duplicates...")
                # Ask for custom threshold
                custom_threshold = get_custom_threshold()
                deduplicator.find_and_remove_duplicates(custom_threshold=custom_threshold)
                print("\n--- Full Workflow Complete ---")
            elif mode == 5:
                deduplicator.clean_database()
            elif mode == 6 and not is_multi_folder:
                # Add another folder path
                print("\n--- Adding Second Folder ---")
                print("Note: The second folder must already have a database (image_database.db).")
                print("If you haven't hashed the images in the second folder yet, please do that first.")
                
                add_folder = input("\nDo you want to add a second folder path? (y/n): ").lower().strip()
                if add_folder == 'y':
                    secondary_folder = get_folder_path("Please enter the full path to the second image folder: ")
                    
                    # Check if database exists in secondary folder
                    secondary_db_path = Path(secondary_folder) / DB_NAME
                    if not secondary_db_path.exists():
                        print(f"\nError: No database found in {secondary_folder}")
                        print("Please run the hashing operation on this folder first (in a separate session).")
                        continue
                    
                    # Close current deduplicator and create new one with both folders
                    deduplicator.close()
                    deduplicator = ImageDeduplicator(folder, secondary_folder)
                    is_multi_folder = True
                    print(f"\n✓ Successfully added second folder: {secondary_folder}")
                    print("You can now find duplicates across both databases!")
            elif mode == 6 or mode == 7:
                print("Exiting.")
                break
                
            print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("Closing database connection.")
        deduplicator.close()
        input("Press Enter to exit.")

if __name__ == '__main__':
    # This guard is CRITICAL for multiprocessing on Windows.
    # It ensures that only the main process runs the main() function.
    main()