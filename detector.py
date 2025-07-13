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
BATCH_SIZE = 512 # CLIP is a bit larger, a smaller batch size is safer

# --- ALGORITHM UPGRADE: New strict threshold for CLIP model ---
# 1.0 is identical. 0.98 is very strict to avoid false positives.
# You can try 0.95 or 0.96 if you find it's missing some duplicates.
SIMILARITY_THRESHOLD = 0.99

# CLIP ViT-B/32 model outputs a 512-dimension vector
FEATURE_DIMENSION = 512

NUM_WORKERS = min(os.cpu_count() - 1, 8) if os.cpu_count() > 1 else 0


# --- UI/UX BUG FIX ---
class ReviewerGUI:
    def __init__(self, root_path, duplicate_groups):
        self.root_path = root_path
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
            files_with_sizes = [(p, (self.root_path / p).stat().st_size) for p in group]
            files_with_sizes.sort(key=lambda x: x[1], reverse=True)
            file_to_keep = files_with_sizes[0][0]
        except FileNotFoundError:
            print(f"Warning: A file in group {self.current_group_index+1} was not found. Skipping.")
            return
        for i, (path, size) in enumerate(files_with_sizes):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            try:
                img = Image.open(self.root_path / path).convert("RGB")
                img.thumbnail((400, 400))
                ax.imshow(img)
            except Exception: ax.text(0.5, 0.5, "Error loading image", ha='center', va='center')
            title = textwrap.fill(path, width=40)
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
    def __init__(self, root_path, image_paths_relative, transform):
        self.root_path = root_path
        self.image_paths = image_paths_relative
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        full_path = self.root_path / rel_path
        try:
            img = Image.open(full_path).convert('RGB')
            # The transform (including converting to tensor) happens here
            return self.transform(img), rel_path
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
    def __init__(self, root_folder):
        self.root_path = Path(root_folder).resolve()
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Error: The specified folder does not exist: {self.root_path}")
        self.db_path = self.root_path / DB_NAME; self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor(); self._initialize_database()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device.upper()}")
        if 'cuda' not in self.device: print("Warning: CUDA not found. Processing will be significantly slower.")
        self.model = None; self.preprocess = None

    def _initialize_database(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS images (relative_path TEXT PRIMARY KEY, hash BLOB NOT NULL)")
        self.conn.commit()

    def _initialize_model(self):
        if self.model is None:
            print("Loading pre-trained model (OpenAI CLIP ViT-B/32)...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.model.to(self.device); self.model.eval()
            print("Model loaded.")

    def _get_image_files(self, rescan_all):
        print("Scanning for image files...")
        all_files = {p.relative_to(self.root_path).as_posix() for p in self.root_path.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS}
        print(f"Found {len(all_files)} total image files.")
        if rescan_all:
            print("Re-hashing all images as requested."); return list(all_files)
        self.cursor.execute("SELECT relative_path FROM images")
        db_files = {row[0] for row in self.cursor.fetchall()}
        new_files = list(all_files - db_files)
        print(f"{len(new_files)} new or changed files to process."); return new_files

    # --- HIGHER PERFORMANCE: This function is now completely rebuilt ---
    def generate_hashes(self, rescan_all):
        self._initialize_model()
        image_files_relative = self._get_image_files(rescan_all)
        
        if not image_files_relative:
            print("No new images to hash. Database is up to date.")
            return

        # 1. Create the custom dataset
        dataset = ImageDataset(self.root_path, image_files_relative, self.preprocess)
        
        # 2. Create the DataLoader to run in parallel
        # pin_memory=True speeds up CPU-to-GPU transfers
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        print(f"Starting hashing with {NUM_WORKERS} parallel workers...")
        
        # 3. Iterate over the DataLoader, which provides batches
        for image_batch, paths in tqdm(data_loader, desc="Hashing Images"):
            if image_batch is None: continue # Skip corrupt batches

            # Batches are already prepared, just move to GPU
            image_batch = image_batch.to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(image_batch)
                features /= features.norm(dim=-1, keepdim=True)
            
            # Save the processed batch to the database
            cpu_features = features.cpu().numpy()
            db_entries = [(path, vec.tobytes()) for path, vec in zip(paths, cpu_features)]
            self.cursor.executemany("INSERT OR REPLACE INTO images (relative_path, hash) VALUES (?, ?)", db_entries)
            self.conn.commit()

        print("Hashing complete.")


    def clean_database(self):
        """
        Scans the database and removes entries for files that no longer exist on disk.
        """
        print("Verifying database against the file system...", flush=True)
        self.cursor.execute("SELECT relative_path FROM images")
        # Fetch all paths as a flat list of strings
        all_db_paths = [row[0] for row in self.cursor.fetchall()]

        if not all_db_paths:
            print("Database is empty. Nothing to clean.")
            return

        missing_files = []
        # Use tqdm for a progress bar during the check, as it can be slow on HDDs
        for rel_path in tqdm(all_db_paths, desc="Checking files"):
            full_path = self.root_path / rel_path
            if not full_path.exists():
                missing_files.append(rel_path)

        if not missing_files:
            print("Database is clean. All file entries are valid.")
            return

        print(f"\nFound {len(missing_files)} entries in the database pointing to deleted files.")
        
        user_input = input("Do you want to permanently remove these ghost entries from the database? (y/n): ").lower().strip()

        if user_input == 'y':
            # Use executemany for efficient bulk deletion
            # The data needs to be a list of tuples, e.g., [('path1',), ('path2',)]
            paths_to_delete = [(path,) for path in missing_files]
            
            self.cursor.executemany("DELETE FROM images WHERE relative_path = ?", paths_to_delete)
            self.conn.commit()
            
            print(f"Successfully removed {len(missing_files)} ghost entries from the database.")
        else:
            print("Database cleaning aborted.")



    def find_and_remove_duplicates(self):
        print("Loading all hashes from the database...")
        self.cursor.execute("SELECT relative_path, hash FROM images")
        rows = self.cursor.fetchall()
        
        if len(rows) < 2:
            print("Not enough images in the database to find duplicates.")
            return

        paths, hashes_blob = zip(*rows)
        num_images = len(paths)
        hashes = np.frombuffer(b''.join(hashes_blob), dtype=np.float32).reshape(num_images, FEATURE_DIMENSION)
        
        # --- BATCHED SIMILARITY SEARCH TO AVOID MEMORY ERROR ---
        # This new approach processes the search in chunks to avoid creating a massive matrix.
        
        CHUNK_SIZE = 4096 # Process 2048 images at a time. Adjust if you still see memory issues.
        print(f"Finding duplicates for {num_images} images in chunks of {CHUNK_SIZE}...")

        # Step 1: Build a graph of connections (adjacency list)
        # This is more memory-efficient than storing pairs directly.
        adj = [set() for _ in range(num_images)]
        
        for i in tqdm(range(0, num_images, CHUNK_SIZE), desc="Calculating Similarities"):
            chunk_hashes = hashes[i:i+CHUNK_SIZE]
            
            # Compare the chunk against the ENTIRE dataset
            # This creates a manageable similarity matrix of shape (CHUNK_SIZE, num_images)
            similarity_matrix_chunk = chunk_hashes @ hashes.T
            
            # Find pairs within this chunk that are above the threshold
            # np.where gives us the row and column indices of the matches
            row_indices, col_indices = np.where(similarity_matrix_chunk >= SIMILARITY_THRESHOLD)

            for r, c in zip(row_indices, col_indices):
                img_idx1 = i + r # The absolute index of the image from our chunk
                img_idx2 = c      # The absolute index of the image from the full dataset
                
                # Don't connect an image to itself
                if img_idx1 == img_idx2:
                    continue
                
                adj[img_idx1].add(img_idx2)
                adj[img_idx2].add(img_idx1)

        # Step 2: Find connected components in the graph (these are the duplicate groups)
        print("Connecting duplicate groups...")
        seen = set()
        duplicate_groups = []
        for i in range(num_images):
            if i in seen:
                continue
            
            # Start a new group with a Breadth-First Search (BFS) to find all connected images
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
            
            # Only consider groups with more than one image
            if len(component) > 1:
                group_paths = [paths[j] for j in component]
                duplicate_groups.append(group_paths)

        if not duplicate_groups:
            print("No duplicate images found with the current threshold.")
            return

        # Determine which files to mark for deletion before showing the GUI
        files_to_delete_relative = set()
        for group in duplicate_groups:
            try:
                files_with_sizes = [(p, (self.root_path / p).stat().st_size) for p in group]
                files_with_sizes.sort(key=lambda x: x[1], reverse=True)
                for path_to_del, size in files_with_sizes[1:]:
                    files_to_delete_relative.add(path_to_del)
            except FileNotFoundError:
                print(f"Warning: A file was not found while analyzing a group. It will be skipped.")
                continue

        print(f"\nFound {len(duplicate_groups)} sets of duplicates.")
        print(f"A total of {len(files_to_delete_relative)} files are marked for deletion.")
        print("Launching visual reviewer (close window to continue)...")

        gui = ReviewerGUI(self.root_path, duplicate_groups)
        gui.run()

        if not files_to_delete_relative:
            print("No files marked for deletion.")
            return

        print("\n--- Deletion Confirmation ---")
        confirm_input = input(f"CONFIRM: You are about to permanently delete {len(files_to_delete_relative)} marked files. This cannot be undone. Are you sure? (y/n): ").lower().strip()
        
        if confirm_input == 'y':
            deleted_count = 0
            for rel_path in tqdm(list(files_to_delete_relative), desc="Deleting Files"):
                try:
                    self.cursor.execute("DELETE FROM images WHERE relative_path = ?", (rel_path,))
                    (self.root_path / rel_path).unlink()
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {rel_path}: {e}")
            self.conn.commit()
            print(f"Successfully deleted {deleted_count} files.")
            self._generate_summary(duplicate_groups, list(files_to_delete_relative))
        else:
            print("Deletion cancelled by user.")

    def _generate_summary(self, initial_groups, deleted_files_relative):
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
        print(f"Total files deleted: {len(deleted_files_relative)}")
        deletions_by_folder = defaultdict(int)
        for rel_path in deleted_files_relative:
            parent_folder = Path(rel_path).parent.as_posix()
            if parent_folder == '.': parent_folder = '[ROOT FOLDER]'
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

# --- Simplified main loop with clearer prompts ---
def get_folder_path():
    while True:
        folder_path = input("Please enter the full path to your image folder: ").strip()
        if Path(folder_path).is_dir(): return folder_path
        else: print("Error: The path you entered is not a valid directory. Please try again.")

def get_run_mode():
    print("\nPlease select a mode to run:")
    print("  [1] Quick Filename Cleanup")
    print("  [2] Generate/Update Image Hashes")
    print("  [3] Find & Remove Duplicates (with Visual Reviewer)")
    print("  [4] Full Workflow (Hash then Remove)")
    print("  [5] Clean/Verify Database (removes entries for deleted files)")
    print("  [6] Exit")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-6): ").strip())
            if 1 <= choice <= 6: return choice
            else: print("Invalid choice. Please enter a number between 1 and 6.")
        except ValueError: print("Invalid input. Please enter a number.")

def main():
    print("--- Welcome to the GPU-Accelerated Image Deduplicator ---")
    folder = get_folder_path()
    
    deduplicator = ImageDeduplicator(folder)
    
    try:
        while True:
            mode = get_run_mode()
            if mode == 1: deduplicator.run_filename_pass()
            elif mode == 2:
                rescan_choice = input("Re-hash ALL images (y) or only process NEW images (n)? [n]: ").lower().strip()
                deduplicator.generate_hashes(rescan_all=(rescan_choice == 'y'))
            elif mode == 3: deduplicator.find_and_remove_duplicates()
            elif mode == 4:
                print("\n--- Running Full Workflow ---")
                rescan_choice = input("Re-hash ALL images (y) or only process NEW images (n)? [n]: ").lower().strip()
                print("\nStep 1: Hashing images...")
                deduplicator.generate_hashes(rescan_all=(rescan_choice == 'y'))
                print("\nStep 2: Finding and removing duplicates...")
                deduplicator.find_and_remove_duplicates()
                print("\n--- Full Workflow Complete ---")
            elif mode == 5: # This is the new mode
                deduplicator.clean_database()
            elif mode == 6: # Exit is now option 6
                print("Exiting."); break
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