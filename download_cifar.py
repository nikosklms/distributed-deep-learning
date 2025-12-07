import os
import sys
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# URL του επίσημου dataset
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = "data"
FILENAME = "cifar-10-python.tar.gz"
EXTRACT_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")

def download_progress_hook(block_num, block_size, total_size):
    """
    Callback function για να δείχνει την πρόοδο του download.
    """
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 100 / total_size
        
        # Μετατροπή σε MB για να είναι ευανάγνωστο
        downloaded_mb = read_so_far / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        # Δημιουργία μπάρας [=====>    ]
        bar_length = 40
        filled_length = int(bar_length * percent // 100)
        bar = '=' * filled_length + '>' + ' ' * (bar_length - filled_length - 1)
        
        # Εκτύπωση στην ίδια γραμμή (\r)
        sys.stdout.write(f"\rDownloading: [{bar}] {percent:.1f}% ({downloaded_mb:.1f} / {total_mb:.1f} MB)")
        sys.stdout.flush()

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    filepath = os.path.join(DATA_DIR, FILENAME)
    
    # 1. Download
    if not os.path.exists(filepath):
        print(f"Starting download from {URL}...")
        # Περνάμε το hook εδώ
        urllib.request.urlretrieve(URL, filepath, reporthook=download_progress_hook)
        print("\nDownload complete.") # Νέα γραμμή μετά την μπάρα
    else:
        print(f"Archive '{FILENAME}' already exists. Skipping download.")

    # 2. Extract
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting archive...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

def save_sample_images():
    # Φόρτωση του πρώτου batch για δείγματα
    batch_file = os.path.join(EXTRACT_DIR, "data_batch_1")
    
    if not os.path.exists(batch_file):
        print("Error: Batch file not found. Extraction might have failed.")
        return

    with open(batch_file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    data = dict[b'data']
    labels = dict[b'labels']
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nSaving 10 sample images to 'cifar_samples/'...")
    if not os.path.exists("cifar_samples"):
        os.makedirs("cifar_samples")

    for i in range(10):
        # Το CIFAR είναι flat vector (N, 3072). Το κάνουμε reshape σε (3, 32, 32)
        img_flat = data[i]
        
        # Προσοχή: Το CIFAR είναι αποθηκευμένο ως Channel-First (3, 32, 32)
        # RRRRR... GGGGG... BBBBB...
        img_R = img_flat[0:1024].reshape(32, 32)
        img_G = img_flat[1024:2048].reshape(32, 32)
        img_B = img_flat[2048:3072].reshape(32, 32)
        
        # Το Matplotlib θέλει (32, 32, 3) -> Channel-Last
        img = np.dstack((img_R, img_G, img_B))
        
        label_name = classes[labels[i]]
        
        plt.imsave(f"cifar_samples/img_{i}_{label_name}.png", img)
        print(f"Saved img_{i}_{label_name}.png")

if __name__ == "__main__":
    download_and_extract()
    save_sample_images()
    print("\nREADY! Data is prepared in 'data/' folder.")