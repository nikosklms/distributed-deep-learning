import pickle
import numpy as np
import os

# Διαδρομή όπου κατέβηκαν τα αρχεία (από το download_cifar.py)
DATA_DIR = "data/cifar-10-python"

def unpickle(file):
    """Βοηθητική συνάρτηση για διάβασμα των αρχείων του CIFAR"""
    with open(file, 'rb') as fo:
        # Το encoding='bytes' είναι απαραίτητο για συμβατότητα με Python 3
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10():
    print("Loading CIFAR-10 dataset...")
    
    # Το CIFAR-10 είναι σπασμένο σε 5 αρχεία εκπαίδευσης (data_batch_1 ... 5)
    # και 1 αρχείο τεστ (test_batch)
    
    train_data = []
    train_labels = []
    
    # 1. Φόρτωση Training Batches
    for i in range(1, 6):
        filename = os.path.join(DATA_DIR, f"data_batch_{i}")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing file: {filename}. Run download_cifar.py first!")
            
        batch = unpickle(filename)
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']
        
    # Ενωση των λιστών σε numpy arrays
    X_train = np.concatenate(train_data)
    y_train = np.array(train_labels)
    
    # 2. Φόρτωση Test Batch
    test_batch = unpickle(os.path.join(DATA_DIR, "test_batch"))
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    # 3. Reshape & Normalize
    # Το CIFAR είναι (N, 3072) -> Θέλουμε (N, 3, 32, 32)
    # Η σειρά στο CIFAR είναι Channel-First (RGB), οπότε το reshape είναι απλό.
    
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)
    
    # Μετατροπή σε float16 (για ταχύτητα και μνήμη, όπως κάναμε στο MNIST)
    # Διαιρούμε με 255.0 για να πάνε οι τιμές στο [0, 1]
    X_train = (X_train / 255.0).astype(np.float16)
    X_test = (X_test / 255.0).astype(np.float16)
    
    # Τα labels πρέπει να είναι int32 ή int64 για το CrossEntropy
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    print(f"CIFAR-10 Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Test ότι δουλεύει
    X, y, Xt, yt = load_cifar10()
    print("Mean value:", X.mean())