import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    X = X / 255.0
    print("Dataset loaded and normalized.")
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return X_train, y_train, X_test, y_test

def show_image(image, label):
    image = image.reshape(28,28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.savefig("mnist_image.png")

if __name__ == "__main__":
    X_train, y_train, _, _ = load_mnist()
    image_index = 839
    print(f"Displaying image at index {image_index}")
    show_image(X_train[image_index], y_train[image_index])