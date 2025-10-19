import numpy as np
from load_data import load_mnist
from core import Linear, ReLU, CrossEntropyLoss, SGD

# --- 1. Set up Hyperparameters ---
LEARNING_RATE = 0.1
EPOCHS = 10
BATCH_SIZE = 32  # How many images to process at once

def main(): 
    # --- 2. Load Data ---
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    num_train_samples = X_train.shape[0]
    num_batches = num_train_samples // BATCH_SIZE
    
    print(f"Training on {num_train_samples} samples, {num_batches} batches per epoch.")

    # --- 3. Build the Network ---
    # We build the network as a "list" of layers
    # Input (784) -> Linear(128) -> ReLU -> Linear(10) -> Softmax(Loss)
    
    # This is a list of all layers that have parameters (weights/biases)
    parameters = [] 
    
    layer1 = Linear(input_size=784, output_size=128)
    parameters.append(layer1)
    
    activation1 = ReLU()
    
    layer2 = Linear(input_size=128, output_size=10)
    parameters.append(layer2)
    
    # --- 4. Define Loss and Optimizer ---
    loss_function = CrossEntropyLoss()
    optimizer = SGD(parameters=parameters, learning_rate=LEARNING_RATE)

    # --- 5. The Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        
        # --- Shuffle data at the start of each epoch ---
        # This helps the model learn better
        permutation = np.random.permutation(num_train_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        running_loss = 0.0
        
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # --- 6. FORWARD PASS ---
            out1 = layer1.forward(X_batch)
            out_relu = activation1.forward(out1)
            logits = layer2.forward(out_relu)
            
            # Calculate the loss
            loss = loss_function.forward(logits, y_batch)
            running_loss += loss
            
            # --- 7. BACKWARD PASS ---
            optimizer.zero_grad()
            
            d_logits = loss_function.backward()
            d_layer2 = layer2.backward(d_logits)
            d_relu = activation1.backward(d_layer2)
            d_layer1 = layer1.backward(d_relu)
            
            # --- 8. UPDATE WEIGHTS ---
            optimizer.step()
        
        avg_loss = running_loss / num_batches
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    # --- 10. Test the Network ---
    print("Training finished. Evaluating on test set...")
    
    # Run a forward pass on the *entire* test set
    out1 = layer1.forward(X_test)
    out_relu = activation1.forward(out1)
    logits = layer2.forward(out_relu)
    
    # Get the final predictions (the class with the highest logit)
    # np.argmax finds the *index* of the max value in each row
    predictions = np.argmax(logits, axis=1)
    
    # Calculate accuracy
    # np.mean(predictions == y_test) gives a value from 0 to 1
    accuracy = np.mean(predictions == y_test)
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# This is the standard Python entry point
if __name__ == "__main__":
    main()