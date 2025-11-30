import numpy as np
import cupy as cp

class LinearGPU:
    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            cp.random.seed(seed)

        self.weights = cp.random.randn(input_size, output_size, dtype=cp.float32) * cp.sqrt(2.0 / input_size)
        self.weights = self.weights.astype(np.float16)
        self.biases = cp.zeros(output_size, dtype=np.float16)

        self.d_weights = None
        self.d_biases = None

    def forward(self, X):
        if isinstance(X, np.ndarray):
            X = cp.asarray(X)

        self.X = X
        output = cp.dot(X, self.weights) + self.biases

        return output

    def backward(self, d_output):
        self.d_weights = cp.dot(self.X.T, d_output)
        self.d_biases = cp.sum(d_output, axis=0)
        d_input = cp.dot(d_output, self.weights.T)
        
        return d_input

class ReLUGPU:
    def __init__(self):
        self.X = None
    
    def forward(self, X):
        self.X = X
        output = cp.maximum(0, X)

        return output
    
    def backward(self, d_output):
        d_input = d_output.copy()
        d_input[self.X <= 0] = 0

        return d_input
    
class CrossEntropyLossGPU:
    def __init__(self):
        self.probs = None
        self.labels = None
    
    def forward(self, logits, labels):
        if isinstance(labels, np.ndarray):
            labels = cp.asarray(labels)

        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        #softmax
        logits_shifted = logits - cp.max(logits, axis=1, keepdims=True)
        exps = cp.exp(logits_shifted)
        self.probs = exps / cp.sum(exps, axis=1, keepdims=True)

        #one-hot encode the labels
        self.labels = cp.zeros((batch_size, num_classes), dtype=cp.float16)
        self.labels[cp.arange(batch_size), labels] = 1

        #calc cross-entropy
        correct_logprobs = -cp.log(self.probs[cp.arange(batch_size), labels] + 1e-9)
        loss = cp.sum(correct_logprobs) / batch_size

        return loss

    def backward(self):
        batch_size = self.probs.shape[0]
        d_logits = (self.probs - self.labels) / batch_size

        return d_logits
    
class SGD_GPU:
    #Stohastic Gradient Decent

    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.d_weights
                layer.biases -= self.learning_rate * layer.d_biases
    
    def zero_grad(self):
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                layer.d_weights = None
                layer.d_biases = None