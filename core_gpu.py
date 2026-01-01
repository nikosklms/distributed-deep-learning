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
        correct_logprobs = -cp.log(self.probs[cp.arange(batch_size), labels] + 1e-4)
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

class AdamW_GPU:
    def __init__(self, master_params, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-2):
        self.master_params = master_params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = {} 
        self.v = {}
        
        for mp in self.master_params:
            if mp.weights is not None:
                self.m[id(mp.weights)] = cp.zeros_like(mp.weights)
                self.v[id(mp.weights)] = cp.zeros_like(mp.weights)
            if mp.biases is not None:
                self.m[id(mp.biases)] = cp.zeros_like(mp.biases)
                self.v[id(mp.biases)] = cp.zeros_like(mp.biases)

    def zero_grad(self):
        for mp in self.master_params:
            if mp.weights is not None: mp.d_weights.fill(0)
            if mp.biases is not None: mp.d_biases.fill(0)

    def step(self):
        self.t += 1
        lr = self.learning_rate
        
        for mp in self.master_params:
            # --- Weights Update ---
            if mp.weights is not None:
                grad = mp.d_weights
                
                # Get moments
                m = self.m[id(mp.weights)]
                v = self.v[id(mp.weights)]
                
                # Adam Logic
                m[:] = self.beta1 * m + (1 - self.beta1) * grad
                v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
                
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                
                # AdamW: Decoupled Weight Decay
                # 1. Standard Adam Update
                update = lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)
                mp.weights -= update
                
                # 2. Weight Decay (applied directly to weights, scaled by LR)
                if self.weight_decay > 0:
                    mp.weights -= lr * self.weight_decay * mp.weights

            # --- Biases Update (Usually no weight decay on biases) ---
            if mp.biases is not None:
                grad = mp.d_biases
                m = self.m[id(mp.biases)]
                v = self.v[id(mp.biases)]
                
                m[:] = self.beta1 * m + (1 - self.beta1) * grad
                v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
                
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                
                mp.biases -= lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)