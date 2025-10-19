import numpy as np

class linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)*np.sqrt(2.0/input_size)
        self.biases = np.zeros(output_size)

        self.d_weights = None
        self.d_biases = None

    def forward(self, X):
        self.X = X
        output = np.dot(X, self.weights) + self.biases

        return output

    def backward(self, d_output):
        self.d_weights = np.dot(self.X.T, d_output)
        self.d_bias = np.sum(d_output, axis=0)
        d_input = np.dot(d_output, self.weights.T)
        
        return d_input

class ReLU:
    def __init__(self):
        self.X = None
    
    def forward(self, X):
        self.X = X
        output = np.maximum(0, X)

        return output
    
    def backward(self, d_output):
        d_input = d_output.copy()
        d_input[self.X <= 0] = 0

        return d_input
    
class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None
    
    def forward(self, logits, labels):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        #softmax
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits_shifted)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        #one-hot encode the labels
        self.labels = np.zeros((batch_size, num_classes))
        self.labels[np.arange(batch_size), labels] = 1

        #calc cross-entropy
        correct_logprobs = -np.log(self.probs[np.arange(batch_size), labels])
        loss = np.sum(correct_logprobs) / batch_size

        return loss

    def backward(self):
        batch_size = self.probs.shape[0]
        d_logits = (self.probs - self.labels) / batch_size

        return d_logits
    
class SGD:
    #Stohastic Gradient Decent

    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.d_weights
                layer.bias -= self.learning_rate * layer.d_biases
    
    def zero_grad(self):
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                layer.d_weights = None
                layer.d_bias = None