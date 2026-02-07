import numpy as np

class Linear:
    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            # Use .astype() to convert the array
            self.weights = rng.standard_normal((input_size, output_size)).astype(np.float32) * np.sqrt(2.0 / input_size).astype(np.float32)
        else:
            # Use .astype() here as well
            self.weights = np.random.randn(input_size, output_size).astype(np.float32) * np.sqrt(2.0 / input_size).astype(np.float32)
        
        # Add dtype=np.float32 here
        self.biases = np.zeros(output_size, dtype=np.float32)
        self.d_weights = None
        self.d_biases = None

    def forward(self, X):
        self.X = X
        output = np.dot(X, self.weights) + self.biases

        return output

    def backward(self, d_output):
        self.d_weights = np.dot(self.X.T, d_output)
        self.d_biases = np.sum(d_output, axis=0)
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
                layer.biases -= self.learning_rate * layer.d_biases
    
    def zero_grad(self):
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                layer.d_weights = None
                layer.d_biases = None


class Adam:
    """Adam optimizer with optional weight decay (AdamW when weight_decay > 0)"""
    
    def __init__(self, parameters, learning_rate=1e-3, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize moment estimates
        self.m = {}
        self.v = {}
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                self.m[id(layer.weights)] = np.zeros_like(layer.weights)
                self.v[id(layer.weights)] = np.zeros_like(layer.weights)
                self.m[id(layer.biases)] = np.zeros_like(layer.biases)
                self.v[id(layer.biases)] = np.zeros_like(layer.biases)

    def step(self):
        self.t += 1
        lr = self.learning_rate
        
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                for param, grad, name in [(layer.weights, layer.d_weights, 'w'),
                                          (layer.biases, layer.d_biases, 'b')]:
                    m = self.m[id(param)]
                    v = self.v[id(param)]
                    
                    m[:] = self.beta1 * m + (1 - self.beta1) * grad
                    v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
                    
                    m_hat = m / (1 - self.beta1 ** self.t)
                    v_hat = v / (1 - self.beta2 ** self.t)
                    
                    update = lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    param -= update
                    
                    # Weight decay (only on weights, not biases)
                    if self.weight_decay > 0 and name == 'w':
                        param -= lr * self.weight_decay * param

    def zero_grad(self):
        for layer in self.parameters:
            if hasattr(layer, 'weights'):
                layer.d_weights = None
                layer.d_biases = None


class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler with optional warmup"""
    
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=0.0):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        self.optimizer.learning_rate = lr
        return lr

    def get_lr(self):
        return self.optimizer.learning_rate


class StepLR:
    """Step learning rate scheduler - decays LR by gamma at specified milestones"""
    
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.current_step = 0

    def step(self):
        self.current_step += 1
        
        # Count how many milestones we've passed
        num_decays = sum(1 for m in self.milestones if self.current_step >= m)
        lr = self.base_lr * (self.gamma ** num_decays)
        
        self.optimizer.learning_rate = lr
        return lr

    def get_lr(self):
        return self.optimizer.learning_rate