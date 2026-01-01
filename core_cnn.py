import numpy as np
import cupy as cp
from cupy.lib.stride_tricks import as_strided

# --- 1. HELPERS ---

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = cp.repeat(cp.arange(field_height), field_width)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(field_width), field_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_strided(x, kernel_size, stride, padding):
    """ Fast Forward: Zero-copy im2col using strides """
    N, C, H, W = x.shape
    p = padding
    if p > 0:
        x_padded = cp.pad(x, ((0,0), (0,0), (p,p), (p,p)), mode='constant')
    else:
        x_padded = x
        
    H_pad, W_pad = x_padded.shape[2], x_padded.shape[3]
    H_out = (H_pad - kernel_size) // stride + 1
    W_out = (W_pad - kernel_size) // stride + 1
    
    ns, cs, hs, ws = x_padded.strides
    view_shape = (N, C, H_out, W_out, kernel_size, kernel_size)
    view_strides = (ns, cs, hs * stride, ws * stride, hs, ws)
    
    x_strided = as_strided(x_padded, shape=view_shape, strides=view_strides)
    return x_strided

def im2col_indices(x, field_height, field_width, padding=1, stride=1, cached_indices=None):
    """ Classic Im2Col: Used for backward pass stability """
    p = padding
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    if cached_indices is not None:
        k, i, j = cached_indices
    else:
        k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

# --- 2. LAYERS ---

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = cp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(cp.float16) * scale
        self.biases = cp.zeros(out_channels, dtype=cp.float16)

        self.d_weights = None
        self.d_biases = None
        
        # Cache
        self.cached_indices = None
        self.cached_input_shape = None
        self.input = None  # <--- ΑΠΑΡΑΙΤΗΤΟ ΓΙΑ ΤΟ BACKWARD
        self.x_cols = None

    def forward(self, x):
        self.input = x # <--- ΑΠΟΘΗΚΕΥΣΗ ΕΙΣΟΔΟΥ
        N, C, H, W = x.shape
        
        # 1. Strided View (Zero Copy Fast Forward)
        windows = im2col_strided(x, self.kernel_size, self.stride, self.padding)
        
        # 2. Tensordot
        w_32 = self.weights.astype(cp.float32)
        win_32 = windows.astype(cp.float32)
        out = cp.tensordot(w_32, win_32, axes=([1, 2, 3], [1, 4, 5]))
        
        # 3. Transpose & Reshape
        out = out.transpose(1, 0, 2, 3)
        out += self.biases.astype(cp.float32).reshape(1, -1, 1, 1)
        
        return out.astype(cp.float16)

    def backward(self, d_output):
        # Υπολογίζουμε τους δείκτες τώρα, αν δεν υπάρχουν
        if self.cached_indices is None or self.cached_input_shape != self.input.shape:
             self.cached_indices = get_im2col_indices(
                 self.input.shape, 
                 self.kernel_size, 
                 self.kernel_size, 
                 self.padding, 
                 self.stride
             )
             self.cached_input_shape = self.input.shape

        # Classic im2col για υπολογισμό gradients (πιο ασφαλές)
        self.x_cols = im2col_indices(
            self.input, 
            self.kernel_size, 
            self.kernel_size, 
            self.padding, 
            self.stride, 
            cached_indices=self.cached_indices
        )
        
        n_filter, d_filter, h_filter, w_filter = self.weights.shape
        d_output_reshaped = d_output.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        
        # dWeights
        self.d_weights = cp.dot(d_output_reshaped, self.x_cols.T).reshape(self.weights.shape)
        
        # dBiases
        self.d_biases = cp.sum(d_output, axis=(0, 2, 3))
        
        # dInput
        weights_reshape = self.weights.reshape(n_filter, -1)
        d_x_cols = cp.dot(weights_reshape.T, d_output_reshaped)
        
        d_x = self._col2im(d_x_cols)
        return d_x

    def _col2im(self, cols):
        k, i, j = self.cached_indices
        N, C, H, W = self.cached_input_shape
        
        H_padded, W_padded = H + 2 * self.padding, W + 2 * self.padding
        x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        
        cols_reshaped = cols.reshape(C * self.kernel_size * self.kernel_size, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        
        cp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        
        if self.padding == 0: return x_padded
        return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

class MaxPool2d:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x_shape = None
        self.x_cols = None
        self.max_idx = None
        
    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        h_out = (H - self.kernel_size) // self.stride + 1
        w_out = (W - self.kernel_size) // self.stride + 1
        
        x_reshaped = x.reshape(N * C, 1, H, W)
        self.x_cols = im2col_indices(x_reshaped, self.kernel_size, self.kernel_size, padding=0, stride=self.stride)
        
        self.max_idx = cp.argmax(self.x_cols, axis=0)
        out = self.x_cols[self.max_idx, range(self.max_idx.size)]
        
        out = out.reshape(h_out, w_out, N, C).transpose(2, 3, 0, 1)
        return out

    def backward(self, d_out):
        d_out_flat = d_out.transpose(2, 3, 0, 1).ravel()
        dX_col = cp.zeros_like(self.x_cols)
        dX_col[self.max_idx, range(self.max_idx.size)] = d_out_flat
        
        # Manual col2im for pooling (simple logic)
        k, i, j = get_im2col_indices((self.x_shape[0]*self.x_shape[1], 1, self.x_shape[2], self.x_shape[3]), 
                                     self.kernel_size, self.kernel_size, padding=0, stride=self.stride)
        
        x_padded = cp.zeros((self.x_shape[0]*self.x_shape[1], 1, self.x_shape[2], self.x_shape[3]), dtype=dX_col.dtype)
        cols_reshaped = dX_col.reshape(1 * self.kernel_size * self.kernel_size, -1, self.x_shape[0]*self.x_shape[1])
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        
        cp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        
        return x_padded.reshape(self.x_shape)

class Flatten:
    def __init__(self):
        self.x_shape = None
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, d_out):
        return d_out.reshape(self.x_shape)
    
class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        
        # Parameters (Gamma/Scale initialized to 1, Beta/Shift to 0)
        self.gamma = cp.ones((1, num_features, 1, 1), dtype=cp.float16)
        self.beta = cp.zeros((1, num_features, 1, 1), dtype=cp.float16)
        
        # Gradients
        self.d_weights = None # d_gamma
        self.d_biases = None  # d_beta
        
        # Running stats (Tracked in float32 for stability)
        self.running_mean = cp.zeros((1, num_features, 1, 1), dtype=cp.float32)
        self.running_var = cp.ones((1, num_features, 1, 1), dtype=cp.float32)
        
        self.cache = None
        self.training = True # Default to training mode

    def forward(self, x):
        # x is (N, C, H, W)
        if self.training:
            # Calculate in float32 to avoid NaN/Inf
            x_f32 = x.astype(cp.float32)
            mean = cp.mean(x_f32, axis=(0, 2, 3), keepdims=True)
            var = cp.var(x_f32, axis=(0, 2, 3), keepdims=True)
            
            # Normalize
            x_norm = (x_f32 - mean) / cp.sqrt(var + self.eps)
            x_norm = x_norm.astype(cp.float16) # Cast back
            
            # Update running stats (Exponential Moving Average)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.cache = (x, x_norm, mean, var)
        else:
            # Inference: Use stored stats
            x_f32 = x.astype(cp.float32)
            x_norm = (x_f32 - self.running_mean) / cp.sqrt(self.running_var + self.eps)
            x_norm = x_norm.astype(cp.float16)

        # Scale and Shift (gamma/beta are float16)
        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        # We need the cached values from the forward pass
        x, x_norm, mean, var = self.cache
        N, C, H, W = x.shape
        
        # 1. Gradients for Gamma and Beta
        self.d_weights = cp.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True) # dGamma
        self.d_biases = cp.sum(dout, axis=(0, 2, 3), keepdims=True)           # dBeta
        
        # 2. Gradient for Input (x)
        # Using float32 for gradient calculation stability
        dout_f32 = dout.astype(cp.float32)
        gamma_f32 = self.gamma.astype(cp.float32)
        x_norm_f32 = x_norm.astype(cp.float32)
        
        d_norm = dout_f32 * gamma_f32
        ivar = 1.0 / cp.sqrt(var + self.eps)
        
        # Standard BN Backward formula
        dx = (1. / (N * H * W)) * ivar * (
            (N * H * W) * d_norm 
            - cp.sum(d_norm, axis=(0, 2, 3), keepdims=True) 
            - x_norm_f32 * cp.sum(d_norm * x_norm_f32, axis=(0, 2, 3), keepdims=True)
        )
        
        return dx.astype(cp.float16)
    
class GlobalAvgPool2d:
    def __init__(self):
        self.x_shape = None
        
    def forward(self, x):
        self.x_shape = x.shape # (N, C, H, W)
        # Average over H and W (axis 2 and 3)
        # Output shape becomes (N, C) - essentially "flattened" but much smaller
        return cp.mean(x, axis=(2, 3))

    def backward(self, d_out):
        # d_out shape is (N, C)
        N, C, H, W = self.x_shape
        
        # The gradient of an average is 1/N. 
        # We need to distribute the gradient (N, C) back to (N, C, H, W).
        
        grad = d_out.reshape(N, C, 1, 1) # Add spatial dims back
        grad = grad / (H * W)            # Divide by number of pixels averaged
        
        # Broadcast copies the value to all H*W pixels
        return cp.broadcast_to(grad, self.x_shape).astype(cp.float16)


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
        
    def forward(self, x):
        if self.training:
            # Create mask: 1 with prob (1-p), 0 with prob p
            # We scale by 1/(1-p) here so we don't have to scale at test time
            self.mask = (cp.random.rand(*x.shape) > self.p) / (1.0 - self.p)
            return x * self.mask.astype(x.dtype)
        else:
            return x

    def backward(self, d_out):
        # Apply the same mask to the gradients
        return d_out * self.mask