import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from time import time
from typing import Union, Callable


class Muon(optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Args:
        learning_rate (float or callable): The learning rate.
        momentum (float, optional): The momentum strength. Default: ``0.95``
        weight_decay (float, optional): The weight decay (L2 penalty). Default: ``0.01``
        nesterov (bool, optional): Enables Nesterov momentum. Default: ``True``
        ns_steps (int, optional): Number of Newton-Schulz iteration steps. Default: ``5``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        super().__init__()
        
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["v"] = mx.zeros_like(parameter)

    def _zeropower_via_newtonschulz5(self, G, steps: int):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
        """
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.astype(mx.bfloat16)
        transpose_needed = G.shape[-2] > G.shape[-1]
        
        if transpose_needed:
            X = X.T
        
        # Ensure spectral norm is at most 1
        norm = mx.sqrt(mx.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-7)
        X = X / norm
        
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        
        if transpose_needed:
            X = X.T
        return X

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Muon parameter update"""
        
        # Apply weight decay
        if self.weight_decay != 0:
            gradient = gradient + self.weight_decay * parameter
        
        # Update momentum buffer
        v = self.momentum * state.get("v")
        v = v + (1 - self.momentum) * gradient
        state["v"] = v
        
        # Get effective gradient
        if self.nesterov:
            effective_grad = gradient * self.momentum + v * (1 - self.momentum)
        else:
            effective_grad = v
        
        # For tensors with fewer than 2 dimensions, skip Newton-Schulz
        if effective_grad.ndim < 2:
            orthogonalized_grad = effective_grad
            scale_factor = 1.0
        else:
            # Save original shape for 4D conv filters
            original_shape = effective_grad.shape
            reshape_needed = effective_grad.ndim > 2
            
            if reshape_needed:
                effective_grad = mx.reshape(effective_grad, (effective_grad.shape[0], -1))
            
            # Apply Newton-Schulz orthogonalization
            orthogonalized_grad = self._zeropower_via_newtonschulz5(effective_grad, steps=self.ns_steps)
            
            # Reshape back if needed
            if reshape_needed:
                orthogonalized_grad = mx.reshape(orthogonalized_grad, original_shape)
            
            # Calculate scaling factor
            scale_factor = max(1, parameter.shape[-2] / parameter.shape[-1]) ** 0.5
        
        return parameter - self.learning_rate.astype(gradient.dtype) * orthogonalized_grad * scale_factor
    


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a dummy dataset
def create_dummy_mnist_dataset(num_samples=1000, batch_size=32):
    # Create random images (784 = 28x28)
    x = np.random.rand(num_samples, 784).astype(np.float32)
    # Create random labels (0-9)
    y = np.random.randint(0, 10, size=(num_samples,)).astype(np.int32)
    
    # Convert to MLX arrays
    x = mx.array(x)
    y = mx.array(y)
    
    # Create batches
    num_batches = num_samples // batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_x = x[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        batches.append((batch_x, batch_y))
    
    return batches

# Fixed loss function - ensure it returns a scalar
def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

# Evaluation function
def evaluate(model, batches):
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in batches:
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        
        # Calculate accuracy
        predictions = mx.argmax(logits, axis=1)
        correct += mx.sum(predictions == y)
        total += len(y)
        total_loss += loss * len(y)
    
    return total_loss / total, correct / total

# Training function
def train(model, optimizer, train_batches, val_batches=None, epochs=5):
    # Create a function to compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    for epoch in range(epochs):
        start_time = time()
        total_loss = 0
        num_batches = len(train_batches)
        
        for i, (x, y) in enumerate(train_batches):
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, x, y)
            
            # Update parameters
            optimizer.update(model, grads)
            
            total_loss += loss
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{num_batches}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        
        # Evaluate on validation set if provided
        if val_batches:
            val_loss, val_acc = evaluate(model, val_batches)
            print(f"Epoch {epoch+1}/{epochs} completed in {time() - start_time:.2f}s - "
                  f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} completed in {time() - start_time:.2f}s - "
                  f"Train Loss: {avg_loss:.4f}")

# Create model and dataset
model = MLP()
train_batches = create_dummy_mnist_dataset(num_samples=5000, batch_size=64)
val_batches = create_dummy_mnist_dataset(num_samples=1000, batch_size=64)

# Initialize the Muon optimizer
optimizer = optim.Muon(
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True,
    ns_steps=3
)

# Train the model
print("Training with Muon optimizer...")
train(model, optimizer, train_batches, val_batches, epochs=8)

# Compare with SGD for reference
print("\nTraining with standard SGD optimizer for comparison...")
model_sgd = MLP()  # Fresh model
optimizer_sgd = optim.SGD(
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True
)
train(model_sgd, optimizer_sgd, train_batches, val_batches, epochs=8)