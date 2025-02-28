import mlx.optimizers as optimizers
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
import numpy as np

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Converted from PyTorch to MLX.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(mx.bfloat16)  # Use astype instead of cast
    if G.shape[-2] > G.shape[-1]:
        X = X.T
    
    # Ensure spectral norm is at most 1
    norm = mx.sqrt(mx.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-7)
    X = X / norm
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    
    if G.shape[-2] > G.shape[-1]:
        X = X.T
    return X

class Muon(optimizers.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    MLX implementation with distributed training support.
    
    Arguments:
        params: The parameters to optimize
        lr: The learning rate used by the internal SGD
        weight_decay: Weight decay coefficient
        momentum: The momentum used by the internal SGD
        nesterov: Whether to use Nesterov-style momentum (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        
        # Initialize momentum buffers
        self.momentum_buffers = [mx.zeros_like(p) for p in self.params]
        
        # Initialize distributed world
        if mx.distributed.is_available():
            self.world = mx.distributed.init()
            self.world_size = self.world.size()  # Call the method with parentheses
            self.rank = self.world.rank()  # Call the method with parentheses
        else:
            self.world_size = 1
            self.rank = 0
        
    def update(self, gradients):
        """Update parameters using Muon optimizer with distributed support"""
        updates = []
        
        for i, (param, grad, buf) in enumerate(zip(self.params, gradients, self.momentum_buffers)):
            # Update momentum buffer
            self.momentum_buffers[i] = buf * self.momentum + grad * (1 - self.momentum)
            
            # Get effective gradient
            if self.nesterov:
                effective_grad = grad * self.momentum + self.momentum_buffers[i] * (1 - self.momentum)
            else:
                effective_grad = self.momentum_buffers[i]
            
            # Reshape for 4D conv filters
            original_shape = effective_grad.shape
            if effective_grad.ndim == 4:
                effective_grad = mx.reshape(effective_grad, (effective_grad.shape[0], -1))
            
            # Apply Newton-Schulz orthogonalization
            orthogonalized_grad = zeropower_via_newtonschulz5(effective_grad, steps=self.ns_steps)
            
            # Reshape back if needed
            if len(original_shape) == 4:
                orthogonalized_grad = mx.reshape(orthogonalized_grad, original_shape)
            
            # Synchronize gradients across all processes if in distributed mode
            if self.world_size > 1:
                # Use all_sum for gradient synchronization
                orthogonalized_grad = mx.distributed.all_sum(orthogonalized_grad) / self.world_size
            
            # Apply weight decay
            param_update = param * (1 - self.lr * self.weight_decay)
            
            # Apply gradient update with scaling
            scale_factor = max(1, param.shape[-2] / param.shape[-1]) ** 0.5
            param_update = param_update - orthogonalized_grad * (self.lr * scale_factor)
            
            updates.append(param_update)
            
        return updates
    
    def step(self, gradients):
        """Perform one optimization step"""
        return self.update(gradients)

# Example usage with distributed training
def train_with_muon(model, optimizer, data_loader, epochs=5):
    # Properly initialize the distributed world
    if mx.distributed.is_available():
        world = mx.distributed.init()
        world_size = world.size()  # Call the method with parentheses
        world_rank = world.rank()  # Call the method with parentheses
    else:
        world_size = 1
        world_rank = 0
    
    def loss_and_grad(model, x, y):
        # Forward pass
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y)
        
        # Backward pass - we need to ensure the loss is a scalar
        grads = mx.grad(lambda m: mx.mean(nn.losses.cross_entropy(m(x), y)))(model)
        
        return loss, grads, logits
    
    # Create a mapping to track which parameters are being optimized
    optimized_params = set(id(p) for p in optimizer.params)
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = len(data_loader)
        
        for i, (x, y) in enumerate(data_loader):
            # Forward and backward pass
            loss, all_grads, logits = loss_and_grad(model, x, y)
            
            # Extract only the gradients for parameters in optimizer.params
            flat_grads = []
            param_to_grad = {}
            
            # First, create a mapping from parameter shapes to gradients
            for module_name, module_grads in all_grads.items():
                for param_name, grad in module_grads.items():
                    shape_key = str(grad.shape)
                    if shape_key not in param_to_grad:
                        param_to_grad[shape_key] = []
                    param_to_grad[shape_key].append((module_name, param_name, grad))
            
            # Now extract gradients for each parameter in optimizer.params
            for param in optimizer.params:
                shape_key = str(param.shape)
                if shape_key in param_to_grad and param_to_grad[shape_key]:
                    # Take the first matching gradient
                    _, _, grad = param_to_grad[shape_key][0]
                    flat_grads.append(grad)
                    # Remove this gradient so we don't use it again
                    param_to_grad[shape_key].pop(0)
                else:
                    # If no matching gradient, use zeros
                    flat_grads.append(mx.zeros_like(param))
            
            # Synchronize gradients across processes
            if world_size > 1:
                flat_grads = [mx.distributed.all_sum(g) / world_size for g in flat_grads]
                loss = mx.distributed.all_sum(loss) / world_size
            
            # Update model parameters
            updates = optimizer.step(flat_grads)
            
            # Apply updates to the model
            param_updates = {}
            for module_name, module_params in model.parameters().items():
                param_updates[module_name] = {}
                for param_name, param in module_params.items():
                    # Skip parameters not being optimized
                    if id(param) not in optimized_params:
                        continue
                    
                    # Find the index of this parameter in optimizer.params
                    idx = next((i for i, p in enumerate(optimizer.params) if id(p) == id(param)), None)
                    if idx is not None:
                        param_updates[module_name][param_name] = updates[idx]
            
            model.update(param_updates)
            
            total_loss += loss
            
            # Calculate and print accuracy periodically
            if i % 10 == 0 or i == num_batches - 1:
                preds = mx.argmax(logits, axis=1)
                accuracy = mx.mean(mx.equal(preds, y))
                if world_rank == 0:
                    print(f"Batch {i+1}/{num_batches}, Loss: {loss}, Accuracy: {accuracy}")
        
        avg_loss = total_loss / num_batches
        if world_rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")

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


# Create model
model = MLP()

# Create optimizer
# Extract parameters for the optimizer
params_dict = model.parameters()
params = []
for module_name, module_params in params_dict.items():
    for param_name, param in module_params.items():
        if param.ndim == 2:  # Only use Muon for 2D parameters (linear layers)
            params.append(param)

optimizer = Muon(params, lr=0.01, momentum=0.9, nesterov=True, ns_steps=5)

# Create dummy dataset
data_loader = create_dummy_mnist_dataset(num_samples=3200, batch_size=64)

# Train model
train_with_muon(model, optimizer, data_loader, epochs=5)