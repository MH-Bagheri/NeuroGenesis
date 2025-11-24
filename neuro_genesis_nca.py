import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
@dataclass
class Config:
    CHANNEL_N: int = 16        # Number of channels (4 for RGBA + 12 hidden states)
    TARGET_PADDING: int = 16   # Padding around the target image
    IMAGE_SIZE: int = 40       # Size of the grid (40x40)
    BATCH_SIZE: int = 8
    POOL_SIZE: int = 1024
    CELL_FIRE_RATE: float = 0.5 # Probability a cell updates per step (stochasticity)
    STEPS_MIN: int = 64        # Min steps to grow
    STEPS_MAX: int = 96        # Max steps to grow
    LR: float = 2e-3           # Learning rate
    ITERATIONS: int = 2000     # Training iterations (kept low for demo, increase for perfection)
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

conf = Config()

# ==========================================
# 1. TARGET PATTERN GENERATION
# ==========================================
def generate_target_emoji():
    """
    Generates a synthetic biological glyph (Alien Artifact) programmatically.
    We use code to generate the image so no external assets are needed.
    """
    size = conf.IMAGE_SIZE
    c_x, c_y = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    
    # 1. Main circle body
    mask = (x - c_x)**2 + (y - c_y)**2 <= (size // 4)**2
    
    # 2. Generate RGBA channels
    target = np.zeros((size, size, 4), dtype=np.float32)
    
    # Gradients for organic look
    target[:, :, 0] = (np.sin(x / 5.0) + 1) / 2  # Red channel wave
    target[:, :, 1] = (np.cos(y / 5.0) + 1) / 2  # Green channel wave
    target[:, :, 2] = 0.8                        # Blue constant
    target[:, :, 3] = 1.0                        # Alpha (Opaque)
    
    # Apply circular mask
    target[~mask] = 0.0
    
    # Add a "nucleus" detail
    nucleus_mask = (x - c_x)**2 + (y - c_y)**2 <= (size // 10)**2
    target[nucleus_mask, 0] = 1.0
    target[nucleus_mask, 1] = 1.0
    target[nucleus_mask, 2] = 0.0
    
    return torch.tensor(target).permute(2, 0, 1).unsqueeze(0).to(conf.DEVICE)

# ==========================================
# 2. THE NEURAL CA MODEL
# ==========================================
class NeuralCA(nn.Module):
    """
    The 'DNA' of the organism.
    It takes the current state of a grid and outputs the update vector.
    It perceives neighbors via a Sobel filter (gradient sensing).
    """
    def __init__(self, channel_n=conf.CHANNEL_N, fire_rate=conf.CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        # Perception Filters (Sobel X and Y)
        # Identifies boundaries and neighbors
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        
        # Stack filters to process all channels
        identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.register_buffer('kernel_x', sobel_x.expand(channel_n, 1, 3, 3))
        self.register_buffer('kernel_y', sobel_y.expand(channel_n, 1, 3, 3))
        self.register_buffer('kernel_id', identity.expand(channel_n, 1, 3, 3))

        # The Neural Network (1x1 Conv acts as a dense layer per pixel)
        # Input: State (16) + SobelX (16) + SobelY (16) = 48 inputs per cell
        self.w1 = nn.Conv2d(channel_n * 3, 128, kernel_size=1)
        self.w2 = nn.Conv2d(128, channel_n, kernel_size=1)

        # Initialize weights to zero to ensure stability at start
        with torch.no_grad():
            self.w2.weight.zero_()
            self.w2.bias.zero_()

    def perception(self, x):
        """Senses the environment (neighbors)"""
        # Grouped convolution to apply filters per channel
        grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.channel_n)
        grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.channel_n)
        # Concatenate: State, Gradient X, Gradient Y
        return torch.cat([x, grad_x, grad_y], dim=1)

    def forward(self, x, steps=1):
        for _ in range(steps):
            pre_life_mask = F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, padding=1, stride=1) > 0.1

            # 1. Perceive
            y = self.perception(x)
            
            # 2. Think (Neural Network)
            y = self.w1(y)
            y = F.relu(y)
            update = self.w2(y)

            # 3. Stochastic Update (Simulate asynchronous cell updates)
            # Create a mask where ~50% of cells fire
            stochastic_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) < self.fire_rate).float()
            
            x = x + update * stochastic_mask

            # 4. Alive Masking (Cells must have a live neighbor or be alive to function)
            # This prevents empty space from spontaneously generating matter
            post_life_mask = F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, padding=1, stride=1) > 0.1
            life_mask = pre_life_mask & post_life_mask
            x = x * life_mask.float()
            
            # Clip to valid range
            x = x.clamp(-10.0, 10.0) 

        return x

# ==========================================
# 3. SAMPLE POOL & UTILS
# ==========================================
class SamplePool:
    """Stores a pool of evolving organisms to maintain diversity."""
    def __init__(self, pool_size, channel_n, height, width):
        self.pool = torch.zeros(pool_size, channel_n, height, width).to(conf.DEVICE)
        self.indices = np.arange(pool_size)
    
    def sample(self, batch_size):
        idx = np.random.choice(self.indices, batch_size, replace=False)
        return self.pool[idx], idx
    
    def commit(self, idx, x):
        self.pool[idx] = x.detach().clone()

def get_living_mask(x):
    """Identify live cells based on Alpha channel."""
    return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

def make_seed(batch_size):
    """Create fresh seeds (empty grids with 1 active center pixel)."""
    seed = torch.zeros(batch_size, conf.CHANNEL_N, conf.IMAGE_SIZE, conf.IMAGE_SIZE).to(conf.DEVICE)
    c = conf.IMAGE_SIZE // 2
    seed[:, 3:, c, c] = 1.0 # Set Alpha to 1
    return seed

def make_damage(x):
    """Injects random damage (holes) into the organisms."""
    batch_n, _, h, w = x.shape
    mask = torch.ones(batch_n, 1, h, w).to(x.device)
    
    # Cut a random circle out of the grid
    for i in range(batch_n):
        cx, cy = np.random.randint(0, h), np.random.randint(0, w)
        r = np.random.randint(3, 8)
        Y, X = np.ogrid[:h, :w]
        dist = (X - cx)**2 + (Y - cy)**2
        mask[i, :, dist <= r**2] = 0.0
        
    return x * mask

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_neurogenesis():
    print(f"--- NeuroGenesis: Initializing on {conf.DEVICE} ---")
    
    # Setup
    target_img = generate_target_emoji()
    model = NeuralCA().to(conf.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=conf.LR)
    pool = SamplePool(conf.POOL_SIZE, conf.CHANNEL_N, conf.IMAGE_SIZE, conf.IMAGE_SIZE)
    
    # Fill pool with initial seeds
    seed_batch = make_seed(conf.POOL_SIZE)
    pool.commit(np.arange(conf.POOL_SIZE), seed_batch)
    
    loss_log = []

    print(f"Target: Synthetic Biological Artifact ({conf.IMAGE_SIZE}x{conf.IMAGE_SIZE})")
    print(f"Training for {conf.ITERATIONS} iterations...")
    
    start_time = time.time()
    
    for i in range(1, conf.ITERATIONS + 1):
        # 1. Sample batch from pool
        inputs, batch_idx = pool.sample(conf.BATCH_SIZE)
        
        # 2. Sort by loss to find the "worst" ones to replace with fresh seeds
        # This prevents the pool from getting stuck in dead states
        with torch.no_grad():
            current_loss = ((inputs[:, :4, :, :] - target_img)**2).mean(dim=[1,2,3])
            worst_indices = torch.argsort(current_loss, descending=True)[:1] # Replace worst 1
            inputs[worst_indices] = make_seed(1) # Reset to seed
        
        # 3. Damage Injection (Teach them to heal!)
        if i % 5 == 0: # Damage every 5th batch
            inputs = make_damage(inputs)

        # 4. Forward Pass (Let them grow)
        optimizer.zero_grad()
        steps = np.random.randint(conf.STEPS_MIN, conf.STEPS_MAX)
        outputs = model(inputs, steps=steps)
        
        # 5. Calculate Loss (MSE on RGBA channels only)
        # We only care that the first 4 channels match the target image
        loss = ((outputs[:, :4, :, :] - target_img)**2).mean()
        
        # 6. Backprop
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 7. Update Pool
        pool.commit(batch_idx, outputs)
        
        loss_log.append(loss.item())
        
        if i % 100 == 0:
            print(f"Iter {i:04d} | Loss: {loss.item():.6f} | Steps: {steps}")

    print(f"Training complete in {time.time() - start_time:.2f}s")
    return model, loss_log, target_img

# ==========================================
# 5. VISUALIZATION & DEMO
# ==========================================
def run_demo(model, target_img):
    """
    Runs a visual simulation of the trained model:
    1. Growth from seed
    2. Severe Damage
    3. Regeneration
    """
    print("\n--- Generating Animation ---")
    
    model.eval()
    
    # Create a fresh seed
    x = make_seed(1)
    
    frames = []
    
    # Phase 1: Growth (0 to 100 steps)
    for _ in range(60):
        with torch.no_grad():
            x = model(x, steps=2)
            # Convert to numpy image [H, W, 4]
            img = x[0, :4, :, :].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            frames.append(img)
            
    # Phase 2: DAMAGE! (Cut half the image off)
    h, w = conf.IMAGE_SIZE, conf.IMAGE_SIZE
    mask = torch.ones_like(x)
    mask[:, :, h//2:, :] = 0.0 # Wipe bottom half
    x = x * mask
    
    # Add flash frame to indicate damage
    frames.append(np.ones((h, w, 4))) 
    
    # Phase 3: Regeneration (100 to 200 steps)
    for _ in range(60):
        with torch.no_grad():
            x = model(x, steps=2)
            img = x[0, :4, :, :].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            frames.append(img)
            
    # Setup Matplotlib Animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Target View
    target_np = target_img[0].permute(1, 2, 0).cpu().numpy()
    ax1.imshow(target_np)
    ax1.set_title("Target DNA")
    ax1.axis('off')
    
    # Live Simulation View
    im = ax2.imshow(frames[0])
    ax2.set_title("NeuroGenesis Simulation")
    ax2.axis('off')
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        if frame_idx < 60:
            ax2.set_xlabel("State: Growing")
        elif frame_idx == 60:
             ax2.set_xlabel("State: !TRAUMA!")
        else:
            ax2.set_xlabel("State: Regenerating")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    
    print("Displaying animation window...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Check for dependencies
    try:
        model, history, target = train_neurogenesis()
        run_demo(model, target)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")