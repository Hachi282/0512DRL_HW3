import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from Gridworld import Gridworld

# Hyperparameters
gamma = 0.9
learning_rate = 1e-3
batch_size = 32
memory_size = 1000
sync_freq = 50 

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def get_state(env):
    state = env.board.render_np().reshape(1, 64)
    return state + np.random.rand(1,64)/10.0

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        state = torch.from_numpy(np.vstack([x[0] for x in minibatch])).float()
        action = torch.tensor([x[1] for x in minibatch])
        reward = torch.tensor([x[2] for x in minibatch]).float()
        next_state = torch.from_numpy(np.vstack([x[3] for x in minibatch])).float()
        done = torch.tensor([x[4] for x in minibatch]).float()
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class LightningDQN(pl.LightningModule):
    def __init__(self):
        super(LightningDQN, self).__init__()
        
        # Network
        self.net = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
        
        self.target_net = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
        self.target_net.load_state_dict(self.net.state_dict())
        
        # RL specifics
        self.env = Gridworld(size=4, mode='random')
        self.buffer = ReplayBuffer(memory_size)
        self.eps = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.step_count = 0
        self.losses = []
        
        # Pre-fill buffer
        self.populate_buffer()
        
    def forward(self, x):
        return self.net(x)
    
    def populate_buffer(self, steps=100):
        state = get_state(self.env)
        for _ in range(steps):
            action = np.random.randint(0, 4)
            self.env.makeMove(action_set[action])
            reward = self.env.reward()
            next_state = get_state(self.env)
            done = reward == 10 or reward == -10
            self.buffer.push(state, action, reward, next_state, done)
            if done:
                self.env = Gridworld(size=4, mode='random')
                state = get_state(self.env)
            else:
                state = next_state
                
    def training_step(self, batch, batch_idx):
        # We manually collect experience in the training loop
        # Play 1 step in environment
        state = get_state(self.env)
        
        if random.random() < self.eps:
            action = np.random.randint(0, 4)
        else:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            q_values = self(state_tensor)
            action = torch.argmax(q_values).item()
            
        self.env.makeMove(action_set[action])
        reward = self.env.reward()
        next_state = get_state(self.env)
        done = reward == 10 or reward == -10
        
        self.buffer.push(state, action, reward, next_state, done)
        
        if done:
            self.env = Gridworld(size=4, mode='random')
            
        if self.eps > self.epsilon_min:
            self.eps *= self.epsilon_decay
            
        # Sample from buffer
        b_states, b_actions, b_rewards, b_next_states, b_dones = self.buffer.sample(batch_size)
        b_states = b_states.to(self.device)
        b_actions = b_actions.to(self.device)
        b_rewards = b_rewards.to(self.device)
        b_next_states = b_next_states.to(self.device)
        b_dones = b_dones.to(self.device)
        
        # Q-learning target
        q_vals = self(b_states).gather(1, b_actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            target_q_vals = self.target_net(b_next_states).max(1)[0]
        targets = b_rewards + gamma * (1 - b_dones) * target_q_vals
        
        loss = nn.MSELoss()(q_vals, targets)
        self.log('train_loss', loss, prog_bar=True)
        self.losses.append(loss.item())
        
        # Target Network Sync
        self.step_count += 1
        if self.step_count % sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        return [optimizer], [scheduler]
    
    # We use a dummy dataloader since RL gathers data dynamically
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.arange(10000).float(), # 10000 dummy steps
            batch_size=1
        )

if __name__ == '__main__':
    # Initialize PyTorch Lightning Model
    model = LightningDQN()
    
    # Enable Gradient Clipping in Trainer (Training Tip from HW3-3)
    trainer = pl.Trainer(
        max_epochs=1, 
        max_steps=5000,
        gradient_clip_val=1.0, # Gradient Clipping implementation!
        logger=False,
        enable_checkpointing=False
    )
    
    print("Starting training with PyTorch Lightning (includes LR Scheduling & Grad Clipping)...")
    trainer.fit(model)
    
    plt.figure(figsize=(10, 5))
    plt.plot(model.losses, label='Lightning DQN (Random Mode)', alpha=0.6, color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Lightning DQN with Gradient Clipping & StepLR')
    plt.savefig('hw3_3_losses.png')
    print("Training finished! Plot saved to hw3_3_losses.png")
