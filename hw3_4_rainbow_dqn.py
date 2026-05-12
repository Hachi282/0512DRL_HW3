import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

from Gridworld import Gridworld

# Hyperparameters
gamma = 0.99
learning_rate = 1e-3
epochs = 300
max_steps = 50
batch_size = 32
memory_size = 2000
sync_freq = 100
n_step = 3 # Multi-step return

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def get_state(env):
    state = env.board.render_np().reshape(1, 64)
    return state + np.random.rand(1,64)/10.0

# Noisy Linear Layer for Exploration (Replacing epsilon-greedy)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

# Rainbow Architecture: Dueling + Noisy Nets
class RainbowDQN(nn.Module):
    def __init__(self):
        super(RainbowDQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.relu1 = nn.ReLU()
        
        self.val_fc = NoisyLinear(150, 100)
        self.val_relu = nn.ReLU()
        self.val_out = NoisyLinear(100, 1)
        
        self.adv_fc = NoisyLinear(150, 100)
        self.adv_relu = nn.ReLU()
        self.adv_out = NoisyLinear(100, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        
        val = self.val_relu(self.val_fc(x))
        val = self.val_out(val)
        
        adv = self.adv_relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val
        
    def reset_noise(self):
        self.val_fc.reset_noise()
        self.val_out.reset_noise()
        self.adv_fc.reset_noise()
        self.adv_out.reset_noise()

def train_rainbow():
    print("Training Rainbow DQN (Dueling, Double, Noisy, Multi-step) in Random Mode...")
    model = RainbowDQN()
    target_model = RainbowDQN()
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() # For full Rainbow we'd use distributional KL Divergence, but MSE is easier for Demo
    
    memory = deque(maxlen=memory_size)
    n_step_buffer = deque(maxlen=n_step)
    losses = []
    
    j = 0
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='random')
        state = get_state(env)
        status = 1
        step = 0
        n_step_buffer.clear()
        
        while status == 1 and step < max_steps:
            j += 1
            state_tensor = torch.from_numpy(state).float()
            
            model.reset_noise()
            with torch.no_grad():
                qval_ = model(state_tensor).data.numpy()
            action = np.argmax(qval_) # Noisy net handles exploration
                
            env.makeMove(action_set[action])
            reward = env.reward()
            next_state = get_state(env)
            done = reward == -10 or reward == 10
            
            n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(n_step_buffer) == n_step:
                # Calculate n-step return
                n_reward = sum([gamma**i * n_step_buffer[i][2] for i in range(n_step)])
                n_state = n_step_buffer[0][0]
                n_action = n_step_buffer[0][1]
                n_next_state = n_step_buffer[-1][3]
                n_done = n_step_buffer[-1][4]
                memory.append((n_state, n_action, n_reward, n_next_state, n_done))
            
            if done:
                # Empty remaining n_step_buffer
                while len(n_step_buffer) > 0:
                    n_reward = sum([gamma**i * n_step_buffer[i][2] for i in range(len(n_step_buffer))])
                    n_state = n_step_buffer[0][0]
                    n_action = n_step_buffer[0][1]
                    n_next_state = n_step_buffer[-1][3]
                    n_done = n_step_buffer[-1][4]
                    memory.append((n_state, n_action, n_reward, n_next_state, n_done))
                    n_step_buffer.popleft()
                status = 0
            
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                
                state_batch = torch.from_numpy(np.vstack([x[0] for x in minibatch])).float()
                action_batch = torch.tensor([x[1] for x in minibatch])
                reward_batch = torch.tensor([x[2] for x in minibatch]).float()
                next_state_batch = torch.from_numpy(np.vstack([x[3] for x in minibatch])).float()
                done_batch = torch.tensor([x[4] for x in minibatch]).float()
                
                model.reset_noise()
                target_model.reset_noise()
                
                qval_batch = model(state_batch)
                
                # Double DQN logic with n-step
                with torch.no_grad():
                    next_actions = model(next_state_batch).argmax(dim=1)
                    newQ_batch = target_model(next_state_batch)
                    maxQ_batch = newQ_batch.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                Y = reward_batch + (gamma ** n_step) * (1 - done_batch) * maxQ_batch
                X = qval_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                
                loss = criterion(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
            if j % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())
                
            state = next_state
            step += 1
            
    return losses

if __name__ == '__main__':
    losses_rainbow = train_rainbow()
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_rainbow, label='Rainbow DQN-Lite (Random Mode)', alpha=0.6, color='purple')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Rainbow DQN (Multi-step, Dueling, Double, Noisy Nets)')
    plt.savefig('hw3_4_losses.png')
    print("Training finished! Plot saved to hw3_4_losses.png")
