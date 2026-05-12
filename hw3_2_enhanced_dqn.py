import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque

from Gridworld import Gridworld

# Hyperparameters
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999
learning_rate = 1e-3
epochs = 1000
max_steps = 50
batch_size = 32
memory_size = 1000
sync_freq = 50 # How often to update target network

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

def get_state(env):
    state = env.board.render_np().reshape(1, 64)
    return state + np.random.rand(1,64)/10.0

# 1. Double DQN Architecture
class DoubleDQN(nn.Module):
    def __init__(self):
        super(DoubleDQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Dueling DQN Architecture
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.relu1 = nn.ReLU()
        
        # Value stream
        self.val_fc = nn.Linear(150, 100)
        self.val_relu = nn.ReLU()
        self.val_out = nn.Linear(100, 1)
        
        # Advantage stream
        self.adv_fc = nn.Linear(150, 100)
        self.adv_relu = nn.ReLU()
        self.adv_out = nn.Linear(100, 4)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        
        val = self.val_relu(self.val_fc(x))
        val = self.val_out(val)
        
        adv = self.adv_relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val

def train_agent(model_class, is_double=False):
    print(f"Training {model_class.__name__} (is_double={is_double})...")
    env = Gridworld(size=4, mode='player')
    
    model = model_class()
    target_model = model_class()
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    eps = epsilon
    memory = deque(maxlen=memory_size)
    j = 0
    
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='player')
        state = get_state(env)
        status = 1
        step = 0
        
        while status == 1 and step < max_steps:
            j += 1
            state_tensor = torch.from_numpy(state).float()
            qval_ = model(state_tensor).data.numpy()
            
            if random.random() < eps:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(qval_)
                
            env.makeMove(action_set[action])
            reward = env.reward()
            next_state = get_state(env)
            
            if reward == -10 or reward == 10:
                status = 0
                done = True
            else:
                done = False
                
            memory.append((state, action, reward, next_state, done))
            
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                
                state_batch = torch.from_numpy(np.vstack([x[0] for x in minibatch])).float()
                action_batch = torch.tensor([x[1] for x in minibatch])
                reward_batch = torch.tensor([x[2] for x in minibatch]).float()
                next_state_batch = torch.from_numpy(np.vstack([x[3] for x in minibatch])).float()
                done_batch = torch.tensor([x[4] for x in minibatch]).float()
                
                qval_batch = model(state_batch)
                
                with torch.no_grad():
                    if is_double:
                        # Double DQN: Main model selects action, Target model evaluates it
                        next_actions = model(next_state_batch).argmax(dim=1)
                        newQ_batch = target_model(next_state_batch)
                        maxQ_batch = newQ_batch.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else:
                        # Standard Target DQN
                        newQ_batch = target_model(next_state_batch)
                        maxQ_batch = torch.max(newQ_batch, dim=1)[0]
                
                Y = reward_batch + gamma * (1 - done_batch) * maxQ_batch
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
            
        if eps > epsilon_min:
            eps *= epsilon_decay
            
    return losses

if __name__ == '__main__':
    losses_ddqn = train_agent(DoubleDQN, is_double=True)
    losses_dueling = train_agent(DuelingDQN, is_double=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_ddqn, label='Double DQN', alpha=0.6)
    plt.plot(losses_dueling, label='Dueling DQN', alpha=0.6)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Double DQN vs Dueling DQN (Player Mode)')
    plt.savefig('hw3_2_losses.png')
    print("Training finished! Plot saved to hw3_2_losses.png")
