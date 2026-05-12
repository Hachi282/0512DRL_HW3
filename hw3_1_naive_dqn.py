import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque
import copy

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

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Gridworld state is [4, 4, 4] -> flatten to 64
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

def get_state(env):
    state = env.board.render_np().reshape(1, 64)
    return state + np.random.rand(1,64)/10.0 # Add tiny noise to avoid exactly 0 states

def train_naive_dqn():
    print("Training Naive DQN (No Experience Replay)...")
    env = Gridworld(size=4, mode='static')
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    eps = epsilon
    
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='static')
        state = get_state(env)
        state_tensor = torch.from_numpy(state).float()
        status = 1
        step = 0
        
        while status == 1 and step < max_steps:
            qval = model(state_tensor)
            qval_ = qval.data.numpy()
            
            if random.random() < eps:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(qval_)
                
            action_str = action_set[action]
            env.makeMove(action_str)
            reward = env.reward()
            
            next_state = get_state(env)
            next_state_tensor = torch.from_numpy(next_state).float()
            newQ = model(next_state_tensor).data.numpy()
            maxQ = np.max(newQ)
            
            y = np.zeros((1, 4))
            y[:] = qval_[:]
            
            if reward == -10 or reward == 10:
                y[0][action] = reward
                status = 0
            else:
                y[0][action] = reward + gamma * maxQ
                
            y_tensor = torch.from_numpy(y).float()
            
            loss = criterion(qval, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            state_tensor = next_state_tensor
            step += 1
            
        if eps > epsilon_min:
            eps *= epsilon_decay
            
    return losses

def train_replay_dqn():
    print("Training Experience Replay DQN...")
    env = Gridworld(size=4, mode='static')
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    eps = epsilon
    memory = deque(maxlen=memory_size)
    
    for epoch in range(epochs):
        env = Gridworld(size=4, mode='static')
        state = get_state(env)
        status = 1
        step = 0
        
        while status == 1 and step < max_steps:
            state_tensor = torch.from_numpy(state).float()
            qval_ = model(state_tensor).data.numpy()
            
            if random.random() < eps:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(qval_)
                
            action_str = action_set[action]
            env.makeMove(action_str)
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
                    newQ_batch = model(next_state_batch)
                maxQ_batch = torch.max(newQ_batch, dim=1)[0]
                
                Y = reward_batch + gamma * (1 - done_batch) * maxQ_batch
                
                # We only want to update the Q-value for the action that was actually taken
                X = qval_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                
                loss = criterion(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
            state = next_state
            step += 1
            
        if eps > epsilon_min:
            eps *= epsilon_decay
            
    return losses

if __name__ == '__main__':
    losses_naive = train_naive_dqn()
    losses_replay = train_replay_dqn()
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_naive, label='Naive DQN', alpha=0.6)
    plt.plot(losses_replay, label='Experience Replay DQN', alpha=0.6)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Naive vs Experience Replay (Static Mode)')
    plt.savefig('hw3_1_losses.png')
    print("Training finished! Plot saved to hw3_1_losses.png")
