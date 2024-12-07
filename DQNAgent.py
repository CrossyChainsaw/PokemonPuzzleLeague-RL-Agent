import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torchvision import transforms

from ReplayBuffer import ReplayBuffer
from DQNCNN import DQNCNN


STACK_SIZE = 1

class DQNAgent:
    def __init__(self, DQNCNN:DQNCNN, action_size:int):
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size=10000) 
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_frequency = 4

        # Create two networks: one for the current Q-function and one for the target Q-function
        self.q_network = DQNCNN
        self.target_network = DQNCNN
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        # Define the loss function
        self.loss_fn = nn.MSELoss()  # Add this line to define the loss function
        # Copy weights from the current network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, explore=True):
        # Combine stack_size and channels dimensions into a single dimension for CNN input
        state = np.array(state)  # Ensure state is a numpy array
        state = state.reshape(-1, state.shape[2], state.shape[3])  # Merge stack_size * channels
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():  # Turn off gradients during evaluation
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def random_action(self):
        return random.randrange(self.action_size)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from the replay memory
        batch = self.memory.sample(self.batch_size)

        # Unzip the batch (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)

        # Reshape states and next_states to [batch_size, stack_size * channels, height, width]
        states = states.reshape(states.shape[0], -1, states.shape[3], states.shape[4])  # [batch_size, stack_size * channels, height, width]
        next_states = next_states.reshape(next_states.shape[0], -1, next_states.shape[3], next_states.shape[4])

        # Convert to torch tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get target Q values
        next_q_values = self.target_network(next_states).max(1)[0]

        # Compute the target Q values for the current batch
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss between predicted Q values and target Q values
        loss = self.loss_fn(q_values, target_q_values.detach())  # Detach target_q_values to prevent gradient flow

        # Backpropagate and update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
