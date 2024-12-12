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


class DQNAgent:
    def __init__(self, q_network: nn.Module, action_size: int):
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_frequency = 4

        # Create two networks: one for the current Q-function and one for the target Q-function
        self.q_network = q_network
        self.target_network = q_network.__class__(action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        # Define the loss function
        self.loss_fn = nn.MSELoss()
        # Copy weights from the current network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, explore=True):
        state = np.array(state)  # Ensure state is a numpy array
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

        # Reshape states and next_states to [batch_size, channels, height, width]
        states = states.reshape(states.shape[0], -1, states.shape[2], states.shape[3])
        print(f'state shape: {states.shape}')
        next_states = next_states.reshape(next_states.shape[0], -1, next_states.shape[2], next_states.shape[3])

        # Convert to torch tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get current Q values
        # Step 1: Get Q-values for all actions from the Q-network
        q_values: torch.Tensor = self.q_network(states)  # Shape: [batch_size, action_size]
        print(f'q_values: {q_values}')

        # Step 2: Unsqueeze the actions tensor to add a new dimension at position 1
        actions_expanded: torch.Tensor = actions.unsqueeze(1)  # Shape: [batch_size, 1]
        print(f'actions_expanded.shape: {actions_expanded.shape}')

        # Step 3: Gather the Q-values for the selected actions using the expanded actions tensor
        selected_q_values: torch.Tensor = q_values.gather(1, actions_expanded)  # Shape: [batch_size, 1]

        # Step 4: Squeeze to remove the singleton dimension from the result
        q_values_for_actions: torch.Tensor = selected_q_values.squeeze(1)  # Shape: [batch_size]


        # Get target Q values
        next_q_values = self.target_network(next_states).max(1)[0]

        # Compute the target Q values for the current batch
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss between predicted Q values and target Q values
        loss = self.loss_fn(q_values_for_actions, target_q_values.detach())

        # Backpropagate and update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
