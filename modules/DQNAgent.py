import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from modules.ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(self, q_network: nn.Module, action_size: int):
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size=10000)
        self.gamma = 0.9  # Discount factor: How many steps back into the past is valuable
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
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
        # Unpack the state components
        image, cursor_h, cursor_v = state  # Assuming state is [image, cursor_H, cursor_V]
        
        # Convert image to a torch tensor and add a batch dimension
        image_tensor = torch.FloatTensor(image).unsqueeze(0)  # Shape: [1, channels, height, width]
        
        # Combine cursor positions into a tensor and add a batch dimension
        cursor_positions = torch.FloatTensor([[cursor_h, cursor_v]])  # Shape: [1, 2]
        
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():  # Turn off gradients during evaluation
            q_values = self.q_network(image_tensor, cursor_positions)
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

        # Extract image data and cursor positions from the states and next states
        state_images = np.array([s[0] for s in states])  # Get images from current states
        state_cursors = np.array([s[1] for s in states])  # Get cursor positions from current states

        next_state_images = np.array([ns[0] for ns in next_states])  # Get images from next states
        next_state_cursors = np.array([ns[1] for ns in next_states])  # Get cursor positions from next states

        # Convert to torch tensors
        state_images = torch.tensor(state_images, dtype=torch.float32)
        state_cursors = torch.tensor(state_cursors, dtype=torch.float32)
        next_state_images = torch.tensor(next_state_images, dtype=torch.float32)
        next_state_cursors = torch.tensor(next_state_cursors, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get current Q-values by passing both image and cursor data to the Q-network
        q_values = self.q_network(state_images, state_cursors)  # state_images: [batch_size, channels, height, width]
        # print(f'q_values: {q_values}')

        # Step 1: Unsqueeze the actions tensor to add a new dimension at position 1
        actions_expanded = actions.unsqueeze(1)  # Shape becomes [batch_size, 1]
        # print(f'actions_expanded.shape: {actions_expanded.shape}')

        # Step 2: Gather the Q-values for the selected actions using the expanded actions tensor
        selected_q_values = q_values.gather(1, actions_expanded)  # Shape: [batch_size, 1]

        # Step 3: Squeeze to remove the singleton dimension from the result
        q_values_for_actions = selected_q_values.squeeze(1)  # Shape: [batch_size]

        # Get target Q values by passing both image and cursor data of next states to the target network
        next_q_values = self.target_network(next_state_images, next_state_cursors).max(1)[0]

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
