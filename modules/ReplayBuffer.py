import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        state, action, reward, next_state, done = experience

        # Ensure state and next_state are lists with image and cursor components
        state_image, state_cursor_h, state_cursor_v = state
        next_state_image, next_state_cursor_h, next_state_cursor_v = next_state

        # Convert image components to numpy arrays for consistency
        state_image = np.array(state_image)
        next_state_image = np.array(next_state_image)

        # Combine cursor components into arrays
        state_cursor = np.array([state_cursor_h, state_cursor_v])
        next_state_cursor = np.array([next_state_cursor_h, next_state_cursor_v])

        # Add the full experience to the memory
        self.memory.append(((state_image, state_cursor), action, reward, (next_state_image, next_state_cursor), done))

    def sample(self, batch_size):
        # Randomly sample experiences from the memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Return the current size of the memory
        return len(self.memory)
