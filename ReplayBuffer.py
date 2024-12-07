import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        state, action, reward, next_state, done = experience
        
        # Ensure state and next_state are numpy arrays and have the correct shape (STACK_SIZE, height, width, channels)
        state = np.array(state)
        next_state = np.array(next_state)
        
        # Optional: print shapes for debugging purposes
        # print(f"State shape: {state.shape}, Next state shape: {next_state.shape}")
        
        # Add experience to the memory
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
