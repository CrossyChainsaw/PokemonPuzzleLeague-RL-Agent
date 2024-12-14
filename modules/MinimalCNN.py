import torch
import torch.nn as nn
import torch.nn.functional as F

class TestCNN(nn.Module):
    def __init__(self, num_actions=6):
        super(TestCNN, self).__init__()
        
        # Convolutional layers for processing the 32x32 RGB image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        
        # Fully connected layers for the flattened output from the convolutional layers
        self.fc_conv = nn.Linear(32 * 32 * 32, 128)  # Flatten to 1D, 128 output features
        
        # Dense layer for the cursor position input
        self.cursor_fc = nn.Linear(2, 32)  # Processing (x, y) coordinates
        
        # Combined layers
        self.fc_combined1 = nn.Linear(128 + 32, 64)  # Combined feature representation
        self.fc_combined2 = nn.Linear(64, num_actions)  # Output layer for action probabilities

    def forward(self, image, cursor):
        """
        Args:
            image: Tensor of shape (batch_size, 3, 32, 32) representing the game board.
            cursor: Tensor of shape (batch_size, 2) representing the cursor position (x, y).
        
        Returns:
            policy: Tensor of shape (batch_size, num_actions) representing the action probabilities.
        """
        # Process the image through convolutional layers
        x = F.relu(self.conv1(image))   # Conv Layer 1
        x = F.relu(self.conv2(x))       # Conv Layer 2
        x = x.view(x.size(0), -1)       # Flatten the output to 1D
        
        # Process the cursor position through a dense layer
        y = F.relu(self.cursor_fc(cursor))
        
        # Combine the features from the image and cursor
        combined = torch.cat((x, y), dim=1)
        
        # Fully connected layers for the combined features
        z = F.relu(self.fc_combined1(combined))
        policy = F.softmax(self.fc_combined2(z), dim=1)  # Action probabilities with softmax
        
        return policy
