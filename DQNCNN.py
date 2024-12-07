import torch
import torch.nn as nn

class DQNCNN(nn.Module):
    def __init__(self, action_size, stack_size=1, channels=3):
        super(DQNCNN, self).__init__()
        
        # Reduce filter sizes and number of layers for speed
        self.conv1 = nn.Conv2d(stack_size * channels, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        
        # Calculate the feature map size manually for fully connected layers
        self.fc_input_size = 32 * 15 * 15  # Adjust based on output of conv layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
