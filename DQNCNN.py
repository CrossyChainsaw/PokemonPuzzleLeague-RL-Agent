import torch
import torch.nn as nn

class DQNCNN(nn.Module):
    def __init__(self, action_size, channels=3):
        super(DQNCNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        
        # Adjust the input size after considering stacked frames
        self.fc_input_size = 32 * 15 * 15  # Adjust based on the output of conv layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


