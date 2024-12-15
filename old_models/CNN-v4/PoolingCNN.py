import torch
import torch.nn as nn

class PoolingCNN(nn.Module):
    def __init__(self, action_size, channels=3, pooled_height=12, pooled_width=6, cursor_input_dim=2):
        super(PoolingCNN, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(5, 10), stride=(5, 10))
        self.fc = nn.Linear(channels * pooled_height * pooled_width + cursor_input_dim, action_size)
    
    def forward(self, x, cursor_positions):
        x = self.pool(x)
        # Flatten image features
        x = torch.flatten(x, start_dim=1)  # Shape: [batch_size, flattened_image_features]
        # Concatenate flattened image features with cursor positions
        combined = torch.cat((x, cursor_positions), dim=1)  # Shape: [batch_size, image_features + cursor_input_dim]
        # Pass through a single fully connected layer to get Q-values
        q_values = self.fc(combined)  # Shape: [batch_size, action_size]
        return q_values

    def check_pooling(self, x):
        pooled_img = self.pool(x)
        return pooled_img