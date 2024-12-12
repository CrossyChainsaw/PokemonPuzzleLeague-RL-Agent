import torch
import torch.nn as nn

class DQNCNN(nn.Module):
    def __init__(self, action_size, channels=3, pooled_height=12, pooled_width=6):
        super(DQNCNN, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=(5, 10), stride=(5, 10))
        self.fc = nn.Linear(in_features=channels * pooled_height * pooled_width, out_features=action_size)
    
    def forward(self, x):
        x = self.pool(x)
        print(f"Shape after pooling: {x.shape}")  # Should be (batch_size, 3, 12, 6)
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        print(f"Shape after flattening: {x.shape}")  # Should be (batch_size, 216)
        x = self.fc(x)
        return x
    
    def check_pooling(self, x):
        pooled_img = self.pool(x)
        return pooled_img

