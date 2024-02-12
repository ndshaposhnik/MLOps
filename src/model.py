import torch.nn as nn


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.linear1 = nn.Linear(4 * 8 * 8, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.dropout(x)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return x
