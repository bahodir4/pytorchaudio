import torch
from torch import nn
from torchsummary import summary 

device = 'mps' if torch.mps.is_available() else "cpu"
class CNNNetwork(nn.Module):
    def __init__(self,):
        super().__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.con4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        cnn_data = self.con1(input_data)
        cnn_data = self.con2(cnn_data)
        cnn_data = self.con3(cnn_data)
        cnn_data = self.con4(cnn_data)
        cnn_data = self.flatten(cnn_data)
        logits = self.linear(cnn_data)
        prediction = self.softmax(logits)

        return prediction

        
if __name__ == "__main__":
    cnn = CNNNetwork()

    summary(cnn.to(device=device), (1, 64, 64))