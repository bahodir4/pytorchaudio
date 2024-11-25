import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeedForwardNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_data):
        flatten_data = self.flatten(input_data)
        logits = self.dense_layer(flatten_data)
        predictions = self.softmax(logits)

        return predictions
    
# download dataset

def download_mnist_download():
    train_data = datasets.MNIST(
        root='data',
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root='data',
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calcu looss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backprogation loss and upt weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss.item()}')


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print('-----------------')
    print('training done') 


if __name__ == '__main__':
    train_data, _ = download_mnist_download()
    

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    print(device)
    feed_forward_net = FeedForwardNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), 'feed_forward.pth')