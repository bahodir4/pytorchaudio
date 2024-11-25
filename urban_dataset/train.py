import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from urbandataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ANNOTATIONS_FILE = ''
AUDIO_DIR = ''
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


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

    mel_spectogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
     
    usb = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device)
    train_data_loader = create_data_loader(usb, BATCH_SIZE)
    
    cnn = CNNNetwork().to(device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), 'feed_forward.pth')