import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
import pandas as pd
import os

ANNOTATIONS_FILE = ''
AUDIO_DIR = ''
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
device = 'mps' if torch.mps.is_available() else 'cpu'

class UrbanSoundDataset(Dataset):

    def __init__(self, annotation_file, audio_dir, tranformer, sample_rate, num_samples, device):
        self.annotation_file = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir 
        self.device = device
        self.transformation = tranformer.to(self.device)
        self.sample_r = sample_rate
        self.num_samples = num_samples
        

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_padding_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _get_audio_sample_path(self, index):
        try:
            fold = f'fold{self.annotation_file.iloc[index, 5]}'
            path = os.path.join(self.audio_dir, fold, self.annotation_file.iloc[index, 0])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            return path
        except Exception as e:
            raise RuntimeError(f"Error retrieving audio sample path: {e}")


    def _get_audio_sample_label(self, index):
        return self.annotation_file.iloc[index, 6]
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_r:
            resampler = Resample(sr, self.sample_r)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
         
        if signal.shape[0] > 1: 
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_padding_if_necessary(self, signal):
        len_signal = signal.shape[1]
        if len_signal < self.num_samples:
            num_sampling = self.num_samples - len_signal
            last_dim_padding = (0, num_sampling)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal 
    
if __name__ == '__main__':

    mel_spectogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
     
    usb = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram, SAMPLE_RATE, NUM_SAMPLES, device)
    
    print(f'len of usb sample: {len(usb)}')
    signal, label = usb[0]



    