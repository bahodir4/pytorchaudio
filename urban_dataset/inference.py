import torch 
from cnn import CNNNetwork
from urbandataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES, MelSpectrogram, device


class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]



def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(input)

        predicted_index = prediction[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target.item()]

    return predicted, expected


if __name__ == '__main__':
    cnn = CNNNetwork().to(device)
    state_dict = torch.load('cnn.pth', map_location=device)
    
    cnn.load_state_dict(state_dict)
    
     # load urban sound dataset dataset
    mel_spectrogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            )
    input, target = usd[0][0], usd[0][1]
    input = input.unsqueeze(0).unsqueeze(1).to(device)  # Shape: [1, 1, n_mels, time_frames]
    target = torch.tensor(target).to(device)
    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f'Predicted: {predicted}, expected: {expected}')
