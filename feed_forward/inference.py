import torch 
from train import FeedForwardNet, download_mnist_download

class_mapping = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'

]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(input)

        predicted_index = prediction[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected


if __name__ == '__main__':

    feed_forward = FeedForwardNet()
    state_dict = torch.load('feed_forward.pth')
    
    feed_forward.load_state_dict(state_dict)
    _, validation = download_mnist_download()

    input, target = validation[0][0], validation[0][1]

    predicted, expected = predict(feed_forward, input, target, class_mapping)

    print(f'Predicted: {predicted}, expected: {expected}')
