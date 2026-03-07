import torch
from torchvision import datasets, transforms
from cnn_model import MNISTCNN


def predict_one():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    image, label = test_dataset[0]

    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    print(f"True label: {label}")
    print(f"Predicted : {prediction}")


if __name__ == "__main__":
    predict_one()