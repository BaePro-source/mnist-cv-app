import torch
from torchvision import datasets, transforms
from cnn_model import load_model


def predict_one():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    image, label = test_dataset[0]

    model, device = load_model("mnist_cnn.pth")
    image = image.unsqueeze(0).to(device)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    print(f"True label: {label}")
    print(f"Predicted : {prediction}")


if __name__ == "__main__":
    predict_one()