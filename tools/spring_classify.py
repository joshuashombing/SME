import numpy as np
import torch
import torchvision.transforms as transforms


class SpringClassifier:
    def __init__(self, model_path='spring_classify.pth', device="cpu"):
        # Load the complete model
        self.model = torch.load(model_path)
        self.model.eval()
        self.device = device
        self.model.to(self.device)

        # Define the transformation for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame: np.ndarray):
        # Preprocess the frame
        input_tensor = self.transform(frame).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # Perform classification on the frame
        prediction = predicted.item()

        # return True if get the spring
        return prediction == 1
