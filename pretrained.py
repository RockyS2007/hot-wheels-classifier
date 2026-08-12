import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import numpy as np
from eval import predict_image

device = "cuda" if torch.cuda.is_available() else "cpu"

load_saved_model = checkpoint = torch.load('hotwheels_resnet18.pt', 
                                            map_location= device)
class_names = load_saved_model['class_names']

model = resnet18(weights=ResNet18_Weights.DEFAULT)
# mimic structure used during training
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

model.load_state_dict(load_saved_model['model_state_dict'])
model.eval()

# see transfer.py for image normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

img_path = 'test_images/raptor_3.jpg'

predict_image(model, class_names, img_path, transform, device)

'''
Works with:
(below are in test_images folder)
- corvette_1
- corvette_2
- lambo_1
- mustang_2
- mustang_3
- raptor_3
(below are in for_fun folder)
- huayra_roadster_1
'''