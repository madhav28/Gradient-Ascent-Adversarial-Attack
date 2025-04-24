import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img = cv2.imread('correct_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = transform(img).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.maxpool,
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256)
        )
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        identity = self.downsample1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.maxpool(out)
        out = out + identity

        identity = self.downsample2(out)
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = out + identity  

        identity = self.downsample3(out)
        out = self.relu(self.bn5(self.conv5(out)))
        out = self.relu(self.bn6(self.conv6(out)))
        out = out + identity 

        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

best_model = CNN(num_classes=10).to(device)
best_model.load_state_dict(torch.load('best_model_dict.pt'))

for param in best_model.parameters():
    param.requires_grad = False

img_tensor.requires_grad = True

optimizer = optim.SGD([img_tensor], lr=0.01)
criterion = nn.CrossEntropyLoss()
pred = torch.tensor([[0, 0, 0, 1, 0, 
                      0, 0, 0, 0, 0]], dtype=torch.float).to(device)

num_iterations = 10
for i in range(num_iterations):
    optimizer.zero_grad()
    outputs = best_model(img_tensor.to(device))
    if torch.argmax(outputs, dim=1).item() == 3:
        break

    loss = criterion(outputs, pred)
    loss.backward()

    optimizer.step()

    img_tensor.data = torch.clamp(img_tensor.data, 0, 1)
    if (i + 1) % 50 == 0:
        print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item()}")

resulting_image = img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
resulting_image = (resulting_image * 255).astype(np.uint8)
cv2.imwrite('fooling_image.jpg', cv2.cvtColor(resulting_image, cv2.COLOR_RGB2BGR))