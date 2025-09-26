import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def set_seed(seed):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

def make_dataloaders(root_dir, TARGET_IMAGE_SIZE, trainStatus):
    mean, std = [0.4850, 0.4949, 0.4120], [0.2442, 0.2419, 0.2658]
    train_transform = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.RandomCrop(size=TARGET_IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_transform2 = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_batch_size = 100
    test_batch_size = 1
    if trainStatus == "train":
        train_dataset_1 = datasets.ImageFolder(root_dir, transform=train_transform)
        train_dataset_2 = datasets.ImageFolder(root_dir, transform=train_transform2)
        train_loader_1 = DataLoader(train_dataset_1, batch_size=train_batch_size, shuffle=True)
        train_loader_2 = DataLoader(train_dataset_2, batch_size=train_batch_size, shuffle=True)
        return train_loader_1, train_loader_2
    elif trainStatus == "test":
        test_dataset = datasets.ImageFolder(root_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        return test_loader
    return None

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(15488, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn9 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.SiLU()
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool3(self.relu(self.bn4(self.conv4(x))))
        x = self.pool3(self.relu(self.bn5(self.conv5(x))))
        x = self.pool3(self.relu(self.bn6(self.conv6(x))))
        x = self.pool3(self.relu(self.bn7(self.conv7(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn8(self.fc1(x)))
        x = self.relu(self.bn9(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def train_model(model, device, criterion, optimizer, scheduler,train_loader_1, train_loader_2, savePath, epochs=30):
    for _ in range(1, epochs + 1):
        model.train()
        for train_loader in [train_loader_1, train_loader_2]:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        path = savePath
        torch.save(model.state_dict(), path)
        scheduler.step()

def test_model(model, test_loader, device, output_csv='bird.csv'):
    model.eval()
    results = []
    with torch.no_grad():  
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            results.extend(predicted.cpu().numpy())
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])

if __name__ == "__main__":
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "default/model/path"
    TARGET_IMAGE_SIZE = (64, 64)
    set_seed(0)
    if trainStatus == "test":
        test_loader = make_dataloaders(dataPath, TARGET_IMAGE_SIZE, trainStatus)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(modelPath, map_location=torch.device(device)))
        test_model(model, test_loader, device)
    elif trainStatus == "train":
        train_loader_1, train_loader_2 = make_dataloaders(dataPath, TARGET_IMAGE_SIZE, trainStatus)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        gamma = 0.9
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        train_model(model,device, criterion, optimizer, train_scheduler, train_loader_1, train_loader_2, modelPath)