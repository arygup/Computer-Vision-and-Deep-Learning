import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

train_data = torch.load("train_data.pt")  
train_labels = torch.load("train_labels.pt")
test_data = torch.load("test_data.pt")
test_labels = torch.load("test_labels.pt")
train_data = train_data.float() 
test_data = test_data.float() 
batch_size = 500
train_ds = TensorDataset(train_data, train_labels)
test_ds = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)
num_classes = len(torch.unique(train_labels))
print(f"Number of classes: {num_classes}")


def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-3):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss /= total_val
        val_acc = correct_val / total_val
        wandb.log({ "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, train acc {train_acc:.4f}, " f"val loss {val_loss:.4f}, val acc {val_acc:.4f}")
    return model


class CustomDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor, transform=None):
        self.data = data_tensor
        self.labels = labels_tensor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img = self.data[idx]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

torch.cuda.empty_cache()  
batch_size = 350
transform_resize = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset_resized = CustomDataset(train_data, train_labels, transform=transform_resize)
test_dataset_resized = CustomDataset(test_data, test_labels, transform=transform_resize)
train_loader_resized = DataLoader(train_dataset_resized, batch_size=batch_size, shuffle=True)
test_loader_resized = DataLoader(test_dataset_resized, batch_size=batch_size)

model_resized = models.resnet18(pretrained=False)
model_resized.fc = nn.Linear(model_resized.fc.in_features, num_classes)
model_resized = model_resized.to("cuda")
# model_resized = model_resized.to("mps")


wandb.init(project="custom-resnet-224", entity="aryan-g")
model_resized = train_model(model_resized, train_loader_resized, test_loader_resized, num_epochs=30)
torch.save(model_resized.state_dict(), "custom-resnet-224.pth")
print("custom-resnet-224 saved successfully.")


pretrained_model_resized = models.resnet18(pretrained=True)
pretrained_model_resized.fc = nn.Linear(pretrained_model_resized.fc.in_features, num_classes)
# pretrained_model_resized = pretrained_model_resized.to("mps")
pretrained_model_resized = pretrained_model_resized.to("cuda")

wandb.init(project="custom-resnet-224-pretrained", entity="aryan-g")
pretrained_model_resized = train_model(pretrained_model_resized, train_loader_resized, test_loader_resized, num_epochs=30)
torch.save(pretrained_model_resized.state_dict(), "custom-resnet-224-pretrained.pth")
print("custom-resnet-224-pretrained saved successfully.")