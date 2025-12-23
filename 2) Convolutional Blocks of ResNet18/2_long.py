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



def modify_initial_layers(model, kernel_size=3, stride=1, remove_maxpool=True):
    in_channels = model.conv1.in_channels
    out_channels = model.conv1.out_channels
    model.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
    if remove_maxpool:
        model.maxpool = nn.Identity()
    return model

model_mod1 = models.resnet18(pretrained=False)
model_mod1.fc = nn.Linear(model_mod1.fc.in_features, num_classes)
model_mod1 = modify_initial_layers(model_mod1, kernel_size=3, stride=1, remove_maxpool=True)
# model_mod1 = model_mod1.to("mps")
model_mod1 = model_mod1.to("cuda")
wandb.init(project="custom-resnet-36-mod1", entity="aryan-g")
model_mod1 = train_model(model_mod1, train_loader, test_loader, num_epochs=30)
torch.save(model_mod1.state_dict(), "custom-resnet-36-mod1.pth")
print("custom-resnet-36-mod1 saved successfully.")





def modify_pretrained_model_mod2(model, kernel_size=5, stride=1, remove_maxpool=False):
    in_channels = model.conv1.in_channels
    out_channels = model.conv1.out_channels
    new_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
    nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
    model.conv1 = new_conv1
    return model

pretrained_mod2 = models.resnet18(pretrained=True)
pretrained_mod2.fc = nn.Linear(pretrained_mod2.fc.in_features, num_classes)
pretrained_mod2 = modify_pretrained_model_mod2(pretrained_mod2, kernel_size=5, stride=1, remove_maxpool=False)
# pretrained_mod2 = pretrained_mod2.to("mps")
pretrained_mod2 = pretrained_mod2.to("cuda")
wandb.init(project="custom-resnet-36-pretrained-mod2-pre", entity="aryan-g")
pretrained_mod2 = train_model(pretrained_mod2, train_loader, test_loader, num_epochs=30)
torch.save(pretrained_mod2.state_dict(), "custom-resnet-36-pretrained-mod2.pth")
print("custom-resnet-36-pretrained-mod2 saved successfully.")



def modify_pretrained_model_mod3(model, remove_maxpool=True):
    in_channels = model.conv1.in_channels
    mid_channels = model.conv1.out_channels  
    conv1a = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(conv1a.weight, mode='fan_out', nonlinearity='relu')
    conv1b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(conv1b.weight, mode='fan_out', nonlinearity='relu')
    new_conv_block = nn.Sequential( conv1a, nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), conv1b, nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True) )
    model.conv1 = new_conv_block
    if remove_maxpool:
        model.maxpool = nn.Identity()
    return model

pretrained_mod3 = models.resnet18(pretrained=True)
pretrained_mod3.fc = nn.Linear(pretrained_mod3.fc.in_features, num_classes)
pretrained_mod3 = modify_pretrained_model_mod3(pretrained_mod3, remove_maxpool=True)
# pretrained_mod3 = pretrained_mod3.to("mps")
pretrained_mod3 = pretrained_mod3.to("cuda")
wandb.init(project="custom-resnet-36-pretrained-mod3", entity="aryan-g")
pretrained_mod3 = train_model(pretrained_mod3, train_loader, test_loader, num_epochs=30)
torch.save(pretrained_mod3.state_dict(), "custom-resnet-36-pretrained-mod3.pth")
print("custom-resnet-36-pretrained-mod3 saved successfully.") 
