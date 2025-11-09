import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# use mac gpu (horrendous behavior)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 25 # definitely use more next time
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# image distortion
transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # doubtful that this helped
])

train_dir = 'train_data'
train_data = datasets.ImageFolder(root=train_dir, transform=transform_train)
num_classes = len(train_data.classes)
print(f"found {num_classes} classes: {train_data.classes}")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# transfer learning from resnet50
class BlockographyCNNFull(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential( # WHY IS THERE AN ERROR HERE
            nn.Linear(in_features, 512), # lower features in next attempt
            nn.ReLU(),
            nn.Dropout(0.25), # 0.3?
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

model = BlockographyCNNFull(num_classes=num_classes).to(device)

# unfreeze hidden layers 2 and 3
for param in model.base_model.parameters():
    param.requires_grad = False
for param in model.base_model.layer3.parameters():
    param.requires_grad = True
for param in model.base_model.layer4.parameters():
    param.requires_grad = True
for param in model.base_model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# training loop
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset) # why the fuck is there an error here
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}: Training Loss = {epoch_loss:.4f}")

torch.save(model.state_dict(), "blockography_resnet50_finetuned.pth")
print("saved 'blockography_resnet50_finetuned.pth'")


""" plt.plot(train_losses, label='Training Loss')
plt.title("Training Loss Curve (ResNet50 Fine-Tuned)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show() """