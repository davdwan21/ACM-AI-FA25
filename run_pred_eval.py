import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os

# same model definition (can import but f it we ball)
class BlockographyCNNFull(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # slightly lower dropout?
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# load model in eval mode
model = BlockographyCNNFull(num_classes=29).to(device)
model.load_state_dict(torch.load("blockography_resnet50_finetuned_v2.pth", map_location=device))
model.eval()
print("model loaded successfully")

# size image (no distortion)
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# run predictions on eval
eval_dir = 'eval_data'
image_files = sorted(os.listdir(eval_dir))
results = []

with torch.no_grad():
    for img_name in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(eval_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Skipping unreadable image: {img_name}")
            continue

        # dont ask me why there are errors here :)
        image = transform_eval(image).unsqueeze(0).to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        pred_label = class_names[predicted.item()]
        results.append({"id": img_name, "prediction": pred_label})


df = pd.DataFrame(results)
df.to_csv("submission.csv", index=False)
print("predictions saved to 'submission.csv'")
