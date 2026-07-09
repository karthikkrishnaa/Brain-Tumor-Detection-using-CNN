# ================== IMPORTS ==================
import os
import random
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ================== SETTINGS ==================
TRAINING_DIR = "/content/drive/MyDrive/combined dataset/Training"
TESTING_DIR = "/content/drive/MyDrive/combined dataset/Testing"
BATCH_SIZE = 32
NUM_EPOCHS = 10
PATIENCE = 3
IMG_SIZE = 224
LEARNING_RATE = 0.001
random.seed(42)
torch.manual_seed(42)

# ================== TRANSFORMS ==================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================== DATA HANDLING ==================
def get_all_samples(root_dir):
    samples = []
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            samples.append((fpath, class_to_idx[cls]))
    return samples, class_to_idx

class TumorDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            img_path, label = self.samples[idx]
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            except (UnidentifiedImageError, OSError):
                print(f"Skipping corrupted image: {img_path}")
                idx = (idx + 1) % len(self.samples)

# ================== LOAD TRAINING DATA ==================
samples, class_to_idx = get_all_samples(TRAINING_DIR)
print("Classes:", class_to_idx)

random.shuffle(samples)
split_idx = int(0.8 * len(samples))
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

train_dataset = TumorDataset(train_samples, transform=train_transform)
val_dataset = TumorDataset(val_samples, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ================== MODEL ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================== TRAINING LOOP ==================
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0

os.makedirs("checkpoints", exist_ok=True)
best_acc = 0
early_stop_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_acc = evaluate(val_loader)
    print(f"\nEpoch {epoch+1} - Loss: {running_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_model combined.pth")
        print(" Saved best model.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Early stop patience: {early_stop_counter}/{PATIENCE}")

    if early_stop_counter >= PATIENCE:
        print(" Early stopping.")
        break

# ================== EVALUATE ON VALIDATION ==================
model.load_state_dict(torch.load("checkpoints/best_model combined.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\nValidation Confusion Matrix:")
print(cm)
print("\nValidation Classification Report:")
print(classification_report(all_labels, all_preds, target_names=list(class_to_idx.keys())))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(class_to_idx.keys()), yticklabels=list(class_to_idx.keys()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Validation Confusion Matrix")
plt.show()

# ================== TESTING ON TEST FOLDER ==================
test_samples, _ = get_all_samples(TESTING_DIR)
test_dataset = TumorDataset(test_samples, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = torch.argmax(out, dim=1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_test_labels, all_test_preds)
print("\n--- Test Results ---")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(all_test_labels, all_test_preds, target_names=list(class_to_idx.keys())))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(class_to_idx.keys()), yticklabels=list(class_to_idx.keys()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Testing Data Confusion Matrix")
plt.show()