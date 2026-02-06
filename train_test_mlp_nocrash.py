import os
import sys
import glob
import random
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.GANmigru3bw.discriminator import Discriminator # change here to match your local setup

# Update these paths to match your local setup before running the code.
# Absolute development paths are not included in the public repository.
root_folder = "outputs/imagesGIDS/CH15"

gan_model_path = "outputs/gan_training_new/CH15_15_bw2/netD_epoch_95.pth"

mlp_save_path = "saved_models/mlp_model_combined_surv_CHbw152.pth"

image_size = (16, 305)
batch_size = 64
train_seed = 42
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(train_seed)
torch.manual_seed(train_seed)

# attack folders
attack_folders = [
    os.path.join(root_folder, d)
    for d in os.listdir(root_folder)
    if os.path.isdir(os.path.join(root_folder, d)) and d.startswith("shell_")
]

# sample files
def filter_and_sample(folder, suffix, percentage):
    files = glob.glob(os.path.join(folder, "**", f"*_{suffix}.png"), recursive=True)
    if len(files) == 0:
        return set()
    sample_count = max(1, int(len(files) * percentage))  # avoid 0 when very small
    sample_count = min(sample_count, len(files))
    return set(random.sample(files, sample_count))

train_files = set()
for folder in attack_folders:
    mal_samples = filter_and_sample(folder, "m", 0.8)
    norm_samples = filter_and_sample(folder, "n", 0.8)
    train_files |= mal_samples | norm_samples

print(f"Total train samples: {len(train_files)}")


class CustomImageDataset(data.Dataset):
    def __init__(self, file_paths, transform):
        self.files = list(file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        # label: 0 for normal (_n), 1 for malicious (_m);
        base = os.path.basename(path)
        label = 0 if "_n" in base else (1 if "_m" in base else 1)
        return img, label


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])




netC = Discriminator().to(device)
netC.load_state_dict(torch.load(gan_model_path, map_location=device))
netC.eval()

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, critic, layer_num=1):
        super().__init__()
        self.main = critic.main
        self.layer_num = layer_num

    def forward(self, x):
        m = self.main
        if self.layer_num == 1:
            x = m[0](x); x = m[1](x); x = m[2](x)
        elif self.layer_num == 2:
            x = m[0](x); x = m[1](x); x = m[2](x)
            x = m[3](x); x = m[4](x); x = m[5](x)
        elif self.layer_num == 3:
            x = m[0](x); x = m[1](x); x = m[2](x)
            x = m[3](x); x = m[4](x); x = m[5](x)
            x = m[6](x); x = m[7](x); x = m[8](x)
        elif self.layer_num == 4:
            x = m(x)
        else:
            raise ValueError("layer_num must be 1–4")
        return x.view(x.size(0), -1)

feature_net = FeatureExtractor(netC, layer_num=1).to(device)
feature_net.eval()

# Extract features
@torch.inference_mode()
def extract_feats_and_labels(loader):
    all_feats, all_lbls = [], []
    for imgs, lbls in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        feats = feature_net(imgs)
        all_feats.append(feats.cpu())
        all_lbls.append(lbls)
    feats = torch.cat(all_feats).numpy()
    lbls = torch.cat(all_lbls).numpy()
    return feats, lbls

# MLP Classifier
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

# TRAIN
train_dataset = CustomImageDataset(train_files, transform)
train_loader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=(device.type == "cuda")
)

X_train_raw, y_train = extract_feats_and_labels(train_loader)
X_train = normalize(X_train_raw)

# TRAIN
with torch.no_grad():
    dummy = torch.zeros(1, 1, *image_size).to(device)
    input_dim = feature_net(dummy).shape[1]

mlp = MLP(input_dim=input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

train_loader_stream = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda")
)

mlp.train()
start = time.time()
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    for imgs, yb in train_loader_stream:
        imgs = imgs.to(device, non_blocking=True)
        yb   = yb.to(device, non_blocking=True)

        # normalized features per batch
        with torch.no_grad():
            feats = feature_net(imgs)
            feats = F.normalize(feats, p=2, dim=1)

        optimizer.zero_grad()
        logits = mlp(feats)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}/{num_epochs} loss={total_loss:.4f}")

print("Training time:", time.time() - start)

os.makedirs(os.path.dirname(mlp_save_path), exist_ok=True)
torch.save(mlp.state_dict(), mlp_save_path)
print(f"MLP model saved to {mlp_save_path}")

"""
train_dataset = CustomImageDataset(train_files, transform)
train_loader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=(device.type == "cuda")
)

X_train_raw, y_train = extract_feats_and_labels(train_loader)
X_train = normalize(X_train_raw)  # sklearn L2-normalize

mlp = MLP(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

mlp.train()
start = time.time()
train_dataset2 = data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.long))
train_loader2 = data.DataLoader(
    train_dataset2, batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=(device.type == "cuda")
)

num_epochs = 50
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    for xb, yb in train_loader2:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = mlp(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{num_epochs} loss={total_loss:.4f}")
print("Training time:", time.time() - start)

# Save model
os.makedirs(os.path.dirname(mlp_save_path), exist_ok=True)
torch.save(mlp.state_dict(), mlp_save_path)
print(f"MLP model saved to {mlp_save_path}")
"""

# TEST
mlp.eval()
print("\nStarting evaluation...\n")

@torch.inference_mode()
def eval_streaming(test_loader):
    all_preds = []
    all_labels = []
    for imgs, lbls in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        # 1) extract features
        feats = feature_net(imgs)
        # 2) L2-normalize per sample (equivalent to sklearn's normalize with L2)
        feats = F.normalize(feats, p=2, dim=1)
        # 3) run classifier
        logits = mlp(feats)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(lbls.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return y_true, y_pred

for attack_folder in attack_folders:
    print(f"=== Evaluating attack folder: {os.path.basename(attack_folder)} ===")

    all_attack_files = set(glob.glob(os.path.join(attack_folder, "**", "*.png"), recursive=True))
    test_files = list(all_attack_files - train_files)

    if not test_files:
        print("--------------------------------------")
        continue

    test_dataset = CustomImageDataset(test_files, transform)
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    y_test, preds = eval_streaming(test_loader)

    print("y_test unique:", np.unique(y_test, return_counts=True))
    print("preds unique:", np.unique(preds, return_counts=True))

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"Samples: {len(test_files)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("--------------------------------------")


