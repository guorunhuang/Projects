'''
Images + ResNet

print:
Epoch 1/5
Train Loss: 0.2431, Val Loss: 0.2809
Target: Dry_Green_g - Train R^2: 0.8123 - Val R^2: 0.4123
Target: Dry_Dead_g - Train R^2: 0.9021 - Val R^2: 0.5877
Target: Dry_Clover_g - Train R^2: 0.7312 - Val R^2: 0.3944
Target: GDM_g - Train R^2: 0.9503 - Val R^2: 0.6891
Target: Dry_Total_g - Train R^2: 0.9714 - Val R^2: 0.6464
'''

import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import r2_score

# Constants
TARGET_NAMES = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTargetDataset(Dataset):
    """Dataset returning image and 5-target vector."""
    def __init__(self, csv_path: str, image_dir: str, transforms=None):
        df = pd.read_csv(csv_path)

        pivot = df.pivot_table(
            index=["image_path"],
            columns="target_name",
            values="target",
            aggfunc="first"
        )

        pivot = pivot.dropna().reset_index()
        self.records = pivot.to_dict(orient="records")

        self.image_dir = image_dir
        self.transforms = transforms if transforms is not None else self.default_transforms()

    def default_transforms(self):
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_name = os.path.basename(rec["image_path"])
        full_path = os.path.join(self.image_dir, image_name)

        img = Image.open(full_path).convert("RGB")
        img_t = self.transforms(img)

        target = np.array([rec[name] for name in TARGET_NAMES], dtype=np.float32)

        return img_t, torch.from_numpy(target), image_name


class TestDataset(Dataset):
    def __init__(self, test_csv: str, image_dir: str, transforms=None):
        df = pd.read_csv(test_csv)
        unique = df[["image_path"]].drop_duplicates().reset_index(drop=True)
        self.image_paths = unique["image_path"].tolist()
        self.image_dir = image_dir

        self.transforms = transforms if transforms is not None else self.default_transforms()

    def default_transforms(self):
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        full_path = os.path.join(self.image_dir, image_name)

        img = Image.open(full_path).convert("RGB")
        img_t = self.transforms(img)

        return img_t, image_name


class ResNetRegressor(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        m = models.resnet18(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.reg_head(feat)
        return out


class PastureBiomassPredictor:
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = ResNetRegressor(out_dim=5, pretrained=True).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=2, verbose=True
        )

    def train(self, train_csv: str, image_dir: str):
        dataset = MultiTargetDataset(train_csv, image_dir)
        n = len(dataset)
        val_size = int(0.1 * n)
        train_size = n - val_size

        torch.manual_seed(RANDOM_SEED)
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        best_val_loss = float("inf")

        for epoch in range(1, NUM_EPOCHS + 1):
            self.model.train()
            train_losses = []
            train_preds = []
            train_targets = []

            for imgs, targets, _ in train_loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(imgs)
                loss = self.criterion(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                train_preds.append(preds.detach().cpu().numpy())
                train_targets.append(targets.cpu().numpy())

            train_preds = np.vstack(train_preds)
            train_targets = np.vstack(train_targets)

            # compute R² per target
            train_r2 = {}
            for i, tname in enumerate(TARGET_NAMES):
                train_r2[tname] = r2_score(train_targets[:, i], train_preds[:, i])

            # validation
            self.model.eval()
            val_losses = []
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for imgs, targets, _ in val_loader:
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)

                    preds = self.model(imgs)
                    loss = self.criterion(preds, targets)

                    val_losses.append(loss.item())
                    val_preds.append(preds.cpu().numpy())
                    val_targets.append(targets.cpu().numpy())

            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)

            val_r2 = {}
            for i, tname in enumerate(TARGET_NAMES):
                val_r2[tname] = r2_score(val_targets[:, i], val_preds[:, i])

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))

            self.scheduler.step(val_loss)

            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            for tname in TARGET_NAMES:
                print(f"Target: {tname} - Train R^2: {train_r2[tname]:.4f} - Val R^2: {val_r2[tname]:.4f}")

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print("Saved best_model.pth")

        print("Training completed.")

        # load best model
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        print("Loaded best_model.pth")

    def predict(self, test_csv: str, image_dir: str):
        test_df = pd.read_csv(test_csv)

        test_dataset = TestDataset(test_csv, image_dir)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        self.model.eval()

        img_to_predvec: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for imgs, image_names in test_loader:
                imgs = imgs.to(self.device)
                preds = self.model(imgs).cpu().numpy()
                for i, nm in enumerate(image_names):
                    img_to_predvec[nm] = preds[i]

        preds = []
        for _, row in test_df.iterrows():
            image_name = os.path.basename(row["image_path"])
            target_name = row["target_name"]
            idx = TARGET_NAMES.index(target_name)
            preds.append(img_to_predvec[image_name][idx])

        return pd.DataFrame({"sample_id": test_df["sample_id"], "target": preds})


if __name__ == "__main__":
    start_time = time.time()

    predictor = PastureBiomassPredictor()

    predictor.train('/kaggle/input/csiro-biomass/train.csv',
                    image_dir='/kaggle/input/csiro-biomass/train')

    train_end = time.time()
    print(f"\nTrain takes: {train_end - start_time:.2f} seconds")

    submission = predictor.predict('/kaggle/input/csiro-biomass/test.csv',
                                   image_dir='/kaggle/input/csiro-biomass/test')

    submission.to_csv("submission.csv", index=False)
    print("\nPrediction completed! Saved as submission.csv")
    print(submission.head())

    end_time = time.time()
    print(f"\nTotal process takes: {end_time - start_time:.2f} seconds")
