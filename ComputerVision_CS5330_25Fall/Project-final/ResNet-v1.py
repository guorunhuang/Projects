'''
对于输入的彩色图，作为输入进入ResNet模型，输出对于五个目标Dry green vegetation * Dry dead material * Dry clover biomass * Green dry matter * Total dry biomass的量的预测
代码能够在Kaggle Notebook里面运行,用python3, 代码使用全英文不要有中文print和注释.此外在main里面要记录训练的总耗时和整个流程的总耗时
'''
# pasture_biomass_resnet_fixed.py
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

TARGET_NAMES = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PastureDataset(Dataset):
    def __init__(self, csv_path: str, image_dir: str, transforms=None):
        df = pd.read_csv(csv_path)

        pivot = df.pivot_table(index=["image_path"], columns="target_name",
                               values="target", aggfunc="first")

        missing_any = pivot.isnull().any(axis=1)
        if missing_any.any():
            pivot = pivot[~missing_any]

        pivot = pivot.reset_index()
        self.records = pivot.to_dict(orient="records")
        self.image_dir = image_dir

        self.transforms = transforms if transforms is not None else self.default_transforms()

    def default_transforms(self):
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_path = rec["image_path"]
        image_name = os.path.basename(image_path)
        full_path = os.path.join(self.image_dir, image_name)

        img = Image.open(full_path).convert("RGB")
        img_t = self.transforms(img)

        target = np.array([rec.get(name, 0.0) for name in TARGET_NAMES], dtype=np.float32)

        return img_t, torch.from_numpy(target), image_name


class TestImageDataset(Dataset):
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
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        full_path = os.path.join(self.image_dir, image_name)

        img = Image.open(full_path).convert("RGB")
        img_t = self.transforms(img)

        return img_t, image_name


class ResNetRegressor(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.backbone(x)


class PastureBiomassPredictor:
    def __init__(self, device=DEVICE, batch_size=BATCH_SIZE):
        self.device = device
        self.batch_size = batch_size
        self.model = ResNetRegressor(out_dim=len(TARGET_NAMES), pretrained=True).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=2, verbose=True
        )

    def train(self, train_csv: str, image_dir: str):
        dataset = PastureDataset(train_csv, image_dir)
        n = len(dataset)
        if n == 0:
            raise RuntimeError("No valid training images found.")

        val_size = int(0.1 * n)
        train_size = n - val_size

        torch.manual_seed(RANDOM_SEED)
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        best_val_loss = float("inf")

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start = time.time()

            self.model.train()
            train_losses = []
            for imgs, targets, _ in train_loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(imgs)
                loss = self.criterion(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for imgs, targets, _ in val_loader:
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)
                    preds = self.model(imgs)
                    loss = self.criterion(preds, targets)
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))

            self.scheduler.step(val_loss)

            epoch_end = time.time()

            print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss: {train_loss:.6f} - "
                  f"val_loss: {val_loss:.6f} - epoch_time: {epoch_end - epoch_start:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print("Saved best_model.pth")

        if os.path.exists("best_model.pth"):
            self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
            print("Loaded best_model.pth")

    def predict(self, test_csv: str, image_dir: str) -> pd.DataFrame:
        test_df = pd.read_csv(test_csv)
        test_dataset = TestImageDataset(test_csv, image_dir)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        self.model.eval()
        preds_by_image: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for imgs, image_names in test_loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs).cpu().numpy()
                for i, name in enumerate(image_names):
                    preds_by_image[name] = outputs[i]

        preds = []
        for _, row in test_df.iterrows():
            image_path = row["image_path"]
            image_name = os.path.basename(image_path)
            target_name = row["target_name"]
            sample_id = row["sample_id"]

            vec = preds_by_image.get(image_name, np.zeros(5))
            idx = TARGET_NAMES.index(target_name)
            pred_val = float(vec[idx])

            preds.append({"sample_id": sample_id, "target": pred_val})

        return pd.DataFrame(preds, columns=["sample_id", "target"])


if __name__ == "__main__":
    start_time = time.time()

    predictor = PastureBiomassPredictor()

    predictor.train('/kaggle/input/csiro-biomass/train.csv',
                    image_dir='/kaggle/input/csiro-biomass/train')

    train_end_time = time.time()
    train_duration = train_end_time - start_time
    print(f"\nTrain takes: {train_duration:.2f} seconds")

    submission = predictor.predict('/kaggle/input/csiro-biomass/test.csv',
                                   image_dir='/kaggle/input/csiro-biomass/test')

    submission.to_csv('submission.csv', index=False)
    print("\nPrediction completed! Submission file saved as submission.csv")
    print(f"Submission file shape: {submission.shape}")

    print("\nPrediction statistics:")
    submission["target_name"] = submission["sample_id"].str.split("__").str[1]
    print(submission.groupby("target_name")["target"].describe())

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nTotal process takes: {total_duration:.2f} seconds")