'''
Implementation of Multi-Modal Ensemble (XGBoost + ResNet)

Workflow:
Deep Features: Images 2 ResNet18 2 512-d embeddings.
Structured Features: Images/Metadata 2 Hand-crafted features 
(ExG, HSV, GLCM, NDVI, Height, State, Species, Month).
Fusion & Prediction: Concatenate embeddings + hand-crafted features 2 XGBoost 2 Final prediction for 5 targets.
'''
import time
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# Config
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
IMAGE_SIZE = 224
RESNET_BATCH = 32
NUM_WORKERS = 4
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = models.resnet18(pretrained=pretrained)
        # remove final fc and global pooling is kept as final layer outputs (we will use avgpool)
        # build backbone up to avgpool (all children except final fc)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # output (B, 512, 1, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x


class ImageDatasetForEmbedding(Dataset):
    def __init__(self, image_paths: List[str], image_dir: str, transforms=None):
        self.image_dir = image_dir
        self.image_paths = image_paths
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


class PastureBiomassPredictor:
    def __init__(self, device=DEVICE):
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.target_names = TARGET_NAMES
        self.device = device
        # initialize resnet extractor
        self.resnet = ResNetFeatureExtractor(pretrained=True).to(self.device)
        self.resnet.eval()

    # handcrafted feature extraction (based on uploaded script but kept concise)
    def extract_all_features(self, img_path: str, metadata: Dict) -> Dict:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        features = {}

        # BGR channels
        b, g, r = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
        total = r + g + b + 1e-6
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total

        # ExG
        exg = 2 * g_norm - r_norm - b_norm
        features['ExG_mean'] = float(np.mean(exg))
        features['ExG_median'] = float(np.median(exg))

        exg_hist, _ = np.histogram(exg, bins=20, range=(-1, 1))
        exg_hist = exg_hist / (exg_hist.sum() + 1e-9)
        # take some bins (guard index)
        for i, bidx in enumerate([0, 1, 5, 7, 8, 9]):
            features[f'ExG_hist_{bidx}'] = float(exg_hist[bidx]) if bidx < len(exg_hist) else 0.0

        # ExR
        exr = 1.4 * r_norm - g_norm
        features['ExR_mean'] = float(np.mean(exr))
        features['ExR_median'] = float(np.median(exr))
        exr_hist, _ = np.histogram(exr, bins=20, range=(-1, 1))
        exr_hist = exr_hist / (exr_hist.sum() + 1e-9)
        for bidx in [4, 9]:
            features[f'ExR_hist_{bidx}'] = float(exr_hist[bidx]) if bidx < len(exr_hist) else 0.0

        # VARI
        vari = (g - r) / (g + r - b + 1e-6)
        features['VARI_mean'] = float(np.mean(vari))

        # CIVE
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
        features['CIVE_mean'] = float(np.mean(cive))

        # HSV features
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        features['HSV_S_std'] = float(np.std(s_channel))
        s_hist = cv2.calcHist([s_channel], [0], None, [32], [0, 256]).flatten()
        s_hist = s_hist / (s_hist.sum() + 1e-9)
        features['HSV_S_hist_4'] = float(s_hist[4]) if len(s_hist) > 4 else 0.0

        h_channel = hsv[:, :, 0]
        h_hist = cv2.calcHist([h_channel], [0], None, [32], [0, 180]).flatten()
        h_hist = h_hist / (h_hist.sum() + 1e-9)
        for bidx in [4, 5, 7]:
            features[f'HSV_H_hist_{bidx}'] = float(h_hist[bidx]) if bidx < len(h_hist) else 0.0

        # GLCM features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, symmetric=True, normed=True)
            homogeneity_values = graycoprops(glcm, 'homogeneity')
            features['GLCM_homogeneity_mean'] = float(np.mean(homogeneity_values))
        except Exception:
            features['GLCM_homogeneity_mean'] = 0.0

        try:
            correlation_values = graycoprops(glcm, 'correlation')
            features['GLCM_correlation_std'] = float(np.std(correlation_values))
        except Exception:
            features['GLCM_correlation_std'] = 0.0

        # Metadata features
        if metadata is not None:
            if 'Pre_GSHH_NDVI' in metadata and pd.notna(metadata['Pre_GSHH_NDVI']):
                features['NDVI'] = float(metadata['Pre_GSHH_NDVI'])
            else:
                features['NDVI'] = 0.0

            if 'Height_Ave_cm' in metadata and pd.notna(metadata['Height_Ave_cm']):
                features['Height_cm'] = float(metadata['Height_Ave_cm'])
            else:
                features['Height_cm'] = 0.0

            if 'State' in metadata and pd.notna(metadata['State']):
                features['State'] = metadata['State']

            if 'Species' in metadata and pd.notna(metadata['Species']):
                features['Species'] = metadata['Species']

            if 'Sampling_Date' in metadata and pd.notna(metadata['Sampling_Date']):
                try:
                    date = pd.to_datetime(metadata['Sampling_Date'])
                    features['Month'] = int(date.month)
                except Exception:
                    features['Month'] = 0
        return features

    def compute_resnet_embeddings(self, image_paths: List[str], image_dir: str, batch_size: int = RESNET_BATCH) -> Dict[str, np.ndarray]:
        dataset = ImageDatasetForEmbedding(image_paths, image_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        embeddings: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for imgs, names in loader:
                imgs = imgs.to(self.device)
                feats = self.resnet(imgs)  # (B, 512)
                feats = feats.cpu().numpy()
                for i, nm in enumerate(names):
                    embeddings[nm] = feats[i]
        return embeddings

    def prepare_features(self, df: pd.DataFrame, image_dir: str, is_training: bool = True) -> np.ndarray:
        all_features = []
        image_paths_ordered = []

        for idx, row in df.iterrows():
            image_path = row['image_path']
            image_name = os.path.basename(image_path)
            # pass the local full path to cv2 reading; earlier we converted image_path to full path
            try:
                feats = self.extract_all_features(image_path, row)
            except Exception as e:
                feats = {}
            all_features.append(feats)
            image_paths_ordered.append(image_path)

        feature_df = pd.DataFrame(all_features)

        # Categorical encoding
        categorical_cols = ['State', 'Species']
        for col in categorical_cols:
            if col in feature_df.columns:
                if is_training:
                    le = LabelEncoder()
                    feature_df[col] = le.fit_transform(feature_df[col].fillna('Unknown'))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        feature_df[col] = feature_df[col].fillna('Unknown')
                        unknown_mask = ~feature_df[col].isin(self.label_encoders[col].classes_)
                        feature_df.loc[unknown_mask, col] = 'Unknown'
                        feature_df[col] = self.label_encoders[col].transform(feature_df[col])
                    else:
                        feature_df[col] = 0

        feature_df = feature_df.fillna(0)

        # compute resnet embeddings and append as features
        # prepare list of raw image names for embedding extraction
        image_names = [os.path.basename(p) for p in image_paths_ordered]
        # compute embeddings (this uses PIL loader and torchvision transforms)
        embeddings = self.compute_resnet_embeddings(image_paths_ordered, image_dir)

        # create embedding columns
        embed_dim = None
        embed_cols_data = []
        for nm in image_names:
            if nm in embeddings:
                emb = embeddings[nm]
            else:
                # fallback zeros
                if embed_dim is None and len(embeddings) > 0:
                    embed_dim = next(iter(embeddings.values())).shape[0]
                emb = np.zeros(embed_dim if embed_dim is not None else 512, dtype=float)
            embed_cols_data.append(emb)
        # convert to array
        embed_array = np.vstack(embed_cols_data)
        if embed_array.ndim == 1:
            embed_array = embed_array.reshape(-1, 1)
        embed_dim = embed_array.shape[1]
        # add embedding columns to feature_df
        for i in range(embed_dim):
            feature_df[f'resnet_{i}'] = embed_array[:, i].astype(float)

        # set feature names and ordering
        if is_training:
            self.feature_names = feature_df.columns.tolist()
        else:
            # add missing features if any
            missing = set(self.feature_names) - set(feature_df.columns)
            for m in missing:
                feature_df[m] = 0.0

        # ensure consistent order
        feature_df = feature_df[self.feature_names]
        return feature_df.values

    def train(self, train_csv_path: str, image_dir: str = '/kaggle/input/csiro-biomass/train'):
        df = pd.read_csv(train_csv_path)
        # normalize image_path to absolute using only filename to avoid duplicated directories
        df['image_path'] = df['image_path'].apply(lambda x: str(Path(image_dir) / Path(x).name))

        unique_images = df.drop_duplicates(subset=['image_path']).copy().reset_index(drop=True)

        X_all = self.prepare_features(unique_images, image_dir=image_dir, is_training=True)

        # mapping image path to index
        image_to_idx = {p: i for i, p in enumerate(unique_images['image_path'].tolist())}

        for target_name in self.target_names:
            target_df = df[df['target_name'] == target_name].copy()
            if target_df.shape[0] == 0:
                continue

            feature_indices = [image_to_idx[str(Path(x).as_posix())] if str(Path(x).as_posix()) in image_to_idx else image_to_idx[str(Path(image_dir) / Path(x).name)] for x in target_df['image_path']]
            X = X_all[feature_indices]
            y = target_df['target'].values

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[target_name] = scaler

            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )

            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )

            self.models[target_name] = model

            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            print(f"Target: {target_name} - Train R2: {train_r2:.4f} - Val R2: {val_r2:.4f}")

            # feature importance display (top 20)
            fi = model.feature_importances_
            fi_df = pd.DataFrame({'feature': self.feature_names, 'importance': fi}).sort_values('importance', ascending=False)
            print(fi_df.head(20).to_string(index=False))

        print("Model training completed.")
        return self

    def predict(self, test_csv_path: str, image_dir: str = '/kaggle/input/csiro-biomass/test'):
        test_df = pd.read_csv(test_csv_path)
        # normalize test image paths
        unique_images = test_df[['image_path']].drop_duplicates().reset_index(drop=True)
        unique_images['image_path'] = unique_images['image_path'].apply(lambda x: str(Path(image_dir) / Path(x).name))

        X_test = self.prepare_features(unique_images, image_dir=image_dir, is_training=False)

        # create a map from image full path to index in unique_images
        img_to_idx = {p: i for i, p in enumerate(unique_images['image_path'].tolist())}

        predictions = []
        for idx, row in test_df.iterrows():
            if idx % 200 == 0:
                print(f"Predicting: {idx}/{len(test_df)}")
            target_name = row['target_name']
            if target_name not in self.models:
                predictions.append(0.0)
                continue

            img_full = str(Path(image_dir) / Path(row['image_path']).name)
            img_idx = img_to_idx.get(img_full, None)
            if img_idx is None:
                # fallback zero features
                X = np.zeros((1, len(self.feature_names)))
            else:
                X = X_test[img_idx:img_idx+1]

            X_scaled = self.scalers[target_name].transform(X)
            pred = self.models[target_name].predict(X_scaled)[0]
            predictions.append(float(max(0, pred)))

        submission = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'target': predictions
        })
        return submission


if __name__ == "__main__":
    start_time = time.time()

    predictor = PastureBiomassPredictor()

    predictor.train('/kaggle/input/csiro-biomass/train.csv', image_dir='/kaggle/input/csiro-biomass/train')

    train_end_time = time.time()
    train_duration = train_end_time - start_time
    print(f"\nTrain takes: {train_duration:.2f} seconds")

    submission = predictor.predict('/kaggle/input/csiro-biomass/test.csv', image_dir='/kaggle/input/csiro-biomass/test')

    submission.to_csv('submission.csv', index=False)
    print("\nPrediction completed! Submission file saved as submission.csv")
    print(f"Submission file shape: {submission.shape}")

    print(f"\nPrediction statistics:")
    # derive target_name from sample_id like ID...__Dry_Green_g
    submission['target_name'] = submission['sample_id'].str.split('__').str[1]
    print(submission.groupby('target_name')['target'].describe())

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nTotal process takes: {total_duration:.2f} seconds")
