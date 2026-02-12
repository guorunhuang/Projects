'''
metadata + handcrafted + sentinel-2 bands into MLP
'''

import os
import time
from pathlib import Path
from datetime import timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configuration
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
BATCH_SIZE = 32
MLP_HIDDEN = 64
MLP_MAX_EPOCHS = 3000
MLP_PATIENCE = 5
PRINT_EVERY = 100


class MLPRegressor(nn.Module):
    """MLP with 2 hidden layers, 64 nodes each, ReLU activation"""
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, MLP_HIDDEN)
        self.fc2 = nn.Linear(MLP_HIDDEN, MLP_HIDDEN)
        self.fc3 = nn.Linear(MLP_HIDDEN, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PastureBiomassPredictor:
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.target_names = TARGET_NAMES
        self.device = DEVICE

    def extract_all_features(self, img_path, metadata=None):
        """Extract selected handcrafted image features plus metadata-derived features."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        features = {}
        
        b, g, r = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
        total = r + g + b + 1e-6
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
        
        exg = 2 * g_norm - r_norm - b_norm
        features['ExG_mean'] = float(np.mean(exg))
        features['ExG_median'] = float(np.median(exg))
        exg_hist, _ = np.histogram(exg, bins=20, range=(-1, 1))
        exg_hist = exg_hist / (exg_hist.sum() + 1e-9)
        features['ExG_hist_0'] = float(exg_hist[0]) if len(exg_hist) > 0 else 0.0
        features['ExG_hist_1'] = float(exg_hist[1]) if len(exg_hist) > 1 else 0.0
        features['ExG_hist_5'] = float(exg_hist[5]) if len(exg_hist) > 5 else 0.0
        features['ExG_hist_7'] = float(exg_hist[7]) if len(exg_hist) > 7 else 0.0
        features['ExG_hist_8'] = float(exg_hist[8]) if len(exg_hist) > 8 else 0.0
        features['ExG_hist_9'] = float(exg_hist[9]) if len(exg_hist) > 9 else 0.0
        
        exr = 1.4 * r_norm - g_norm
        features['ExR_mean'] = float(np.mean(exr))
        features['ExR_median'] = float(np.median(exr))
        exr_hist, _ = np.histogram(exr, bins=20, range=(-1, 1))
        exr_hist = exr_hist / (exr_hist.sum() + 1e-9)
        features['ExR_hist_4'] = float(exr_hist[4]) if len(exr_hist) > 4 else 0.0
        features['ExR_hist_9'] = float(exr_hist[9]) if len(exr_hist) > 9 else 0.0
        
        vari = (g - r) / (g + r - b + 1e-6)
        features['VARI_mean'] = float(np.mean(vari))
        
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
        features['CIVE_mean'] = float(np.mean(cive))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        features['HSV_S_std'] = float(np.std(s_channel))
        s_hist = cv2.calcHist([s_channel], [0], None, [32], [0, 256]).flatten()
        s_hist = s_hist / (s_hist.sum() + 1e-9)
        features['HSV_S_hist_4'] = float(s_hist[4]) if len(s_hist) > 4 else 0.0
        
        h_channel = hsv[:, :, 0]
        h_hist = cv2.calcHist([h_channel], [0], None, [32], [0, 180]).flatten()
        h_hist = h_hist / (h_hist.sum() + 1e-9)
        features['HSV_H_hist_4'] = float(h_hist[4]) if len(h_hist) > 4 else 0.0
        features['HSV_H_hist_5'] = float(h_hist[5]) if len(h_hist) > 5 else 0.0
        features['HSV_H_hist_7'] = float(h_hist[7]) if len(h_hist) > 7 else 0.0
        
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

    def load_sentinel_table(self, s2_csv_path):
        s2 = pd.read_csv(s2_csv_path)
        s2['S2_Date'] = pd.to_datetime(s2['Date'], errors='coerce')
        s2 = s2.dropna(subset=['S2_Date']).reset_index(drop=True)
        return s2

    def prepare_features(self, df_images: pd.DataFrame, image_dir: str, s2_df: pd.DataFrame, is_training=True):
        """Extract image handcrafted features and match sentinel-2 bands for NSW samples."""
        all_features = []
        image_paths = []
        for idx, row in df_images.iterrows():
            img_path = row['image_path']
            image_paths.append(img_path)
            try:
                feats = self.extract_all_features(img_path, row)
            except Exception as e:
                feats = {}
            all_features.append(feats)
        feature_df = pd.DataFrame(all_features)
        # Attach sentinel-2 band columns initialized to NaN
        s2_band_cols = [c for c in s2_df.columns if c.startswith('nbart_')]
        for c in s2_band_cols:
            if c not in feature_df.columns:
                feature_df[c] = np.nan
        # Match sentinel-2 by date within +/-5 days for NSW samples only
        if 'Sampling_Date' in df_images.columns:
            df_images_dates = df_images.copy()
            df_images_dates['Sampling_Date'] = pd.to_datetime(df_images_dates['Sampling_Date'], errors='coerce')
            for i, row in df_images_dates.iterrows():
                state_val = row.get('State', None)
                if state_val is None:
                    continue
                if str(state_val).strip().lower() != 'NSW':
                    continue
                sample_date = row['Sampling_Date']
                if pd.isna(sample_date):
                    continue
                candidates = s2_df[(s2_df['S2_Date'] >= sample_date - timedelta(days=5)) &
                                   (s2_df['S2_Date'] <= sample_date + timedelta(days=5))].copy()
                if candidates.shape[0] == 0:
                    continue
                candidates['date_diff'] = (candidates['S2_Date'] - sample_date).abs()
                best = candidates.sort_values('date_diff').iloc[0]
                for c in s2_band_cols:
                    try:
                        feature_df.at[i, c] = float(best.get(c, np.nan))
                    except Exception:
                        feature_df.at[i, c] = np.nan
        # metadata categorical encoding
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
        if is_training:
            self.feature_names = feature_df.columns.tolist()
        else:
            missing_features = set(self.feature_names) - set(feature_df.columns)
            for mf in missing_features:
                feature_df[mf] = 0.0
        feature_df = feature_df[self.feature_names]
        return feature_df.values

    def train(self, train_csv_path, image_dir='/kaggle/input/csiro-biomass/train', s2_csv_path=None):
        """Train MLP models per target using metadata + handcrafted + sentinel bands"""
        df = pd.read_csv(train_csv_path)
        # Normalize image_path to full path using filename to avoid duplicated folders
        df['image_path'] = df['image_path'].apply(lambda x: str(Path(image_dir) / Path(x).name))
        # load sentinel table if provided
        s2_df = None
        if s2_csv_path is not None:
            s2_df = self.load_sentinel_table(s2_csv_path)
        else:
            # create empty with no nbart columns
            s2_df = pd.DataFrame(columns=['S2_Date'])
        # unique images (one row per image; metadata columns preserved from first occurrence)
        unique_images = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
        # ensure Sampling_Date column parsed
        if 'Sampling_Date' in unique_images.columns:
            unique_images['Sampling_Date'] = pd.to_datetime(unique_images['Sampling_Date'], errors='coerce')
        print(f"Number of unique images for feature extraction: {len(unique_images)}")
        # Extract features and match sentinel bands
        X_all = self.prepare_features(unique_images, image_dir=image_dir, s2_df=s2_df, is_training=True)
        # map image_path to index
        image_to_idx = {p: idx for idx, p in enumerate(unique_images['image_path'].tolist())}
        # Train one MLP per target
        for target_name in self.target_names:
            print(f"\n{'='*60}")
            print(f"Training target: {target_name}")
            print(f"{'='*60}")
            target_df = df[df['target_name'] == target_name].copy()
            if target_df.shape[0] == 0:
                print(f"Warning: No data for target {target_name}")
                continue
            # map rows to feature rows
            feature_indices = []
            for p in target_df['image_path']:
                if p in image_to_idx:
                    feature_indices.append(image_to_idx[p])
                else:
                    # fallback: try basename match
                    base = str(Path(p).name)
                    for k, v in image_to_idx.items():
                        if str(Path(k).name) == base:
                            feature_indices.append(v)
                            break
                    else:
                        feature_indices.append(None)
            valid_mask = [i is not None for i in feature_indices]
            if not any(valid_mask):
                print("No valid feature rows found for this target.")
                continue
            # filter
            target_df = target_df.iloc[[i for i, ok in enumerate(valid_mask) if ok]].reset_index(drop=True)
            feature_indices = [fi for fi in feature_indices if fi is not None]
            X = X_all[feature_indices]
            y = target_df['target'].values
            # train/val split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[target_name] = scaler
            # tensors and loaders
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
            # initialize model
            input_dim = X_train_scaled.shape[1]
            model = MLPRegressor(input_dim).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None
            for epoch in range(MLP_MAX_EPOCHS):
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_X.size(0)
                train_loss /= len(train_loader.dataset)
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                if (epoch + 1) % PRINT_EVERY == 0:
                    print(f"Epoch {epoch+1}/{MLP_MAX_EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                if patience_counter >= MLP_PATIENCE:
                    # early stop
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
            # save model
            self.models[target_name] = model
            # evaluation R2
            model.eval()
            with torch.no_grad():
                train_pred = model(torch.FloatTensor(X_train_scaled).to(self.device)).cpu().numpy().flatten()
                val_pred = model(X_val_tensor).cpu().numpy().flatten()
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            print(f"Target: {target_name} - Train R^2: {train_r2:.4f} - Val R^2: {val_r2:.4f}")
        print("\nModel training completed!")
        return self

    def predict(self, test_csv_path, image_dir='/kaggle/input/csiro-biomass/test', s2_csv_path=None):
        test_df = pd.read_csv(test_csv_path)
        test_df['image_path_full'] = test_df['image_path'].apply(lambda x: str(Path(image_dir) / Path(x).name))
        unique_images = test_df[['image_path']].drop_duplicates().reset_index(drop=True)
        unique_images['image_path'] = unique_images['image_path'].apply(lambda x: str(Path(image_dir) / Path(x).name))
        s2_df = None
        if s2_csv_path is not None:
            s2_df = self.load_sentinel_table(s2_csv_path)
        else:
            s2_df = pd.DataFrame(columns=['S2_Date'])
        X_test = self.prepare_features(unique_images, image_dir=image_dir, s2_df=s2_df, is_training=False)
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
                X = np.zeros((1, len(self.feature_names)))
            else:
                X = X_test[img_idx:img_idx+1]
            X_scaled = self.scalers[target_name].transform(X)
            model = self.models[target_name]
            model.eval()
            with torch.no_grad():
                pred = model(torch.FloatTensor(X_scaled).to(self.device)).cpu().numpy().flatten()[0]
            predictions.append(float(max(0.0, pred)))
        submission = pd.DataFrame({'sample_id': test_df['sample_id'], 'target': predictions})
        return submission


if __name__ == "__main__":
    start_time = time.time()
    predictor = PastureBiomassPredictor()
    predictor.train('/kaggle/input/csiro-biomass/train.csv',
                    image_dir='/kaggle/input/csiro-biomass/train',
                    s2_csv_path='/kaggle/input/csiro-s2-nsw/sentinel2_dea_kaggle-NSW.csv')
    train_end_time = time.time()
    train_duration = train_end_time - start_time
    print(f"\nTrain takes: {train_duration:.2f} seconds")
    submission = predictor.predict('/kaggle/input/csiro-biomass/test.csv',
                                   image_dir='/kaggle/input/csiro-biomass/test',
                                   s2_csv_path='/kaggle/input/csiro-s2-nsw/sentinel2_dea_kaggle-NSW.csv')
    submission.to_csv('submission.csv', index=False)
    print("\nPrediction completed! Submission file saved as submission.csv")
    print(f"Submission file shape: {submission.shape}")
    print(f"\nPrediction statistics:")
    submission['target_name'] = submission['sample_id'].str.split('__').str[1]
    print(submission.groupby('target_name')['target'].describe())
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nTotal process takes: {total_duration:.2f} seconds")
