'''
我发现用的特征太多了，现在我希望只保留以下特征作为模型的输入：几何/基本Height_cm 日期/状态State, Month, Species 
颜色指数 (ExG)ExG_median, ExG_mean, ExG_hist_9, ExG_hist_8, ExG_hist_7, ExG_hist_5, ExG_hist_1, ExG_hist_0 
颜色指数 (ExR)ExR_mean, ExR_median, ExR_hist_9, ExR_hist_4 颜色指数 (其它)CIVE_mean, NDVI, VARI_mean 
HSV 特征HSV_S_std, HSV_S_hist_4, HSV_H_hist_7, HSV_H_hist_5, HSV_H_hist_4 GLCM 纹理GLCM_homogeneity_mean, 
GLCM_correlation_std（如果计算出这些特征需要一些中间的计算步骤而会产生其他的特征则没关系）
在 prepare_features 中添加特征过滤逻辑（第244-256行）
pythonif is_training:
    # 提取所有特征
    all_feature_names = feature_df.columns.tolist()
    
    # 只保留选定的特征
    available_selected = [f for f in self.selected_features if f in all_feature_names]
    missing_selected = [f for f in self.selected_features if f not in all_feature_names]
'''
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PastureBiomassPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.target_names = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
        
        # Selected features to use in the model
        self.selected_features = [
            # Geometric/Basic
            'Height_cm',
            # Date/State
            'State', 'Month', 'Species',
            # Color Index (ExG)
            'ExG_median', 'ExG_mean', 'ExG_hist_9', 'ExG_hist_8', 'ExG_hist_7', 
            'ExG_hist_5', 'ExG_hist_1', 'ExG_hist_0',
            # Color Index (ExR)
            'ExR_mean', 'ExR_median', 'ExR_hist_9', 'ExR_hist_4',
            # Color Index (Other)
            'CIVE_mean', 'NDVI', 'VARI_mean',
            # HSV Features
            'HSV_S_std', 'HSV_S_hist_4', 'HSV_H_hist_7', 'HSV_H_hist_5', 'HSV_H_hist_4',
            # GLCM Texture
            'GLCM_homogeneity_mean', 'GLCM_correlation_std'
        ]
        
    def extract_color_features(self, img):
        """Extract RGB, HSV and hue distribution features"""
        features = {}
        
        # RGB features
        for i, color in enumerate(['R', 'G', 'B']):
            channel = img[:, :, i]
            features[f'{color}_mean'] = np.mean(channel)
            features[f'{color}_std'] = np.std(channel)
            features[f'{color}_median'] = np.median(channel)
            
            # RGB histogram features
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()
            for j in range(min(10, len(hist))):  # Top 10 bins
                features[f'{color}_hist_{j}'] = hist[j]
        
        # HSV features
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, channel_name in enumerate(['H', 'S', 'V']):
            channel = hsv[:, :, i]
            features[f'HSV_{channel_name}_mean'] = np.mean(channel)
            features[f'HSV_{channel_name}_std'] = np.std(channel)
            features[f'HSV_{channel_name}_median'] = np.median(channel)
            
            # HSV histogram
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256 if i > 0 else 180])
            hist = hist.flatten() / hist.sum()
            for j in range(min(8, len(hist))):
                features[f'HSV_{channel_name}_hist_{j}'] = hist[j]
        
        return features
    
    def calculate_vegetation_indices(self, img):
        """Calculate vegetation indices: ExG, ExR, VARI, CIVE, NDI"""
        features = {}
        
        # Normalize RGB
        b, g, r = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
        total = r + g + b + 1e-6
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
        
        # ExG (Excess Green)
        exg = 2 * g_norm - r_norm - b_norm
        features['ExG_mean'] = np.mean(exg)
        features['ExG_std'] = np.std(exg)
        features['ExG_median'] = np.median(exg)
        features['ExG_min'] = np.min(exg)
        features['ExG_max'] = np.max(exg)
        
        # ExG histogram
        exg_hist, _ = np.histogram(exg, bins=20, range=(-1, 1))
        exg_hist = exg_hist / exg_hist.sum()
        for i in range(min(10, len(exg_hist))):
            features[f'ExG_hist_{i}'] = exg_hist[i]
        
        # ExR (Excess Red)
        exr = 1.4 * r_norm - g_norm
        features['ExR_mean'] = np.mean(exr)
        features['ExR_std'] = np.std(exr)
        features['ExR_median'] = np.median(exr)
        
        # ExR histogram
        exr_hist, _ = np.histogram(exr, bins=20, range=(-1, 1))
        exr_hist = exr_hist / exr_hist.sum()
        for i in range(min(10, len(exr_hist))):
            features[f'ExR_hist_{i}'] = exr_hist[i]
        
        # VARI (Visible Atmospherically Resistant Index)
        vari = (g - r) / (g + r - b + 1e-6)
        features['VARI_mean'] = np.mean(vari)
        features['VARI_std'] = np.std(vari)
        
        # CIVE (Color Index of Vegetation Extraction)
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
        features['CIVE_mean'] = np.mean(cive)
        features['CIVE_std'] = np.std(cive)
        
        # NDI (Normalized Difference Index)
        ndi = (g - r) / (g + r + 1e-6)
        features['NDI_mean'] = np.mean(ndi)
        features['NDI_std'] = np.std(ndi)
        
        # Green coverage
        green_mask = exg > 0
        features['green_coverage'] = np.sum(green_mask) / green_mask.size
        
        return features
    
    def extract_glcm_features(self, img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Extract GLCM texture features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        # Extract texture features
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            try:
                values = graycoprops(glcm, prop)
                features[f'GLCM_{prop}_mean'] = np.mean(values)
                features[f'GLCM_{prop}_std'] = np.std(values)
            except:
                features[f'GLCM_{prop}_mean'] = 0
                features[f'GLCM_{prop}_std'] = 0
        
        return features
    
    def extract_all_features(self, img_path, metadata=None):
        """Extract all features"""
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        features = {}
        
        # Extract color features
        color_features = self.extract_color_features(img)
        features.update(color_features)
        
        # Extract vegetation indices
        veg_features = self.calculate_vegetation_indices(img)
        features.update(veg_features)
        
        # Extract GLCM texture features
        glcm_features = self.extract_glcm_features(img)
        features.update(glcm_features)
        
        # Add metadata features
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
                    features['Month'] = date.month
                    features['DayOfYear'] = date.dayofyear
                except:
                    features['Month'] = 0
                    features['DayOfYear'] = 0
        
        return features
    
    def prepare_features(self, df, is_training=True):
        """Prepare feature matrix"""
        print("Extracting features...")
        
        all_features = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing: {idx}/{len(df)}")
            
            try:
                features = self.extract_all_features(row['image_path'], row)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing image {row['image_path']}: {e}")
                # Use zero features
                all_features.append({})
        
        feature_df = pd.DataFrame(all_features)
        
        # Handle categorical features
        categorical_cols = ['State', 'Species']
        for col in categorical_cols:
            if col in feature_df.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    feature_df[col] = self.label_encoders[col].fit_transform(
                        feature_df[col].fillna('Unknown')
                    )
                else:
                    # Only encode if we have the encoder from training
                    if col in self.label_encoders:
                        feature_df[col] = feature_df[col].fillna('Unknown')
                        # Handle unseen categories
                        unknown_mask = ~feature_df[col].isin(self.label_encoders[col].classes_)
                        feature_df.loc[unknown_mask, col] = 'Unknown'
                        feature_df[col] = self.label_encoders[col].transform(feature_df[col])
        
        # Fill missing values
        feature_df = feature_df.fillna(0)
        
        if is_training:
            # Store all extracted feature names first
            all_feature_names = feature_df.columns.tolist()
            
            # Filter to only selected features
            available_selected = [f for f in self.selected_features if f in all_feature_names]
            missing_selected = [f for f in self.selected_features if f not in all_feature_names]
            
            if missing_selected:
                print(f"Warning: {len(missing_selected)} selected features not found in extracted features:")
                print(f"  {missing_selected}")
            
            self.feature_names = available_selected
            print(f"\nUsing {len(self.feature_names)} selected features out of {len(all_feature_names)} total extracted features")
            
        else:
            # For test set, add missing features that were present during training
            missing_features = set(self.feature_names) - set(feature_df.columns)
            if missing_features:
                print(f"Adding {len(missing_features)} missing features: {sorted(list(missing_features))[:10]}...")
                for feature_name in missing_features:
                    feature_df[feature_name] = 0
        
        # Ensure correct column order
        return feature_df[self.feature_names].values
    
    def train(self, train_csv_path, image_dir='train'):
        """Train models"""
        print("Loading training data...")
        df = pd.read_csv(train_csv_path)
        
        # Add full image path
        df['image_path'] = df['image_path'].apply(lambda x: str(Path(image_dir) / Path(x).name))
        
        print(f"Training data shape: {df.shape}")
        print(f"Target variables: {df['target_name'].unique()}")
        
        # Extract features ONCE for all unique images
        print("\n" + "="*50)
        print("Extracting features from unique images (will be reused for all targets)")
        print("="*50)
        unique_images = df.drop_duplicates(subset=['image_path']).copy()
        print(f"Number of unique images: {len(unique_images)}")
        
        X_all = self.prepare_features(unique_images, is_training=True)
        
        # Create a mapping from image_path to feature index
        image_to_idx = {img_path: idx for idx, img_path in enumerate(unique_images['image_path'])}
        
        # Train separate model for each target
        for target_name in self.target_names:
            print(f"\n{'='*50}")
            print(f"Training target: {target_name}")
            print(f"{'='*50}")
            
            # Filter data for current target
            target_df = df[df['target_name'] == target_name].copy()
            
            if len(target_df) == 0:
                print(f"Warning: No training data found for {target_name}")
                continue
            
            # Get features by mapping image paths to indices
            feature_indices = [image_to_idx[img_path] for img_path in target_df['image_path']]
            X = X_all[feature_indices]
            y = target_df['target'].values
            
            # Split train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[target_name] = scaler
            
            # Train XGBoost model
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
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            self.models[target_name] = model
            
            # Evaluation
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            print(f"Training R²: {train_r2:.4f}")
            print(f"Validation R²: {val_r2:.4f}")
            
            # Print all feature importance scores
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nAll Feature Importance Scores:")
            print(importance_df.to_string(index=False))
        
        print("\nModel training completed!")
        return self
    
    def predict(self, test_csv_path, image_dir='test'):
        """Predict test set"""
        print("Loading test data...")
        test_df = pd.read_csv(test_csv_path)
        
        # Get unique images
        unique_images = test_df[['image_path']].drop_duplicates()
        unique_images['image_path'] = unique_images['image_path'].apply(
            lambda x: str(Path(image_dir) / Path(x).name)
        )
        
        print(f"Number of test images: {len(unique_images)}")
        
        # Extract features for each image (only once)
        X_test = self.prepare_features(unique_images, is_training=False)
        
        # Predict for each target
        predictions = []
        
        for idx, row in test_df.iterrows():
            if idx % 100 == 0:
                print(f"Prediction progress: {idx}/{len(test_df)}")
            
            target_name = row['target_name']
            
            if target_name not in self.models:
                print(f"Warning: Model not found for {target_name}")
                predictions.append(0.0)
                continue
            
            # Find corresponding image index
            img_path = str(Path(image_dir) / Path(row['image_path']).name)
            img_idx = unique_images[unique_images['image_path'] == img_path].index[0]
            img_idx = unique_images.index.get_loc(img_idx)
            
            # Get features and standardize
            X = X_test[img_idx:img_idx+1]
            X_scaled = self.scalers[target_name].transform(X)
            
            # Predict
            pred = self.models[target_name].predict(X_scaled)[0]
            predictions.append(max(0, pred))  # Ensure non-negative predictions
        
        # Create submission file
        submission = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'target': predictions
        })
        
        return submission


# Usage example
if __name__ == "__main__":
    # Initialize predictor
    predictor = PastureBiomassPredictor()
    
    # Train models
    predictor.train('train.csv', image_dir='train')
    
    # Predict test set
    submission = predictor.predict('test.csv', image_dir='test')
    
    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print("\nPrediction completed! Submission file saved as submission.csv")
    print(f"Submission file shape: {submission.shape}")
    print(f"\nPrediction statistics:")
    print(submission.groupby(submission['sample_id'].str.split('__').str[1])['target'].describe())