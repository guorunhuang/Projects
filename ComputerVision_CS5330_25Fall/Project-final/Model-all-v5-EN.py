'''
Hand-crafted + Metadata + XGBoost 

extract 28 features that have high contribute score, instead of all
'''
import time
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
    
    def extract_all_features(self, img_path, metadata=None):
        """Extract only the selected features"""
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        features = {}
        
        # Normalize RGB for vegetation indices
        b, g, r = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
        total = r + g + b + 1e-6
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
        
        # ExG (Excess Green) - needed for ExG features
        exg = 2 * g_norm - r_norm - b_norm
        features['ExG_mean'] = np.mean(exg)
        features['ExG_median'] = np.median(exg)
        
        # ExG histogram (bins 0, 1, 5, 7, 8, 9)
        exg_hist, _ = np.histogram(exg, bins=20, range=(-1, 1))
        exg_hist = exg_hist / exg_hist.sum()
        features['ExG_hist_0'] = exg_hist[0]
        features['ExG_hist_1'] = exg_hist[1]
        features['ExG_hist_5'] = exg_hist[5]
        features['ExG_hist_7'] = exg_hist[7]
        features['ExG_hist_8'] = exg_hist[8]
        features['ExG_hist_9'] = exg_hist[9]
        
        # ExR (Excess Red) - needed for ExR features
        exr = 1.4 * r_norm - g_norm
        features['ExR_mean'] = np.mean(exr)
        features['ExR_median'] = np.median(exr)
        
        # ExR histogram (bins 4, 9)
        exr_hist, _ = np.histogram(exr, bins=20, range=(-1, 1))
        exr_hist = exr_hist / exr_hist.sum()
        features['ExR_hist_4'] = exr_hist[4]
        features['ExR_hist_9'] = exr_hist[9]
        
        # VARI (Visible Atmospherically Resistant Index)
        vari = (g - r) / (g + r - b + 1e-6)
        features['VARI_mean'] = np.mean(vari)
        
        # CIVE (Color Index of Vegetation Extraction)
        cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
        features['CIVE_mean'] = np.mean(cive)
        
        # HSV features - only specific ones needed
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # HSV_S_std
        s_channel = hsv[:, :, 1]
        features['HSV_S_std'] = np.std(s_channel)
        
        # HSV_S_hist_4
        s_hist = cv2.calcHist([s_channel], [0], None, [32], [0, 256])
        s_hist = s_hist.flatten() / s_hist.sum()
        features['HSV_S_hist_4'] = s_hist[4]
        
        # HSV_H histograms (bins 4, 5, 7)
        h_channel = hsv[:, :, 0]
        h_hist = cv2.calcHist([h_channel], [0], None, [32], [0, 180])
        h_hist = h_hist.flatten() / h_hist.sum()
        features['HSV_H_hist_4'] = h_hist[4]
        features['HSV_H_hist_5'] = h_hist[5]
        features['HSV_H_hist_7'] = h_hist[7]
        
        # GLCM texture features - only specific ones needed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           levels=256, symmetric=True, normed=True)
        
        try:
            homogeneity_values = graycoprops(glcm, 'homogeneity')
            features['GLCM_homogeneity_mean'] = np.mean(homogeneity_values)
        except:
            features['GLCM_homogeneity_mean'] = 0
        
        try:
            correlation_values = graycoprops(glcm, 'correlation')
            features['GLCM_correlation_std'] = np.std(correlation_values)
        except:
            features['GLCM_correlation_std'] = 0
        
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
                except:
                    features['Month'] = 0
        
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
            self.feature_names = feature_df.columns.tolist()
            print(f"Total features extracted: {len(self.feature_names)}")
        else:
            # For test set, add missing features that were present during training
            missing_features = set(self.feature_names) - set(feature_df.columns)
            if missing_features:
                print(f"Adding {len(missing_features)} missing features: {sorted(list(missing_features))[:10]}...")
                for feature_name in missing_features:
                    feature_df[feature_name] = 0
        
        # Ensure correct column order
        return feature_df[self.feature_names].values
    
    def train(self, train_csv_path, image_dir='/kaggle/input/csiro-biomass/train'):
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
    
    def predict(self, test_csv_path, image_dir='/kaggle/input/csiro-biomass/test'):
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
    start_time = time.time()
    
    # Initialize predictor
    predictor = PastureBiomassPredictor()
    
    # Train models
    predictor.train('/kaggle/input/csiro-biomass/train.csv', image_dir='/kaggle/input/csiro-biomass/train')
    
    train_end_time = time.time()
    train_duration = train_end_time - start_time
    print(f"\nTrain takes: {train_duration:.2f} seconds")

    # Predict test set
    submission = predictor.predict('/kaggle/input/csiro-biomass/test.csv', image_dir='/kaggle/input/csiro-biomass/test')
    
    # Save submission file
    submission.to_csv('submission.csv', index=False)
    print("\nPrediction completed! Submission file saved as submission.csv")
    print(f"Submission file shape: {submission.shape}")
    print(f"\nPrediction statistics:")
    print(submission.groupby(submission['sample_id'].str.split('__').str[1])['target'].describe())
    
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nTotal process takes: {total_duration:.2f} seconds")