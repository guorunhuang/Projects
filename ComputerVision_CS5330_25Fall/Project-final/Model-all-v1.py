# 绿色覆盖率
# green_mask = exg > 0

import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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
        
    def extract_color_features(self, img):
        """提取RGB、HSV和色相分布特征"""
        features = {}
        
        # RGB特征
        for i, color in enumerate(['R', 'G', 'B']):
            channel = img[:, :, i]
            features[f'{color}_mean'] = np.mean(channel)
            features[f'{color}_std'] = np.std(channel)
            features[f'{color}_median'] = np.median(channel)
            
            # RGB直方图特征
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()
            for j in range(min(10, len(hist))):  # 前10个bin
                features[f'{color}_hist_{j}'] = hist[j]
        
        # HSV特征
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, channel_name in enumerate(['H', 'S', 'V']):
            channel = hsv[:, :, i]
            features[f'HSV_{channel_name}_mean'] = np.mean(channel)
            features[f'HSV_{channel_name}_std'] = np.std(channel)
            features[f'HSV_{channel_name}_median'] = np.median(channel)
            
            # HSV直方图
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256 if i > 0 else 180])
            hist = hist.flatten() / hist.sum()
            for j in range(min(8, len(hist))):
                features[f'HSV_{channel_name}_hist_{j}'] = hist[j]
        
        return features
    
    def calculate_vegetation_indices(self, img):
        """计算植被指数：ExG, ExR, VARI, CIVE, NDI"""
        features = {}
        
        # 归一化RGB
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
        
        # ExG直方图
        exg_hist, _ = np.histogram(exg, bins=20, range=(-1, 1))
        exg_hist = exg_hist / exg_hist.sum()
        for i in range(min(10, len(exg_hist))):
            features[f'ExG_hist_{i}'] = exg_hist[i]
        
        # ExR (Excess Red)
        exr = 1.4 * r_norm - g_norm
        features['ExR_mean'] = np.mean(exr)
        features['ExR_std'] = np.std(exr)
        features['ExR_median'] = np.median(exr)
        
        # ExR直方图
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
        
        # 绿色覆盖率
        green_mask = exg > 0
        features['green_coverage'] = np.sum(green_mask) / green_mask.size
        
        return features
    
    def extract_glcm_features(self, img, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """提取GLCM纹理特征"""
        features = {}
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算GLCM
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        # 提取纹理特征
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
        """提取所有特征"""
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        features = {}
        
        # 提取颜色特征
        color_features = self.extract_color_features(img)
        features.update(color_features)
        
        # 提取植被指数
        veg_features = self.calculate_vegetation_indices(img)
        features.update(veg_features)
        
        # 提取GLCM纹理特征
        glcm_features = self.extract_glcm_features(img)
        features.update(glcm_features)
        
        # 添加元数据特征
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
        """准备特征矩阵"""
        print("提取特征中...")
        
        all_features = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{len(df)}")
            
            try:
                features = self.extract_all_features(row['image_path'], row)
                all_features.append(features)
            except Exception as e:
                print(f"处理图像 {row['image_path']} 时出错: {e}")
                # 使用零特征
                all_features.append({})
        
        feature_df = pd.DataFrame(all_features)
        
        # 处理分类特征
        categorical_cols = ['State', 'Species']
        for col in categorical_cols:
            if col in feature_df.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    feature_df[col] = self.label_encoders[col].fit_transform(
                        feature_df[col].fillna('Unknown')
                    )
                else:
                    feature_df[col] = feature_df[col].fillna('Unknown')
                    # 处理未见过的类别
                    unknown_mask = ~feature_df[col].isin(self.label_encoders[col].classes_)
                    feature_df.loc[unknown_mask, col] = 'Unknown'
                    feature_df[col] = self.label_encoders[col].transform(feature_df[col])
        
        # 填充缺失值
        feature_df = feature_df.fillna(0)
        
        if is_training:
            self.feature_names = feature_df.columns.tolist()
        
        return feature_df[self.feature_names].values
    
    def train(self, train_csv_path, image_dir='train'):
        """训练模型"""
        print("加载训练数据...")
        df = pd.read_csv(train_csv_path)
        
        # 添加完整图像路径
        df['image_path'] = df['image_path'].apply(lambda x: str(Path(image_dir) / Path(x).name))
        
        print(f"训练数据形状: {df.shape}")
        print(f"目标变量: {df['target_name'].unique()}")
        
        # 为每个目标训练单独的模型
        for target_name in self.target_names:
            print(f"\n{'='*50}")
            print(f"训练目标: {target_name}")
            print(f"{'='*50}")
            
            # 过滤当前目标的数据
            target_df = df[df['target_name'] == target_name].copy()
            
            if len(target_df) == 0:
                print(f"警告: 没有找到 {target_name} 的训练数据")
                continue
            
            # 提取特征
            X = self.prepare_features(target_df, is_training=True)
            y = target_df['target'].values
            
            # 分割训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[target_name] = scaler
            
            # 训练XGBoost模型
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
            
            # 评估
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            print(f"训练集 R²: {train_r2:.4f}")
            print(f"验证集 R²: {val_r2:.4f}")
            
            # 显示特征重要性（前20个）
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n前20个重要特征:")
            print(importance_df.head(20).to_string(index=False))
        
        print("\n模型训练完成!")
        return self
    
    def predict(self, test_csv_path, image_dir='test'):
        """预测测试集"""
        print("加载测试数据...")
        test_df = pd.read_csv(test_csv_path)
        
        # 获取唯一的图像
        unique_images = test_df[['image_path']].drop_duplicates()
        unique_images['image_path'] = unique_images['image_path'].apply(
            lambda x: str(Path(image_dir) / Path(x).name)
        )
        
        print(f"测试图像数量: {len(unique_images)}")
        
        # 为每个图像提取特征（只提取一次）
        X_test = self.prepare_features(unique_images, is_training=False)
        
        # 为每个目标进行预测
        predictions = []
        
        for idx, row in test_df.iterrows():
            if idx % 100 == 0:
                print(f"预测进度: {idx}/{len(test_df)}")
            
            target_name = row['target_name']
            
            if target_name not in self.models:
                print(f"警告: 没有找到 {target_name} 的模型")
                predictions.append(0.0)
                continue
            
            # 找到对应的图像索引
            img_path = str(Path(image_dir) / Path(row['image_path']).name)
            img_idx = unique_images[unique_images['image_path'] == img_path].index[0]
            img_idx = unique_images.index.get_loc(img_idx)
            
            # 获取特征并标准化
            X = X_test[img_idx:img_idx+1]
            X_scaled = self.scalers[target_name].transform(X)
            
            # 预测
            pred = self.models[target_name].predict(X_scaled)[0]
            predictions.append(max(0, pred))  # 确保预测值非负
        
        # 创建提交文件
        submission = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'target': predictions
        })
        
        return submission
    
    def plot_feature_importance(self, target_name, top_n=20):
        """绘制特征重要性图"""
        if target_name not in self.models:
            print(f"未找到 {target_name} 的模型")
            return
        
        model = self.models[target_name]
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features for {target_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df


# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = PastureBiomassPredictor()
    
    # 训练模型
    predictor.train('train.csv', image_dir='train')
    
    # 对每个目标绘制特征重要性
    for target_name in predictor.target_names:
        if target_name in predictor.models:
            print(f"\n{target_name} 的特征重要性:")
            importance_df = predictor.plot_feature_importance(target_name, top_n=20)
    
    # 预测测试集
    submission = predictor.predict('test.csv', image_dir='test')
    
    # 保存提交文件
    submission.to_csv('submission.csv', index=False)
    print("\n预测完成! 提交文件已保存为 submission.csv")
    print(f"提交文件形状: {submission.shape}")
    print(f"\n预测统计:")
    print(submission.groupby(submission['sample_id'].str.split('__').str[1])['target'].describe())