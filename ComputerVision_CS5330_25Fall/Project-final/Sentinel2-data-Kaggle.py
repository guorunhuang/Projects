'''
Using the STAC API to get data from Kaggle Notebook platform
DEA (Digital Earth Australia) provides a STAC API. This standard protocol allows you to index and download data
stored on AWS S3 directly from anywhere (Kaggle, Colab, or your local machine) without needing a direct database connection.

the odc-stac library. Its syntax is very similar to datacube, allowing for a seamless transition.
 (https://explorer.dea.ga.gov.au/stac)
'''
!pip install odc-stac pystac-client pandas xarray

import sys
import numpy as np
import pandas as pd
import xarray as xr
from pystac_client import Client
from odc.stac import load

# 1. 配置参数 (0.1度 x 0.1度)
lat_center = -31.2
lon_center = 145.9
# 生成 BBox (Bounding Box): [min_lon, min_lat, max_lon, max_lat]
bbox = [lon_center - 0.05, lat_center - 0.05, lon_center + 0.05, lat_center + 0.05]
time_range = "2015-07-01/2015-07-31"

# 2. 连接 DEA STAC API
print("Connecting to DEA STAC API...")
catalog = Client.open("https://explorer.dea.ga.gov.au/stac")

# 3. 搜索数据
# collections=['ga_s2am_ard_3'] 对应 Sentinel-2A ARD 数据
query = catalog.search(
    bbox=bbox,
    collections=['ga_s2am_ard_3'],
    datetime=time_range
)
items = list(query.items())
print(f"Found {len(items)} scenes.")

# 4. 加载数据 (odc.stac.load 替代了 datacube.load)
# 这里的参数和之前几乎一样，它会自动从 AWS S3 下载并重采样
print("Loading and resampling data (this might take a moment)...")

# 定义波段映射 (STAC 的名称可能略有不同，但 odc-stac通常能自动识别)
# DEA STAC 里的波段名称通常是: 'blue', 'green', 'red', 'red_edge_1', etc.
# oa_fmask 也在其中
bands = [
    'nbart_blue', 'nbart_green', 'nbart_red', 
    'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3', 
    'nbart_nir_1', 'nbart_nir_2', 
    'nbart_swir_2', 'nbart_swir_3', 
    'oa_fmask'
]

ds = load(
    items,
    bands=bands,
    crs="EPSG:3577",          # 投影到澳洲标准坐标系
    resolution=20,            # 设置分辨率为 20m (自动重采样)
    resampling="nearest",     # 最近邻插值
    groupby="solar_day",      # 按天合并
    chunks={}                 # 使用 Dask 延迟加载，防止 Kaggle 内存溢出
)

# 5. 数据清洗 (Cloud Masking)
print("Cleaning cloud cover...")
# 同样使用 oa_fmask: 1=valid
mask_clear = ds.oa_fmask == 1
ds_clean = ds.where(mask_clear, other=np.nan).drop_vars(['oa_fmask'])

# 6. 计算中位数
print("Calculating median...")
# 这一步会触发下载和计算
median_reflectance = ds_clean.median(dim=['x', 'y']).compute()

# 7. 格式化输出 (完全复用之前的逻辑)
df = median_reflectance.to_dataframe()
df = df.dropna(how='all')
df = df.reset_index()

# 格式化日期 2015/7/2
df['Date'] = df['time'].apply(lambda x: f"{x.year}/{x.month}/{x.day}")

# 添加经纬度
df['Latitude'] = lat_center
df['Longitude'] = lon_center

# 整理列
cols = ['Date', 'Latitude', 'Longitude'] + [c for c in df.columns if c not in ['Date', 'Latitude', 'Longitude', 'time', 'spatial_ref']]
df_final = df[cols]

print("\n=== Final Result Preview ===")
print(df_final)

# 保存 CSV (Kaggle 输出目录)
df_final.to_csv('sentinel2_dea_kaggle.csv', index=False)
print("CSV saved to output directory.")