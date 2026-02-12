'''
Using the STAC API to get data from Kaggle Notebook platform
DEA (Digital Earth Australia) provides a STAC API. This standard protocol allows you to index and download data
stored on AWS S3 directly from anywhere (Kaggle, Colab, or your local machine) without needing a direct database connection.

the odc-stac library. Its syntax is very similar to datacube, allowing for a seamless transition.
 (https://explorer.dea.ga.gov.au/stac)

DEA in AWS S3 , dea-public-data, Unsigned Request）

'''
!pip install odc-stac pystac-client pandas xarray

import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
from pystac_client import Client
from odc.stac import load, configure_rio 

# ==========================================
# 关键修复：配置 AWS S3 为匿名访问模式
# ==========================================
# 这告诉 GDAL/Rasterio 不要寻找密钥，直接匿名读取公共数据
configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

# 1. 配置参数 (0.1度 x 0.1度)
lat_center = -31.2
lon_center = 145.9
# 生成 BBox: [min_lon, min_lat, max_lon, max_lat]
bbox = [lon_center - 0.05, lat_center - 0.05, lon_center + 0.05, lat_center + 0.05]
time_range = "2015-07-01/2015-07-31"

# 2. 连接 DEA STAC API
print("Connecting to DEA STAC API...")
catalog = Client.open("https://explorer.dea.ga.gov.au/stac")

# 3. 搜索数据
print("Searching for scenes...")
query = catalog.search(
    bbox=bbox,
    collections=['ga_s2am_ard_3'], # Sentinel-2A
    datetime=time_range
)
items = list(query.items())
print(f"Found {len(items)} scenes.")

# 4. 加载数据
print("Loading and resampling data (Anonymous Mode)...")

# 定义波段
bands = [
    'nbart_blue', 'nbart_green', 'nbart_red', 
    'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3', 
    'nbart_nir_1', 'nbart_nir_2', 
    'nbart_swir_2', 'nbart_swir_3', 
    'oa_fmask'
]

# 加载数据
ds = load(
    items,
    bands=bands,
    crs="EPSG:3577",          # 投影到澳洲标准坐标系
    resolution=20,            # 分辨率 20m
    resampling="nearest",     # 最近邻插值
    groupby="solar_day",      # 按天合并
    chunks={}                 # 使用 Dask 延迟加载
)

# 5. 数据清洗 (Cloud Masking)
print("Cleaning cloud cover...")
mask_clear = ds.oa_fmask == 1
ds_clean = ds.where(mask_clear, other=np.nan).drop_vars(['oa_fmask'])

# 6. 计算中位数
print("Calculating median (this triggers the download)...")
# 这一步现在应该能正常下载了
median_reflectance = ds_clean.median(dim=['x', 'y']).compute()

# 7. 格式化输出
df = median_reflectance.to_dataframe()
df = df.dropna(how='all')
df = df.reset_index()

# 格式化日期
if not df.empty:
    df['Date'] = df['time'].apply(lambda x: f"{x.year}/{x.month}/{x.day}")
    df['Latitude'] = lat_center
    df['Longitude'] = lon_center

    # 整理列
    cols = ['Date', 'Latitude', 'Longitude'] + [c for c in df.columns if c not in ['Date', 'Latitude', 'Longitude', 'time', 'spatial_ref']]
    df_final = df[cols]

    print("\n=== Final Result Preview ===")
    print(df_final)

    # 保存 CSV
    df_final.to_csv('sentinel2_dea_kaggle.csv', index=False)
    print("CSV saved successfully.")
else:
    print("No valid data found for this period (possibly all clouds).")