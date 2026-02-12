'''
帮我写一个能在DEA Sandbox运行的代码，获取2015年7月，经纬度31.2S上下浮动0.5° 145.9E左右浮动0.5°, 
所有波段，Sentinel-2A，将resolution 10m 聚合到 20m的数据（Nearest Neighbour最近邻插值法）。
然后处理数据，（1）数据清洗 (Cloud Masking)。读取oa_fmask这个波段，把标记为“云”或“云影”的像素点的反射率设为 NaN (空值)，（
2）提取反射率 (Extraction)，把我给出的范围，提取所有像素的值，计算中位数
'''

# version 3
'''
这个代码是不是2015年7月里面，有哪些日子有卫星数据，就把当天的数据各个波段计算中位数？输出的时候有Data这个字段吗？没有的话加上，
格式要求为2015/7/2。此外，输出有经纬度这个字段吗？没有的话也加上
回答你的问题关于计算逻辑：是的，你的理解完全正确。 代码逻辑 group_by='solar_day' 配合 .median(dim=['x', 'y']) 的作用就是：
在 2015 年 7 月的时间范围内，找出所有有卫星飞过的日子，然后针对每一天，把该区域内所有像素（去除云之后）的反射率取中位数。
关于 Date 字段：默认输出的 DataFrame Date（时间）通常是 索引（Index），格式是 2015-07-02 00:00:00。如果不处理，
它不会叫 "Date"，且格式带时分秒。我们需要手动处理把它变成你要求的 2015/7/2 格式。关于经纬度字段：默认输出是没有经纬度的。
原因： 因为你执行了 median(dim=['x', 'y'])，这意味着你把空间上的所有像素“压缩”成了一个值。原来的 $x$ 和 $y$ 坐标在计算中位数时
被“消耗”掉了。解决办法： 既然代表的是这一整块区域的中位数，我们可以在输出表中人为加上这块区域的 中心点经纬度，
以满足你的格式要求。修改后的代码这份代码在之前的逻辑上增加了 “数据格式化” 的步骤，完美满足你的输出要求。
代码输出样式预览运行上述代码后，你会得到类似这样的表格：
index  Date  Latitude  Longitude  nbart_blue  nbart_green  nbart_red...
0  2015/7/2  -31.2  145.9  345.0  412.5  560.0...
1  2015/7/7  -31.2  145.9  320.1  400.2  550.3...
2  2015/7/12  -31.2  145.9  350.4  415.8  565.1...
'''
import datacube
import numpy as np
import pandas as pd
import xarray as xr

# 1. 初始化
dc = datacube.Datacube(app='s2_biomass_analysis_formatted')

# 2. 定义范围 (0.1度 x 0.1度)
lat_center = -31.2
lon_center = 145.9
lat_range = (lat_center - 0.05, lat_center + 0.05)
lon_range = (lon_center - 0.05, lon_center + 0.05)
time_range = '2015-07'

# 3. 加载数据
print("正在加载并重采样数据...")
measurements = [
    'nbart_blue', 'nbart_green', 'nbart_red', 
    'nbart_red_edge_1', 'nbart_red_edge_2', 'nbart_red_edge_3', 
    'nbart_nir_1', 'nbart_nir_2', 
    'nbart_swir_2', 'nbart_swir_3', 
    'oa_fmask'
]

ds = dc.load(
    product='ga_s2am_ard_3',
    measurements=measurements,
    x=lon_range,
    y=lat_range,
    time=time_range,
    output_crs='EPSG:3577',
    resolution=(-20, 20),
    resampling='nearest',
    group_by='solar_day'
)

# 4. 数据清洗 (云掩膜)
print("正在清洗云层...")
mask_clear = ds.oa_fmask == 1
ds_clean = ds.where(mask_clear, other=np.nan).drop_vars(['oa_fmask'])

# 5. 计算区域中位数
print("正在计算中位数...")
median_reflectance = ds_clean.median(dim=['x', 'y'])

# 6. === 关键修改：格式化输出表格 ===

# (A) 转为 DataFrame
df = median_reflectance.to_dataframe()

# (B) 删除全为空的行（没有卫星数据的日子）
df = df.dropna(how='all')

# (C) 重置索引，把 'time' 从索引变成一列
df = df.reset_index()

# (D) 添加 Date 字段，格式化为 2015/7/2 (去除前导0)
# lambda x: f"{x.year}/{x.month}/{x.day}" 能保证月份和日期没有0开头
df['Date'] = df['time'].apply(lambda x: f"{x.year}/{x.month}/{x.day}")

# (E) 添加 经纬度 字段 (使用区域中心点)
df['Latitude'] = lat_center
df['Longitude'] = lon_center

# (F) 整理列顺序：把 Date, Latitude, Longitude 放到最前面，去掉原始的 time 列
cols = ['Date', 'Latitude', 'Longitude'] + [c for c in df.columns if c not in ['Date', 'Latitude', 'Longitude', 'time']]
df_final = df[cols]

# 7. 打印最终结果
print("\n=== 最终结果预览 ===")
print(df_final)

# 可选：导出为 CSV
# df_final.to_csv('sentinel2_median_201507.csv', index=False)


# version 2
'''
那我改一下吧，经纬度范围改成0.1度 x 0.1度
太棒了！将范围缩小到 0.1° x 0.1°（约 10km x 10km）是一个非常明智的选择。这将极大地减少内存占用，计算速度会飞快，而且结果更容易验证。

以下是更新后的代码。我已经帮你把经纬度范围调整好了（中心点向外各扩 0.05度，总宽 0.1度）。

你可以直接复制粘贴到 DEA Sandbox 的 Jupyter Notebook 中运行。
'''
import datacube
import numpy as np
import xarray as xr

# 1. 初始化 Datacube
dc = datacube.Datacube(app='s2_biomass_analysis_small_area')

# 2. 定义参数 (已更新为 0.1度 x 0.1度 范围)
# 中心点: 31.2S (-31.2), 145.9E
# 为了得到 0.1度的总跨度，我们在中心点基础上 +/- 0.05度
lat_range = (-31.25, -31.15) 
lon_range = (145.85, 145.95)
time_range = '2015-07'

# 定义需要加载的波段
measurements = [
    'nbart_blue',       # B2
    'nbart_green',      # B3
    'nbart_red',        # B4
    'nbart_red_edge_1', # B5
    'nbart_red_edge_2', # B6
    'nbart_red_edge_3', # B7
    'nbart_nir_1',      # B8
    'nbart_nir_2',      # B8A
    'nbart_swir_2',     # B11
    'nbart_swir_3',     # B12
    'oa_fmask'          # 云掩膜
]

print(f"正在加载数据...\n经度范围: {lon_range}\n纬度范围: {lat_range}")

# 3. 加载 Sentinel-2A 数据
# 因为范围变小了，这里不需要 dask_chunks 也能轻松跑通，直接加载进内存即可
ds = dc.load(
    product='ga_s2am_ard_3',        # Sentinel-2A
    measurements=measurements,
    x=lon_range,
    y=lat_range,
    time=time_range,
    output_crs='EPSG:3577',         # 投影到澳洲标准坐标系(米)
    resolution=(-20, 20),           # 重采样到 20m 像素
    resampling='nearest',           # 【关键】使用最近邻插值法
    group_by='solar_day'            # 合并同一天的轨道数据
)

print(f"原始数据加载完成。数据维度: {ds.sizes}")

# 4. 数据清洗 (Cloud Masking) 函数
def apply_cloud_mask(dataset):
    # 读取 fmask 波段
    fmask = dataset.oa_fmask
    
    # DEA fmask 代码: 0=nodata, 1=valid, 2=cloud, 3=shadow, 4=snow, 5=water
    # 论文要求: 去除 "cloud" (2) 和 "cloud shadow" (3)
    # 这里我们只保留 "valid" (1)，这通常是处理牧场生物量最安全的做法
    mask_clear = fmask == 1
    
    # 将不符合条件的像素设为 NaN
    dataset_masked = dataset.where(mask_clear, other=np.nan)
    
    # 丢弃 fmask 波段，因为它已经完成了使命
    return dataset_masked.drop_vars(['oa_fmask'])

# 执行清洗
ds_clean = apply_cloud_mask(ds)

# 5. 提取反射率并计算中位数
# 计算这个 0.1x0.1度 范围内，所有像素在每一天的中位数
# 结果将是一个时间序列：每一天对应 10 个波段的中位数值
median_reflectance = ds_clean.median(dim=['x', 'y'])

# 6. 打印结果
print("\n=== 2015年7月 反射率中位数提取结果 ===")
# 将结果转换为 Pandas DataFrame 方便查看
df = median_reflectance.to_dataframe()

# 过滤掉全都是 NaN 的日期 (即那天全是云，或者那天没卫星经过)
df_clean = df.dropna(how='all')

print(df_clean)

# 可选：计算 NDVI 验证数据是否合理
# NDVI = (NIR - Red) / (NIR + Red)
if not df_clean.empty:
    ndvi = (df_clean['nbart_nir_1'] - df_clean['nbart_red']) / (df_clean['nbart_nir_1'] + df_clean['nbart_red'])
    print("\n=== 计算的 NDVI ===")
    print(ndvi)
else:
    print("\n警告：该时间段内没有有效数据（可能全是云或没有卫星过境）。")


# version 1
'''
这是一个可以直接在 Digital Earth Australia (DEA) Sandbox (JupyterLab 环境) 中运行的完整 Python 代码。

代码说明
这段代码执行了以下操作：

加载数据 (dc.load)：

指定产品为 ga_s2am_ard_3 (Sentinel-2A)。

空间范围：根据你提供的中心点 (31.2S, 145.9E) 上下左右各浮动 0.5°（即 1° x 1° 的巨大范围）。

时间：2015年7月。

重采样：设置 resolution=(-20, 20) 将分辨率统一为 20米，并指定 resampling='nearest' (最近邻插值) 以及 output_crs='EPSG:3577' (澳洲标准投影，单位为米)。

云掩膜 (Cloud Masking)：

使用 oa_fmask 波段。

将云 (2) 和云影 (3) 的像素设为 NaN。

计算中位数 (Median)：

计算该区域内所有有效像素在每个时间点上的空间中位数。

重要警告：数据量过大
你请求的范围是 1度 x 1度（大约 100km x 100km）。在 20米分辨率下，这是一张 5000 x 5000 像素 的超大图像。加载整个月的序列可能会导致 Sandbox 内存溢出 (OOM)。

为了防止崩溃，我在代码中使用了 Dask (延迟加载)。代码不会立即下载图片，而是在最后计算中位数（.compute()）时才开始处理。请耐心等待运行结果。
'''
import datacube
import numpy as np
import xarray as xr
from odc.algo import enum_to_bool

# 1. 初始化 Datacube
dc = datacube.Datacube(app='s2_biomass_analysis')

# 2. 定义参数
# 中心点: 31.2S, 145.9E. 范围 +/- 0.5度
lat_range = (-31.7, -30.7) # 注意南半球纬度是负数
lon_range = (145.4, 146.4)
time_range = '2015-07'

# 定义需要加载的波段 (对应论文中的波段 + fmask)
# 论文提到: Visible, NIR, SWIR bands. 
# DEA中的命名通常为 nbart_{band_name}
measurements = [
    'nbart_blue',       # B2
    'nbart_green',      # B3
    'nbart_red',        # B4
    'nbart_red_edge_1', # B5
    'nbart_red_edge_2', # B6
    'nbart_red_edge_3', # B7
    'nbart_nir_1',      # B8
    'nbart_nir_2',      # B8A (Narrow NIR)
    'nbart_swir_2',     # B11
    'nbart_swir_3',     # B12
    'oa_fmask'          # 用于云检测的掩膜波段
]

print("正在配置数据加载任务...")

# 3. 加载 Sentinel-2A 数据
# 使用 dask_chunks 实现延迟加载，防止内存爆炸
ds = dc.load(
    product='ga_s2am_ard_3',        # Sentinel-2A ARD 数据
    measurements=measurements,
    x=lon_range,
    y=lat_range,
    time=time_range,
    output_crs='EPSG:3577',         # GDA94 / Australian Albers (以米为单位)
    resolution=(-20, 20),           # 聚合到 20m 分辨率
    resampling='nearest',           # 使用最近邻插值法 (Nearest Neighbour)
    group_by='solar_day',           # 将同一次过境的数据合并
    dask_chunks={'time': 1, 'x': 2000, 'y': 2000} # 使用 Dask 分块处理
)

print(f"数据加载对象已创建: {ds}")

# 4. 数据清洗 (Cloud Masking)
def apply_cloud_mask(dataset):
    # 获取 fmask 波段
    fmask = dataset.oa_fmask
    
    # DEA fmask 定义:
    # 0: null, 1: valid, 2: cloud, 3: cloud shadow, 4: snow, 5: water
    
    # 我们希望保留的是 'valid' (1), 可能还有 'snow' (4) 或 'water' (5) 视情况而定
    # 论文明确说要去除 Cloud(2) 和 Cloud Shadow(3)
    # 下面的逻辑是：创建一个掩膜，只要不是云(2)且不是云影(3)，就为 True
    
    # 方式A: 使用位运算或isin (比较直观)
    # mask_good = ~fmask.isin([2, 3]) 
    
    # 方式B: 只保留确定的“晴朗陆地” (更严格，通常推荐)
    # 这样会把水体也掩膜掉，如果是牧场生物量，通常只保留 1 (Valid)
    mask_clear = fmask == 1
    
    # 将不满足条件的地方设为 NaN (对于 int 类型数据，xarray会自动转为 float 以存储 NaN)
    # keep_attrs=True 保持元数据
    dataset_masked = dataset.where(mask_clear, other=np.nan)
    
    # 删除不再需要的 fmask 波段，节省内存
    return dataset_masked.drop_vars(['oa_fmask'])

# 应用掩膜
# 注意：此时因为用了 Dask，这里还没有真正计算，只是构建了计算图
ds_clean = apply_cloud_mask(ds)

print("已构建云掩膜计算图，准备计算中位数...")

# 5. 提取反射率并计算中位数
# 计算每个时间步(time)下，所有空间像素(x, y)的中位数
# 这将把巨大的地图压缩成每个日期一个数值
median_reflectance = ds_clean.median(dim=['x', 'y'])

# 6. 触发计算 (Compute)
# 这一步会真正开始下载数据并处理，可能需要几分钟
result = median_reflectance.compute()

# 7. 打印结果
print("\n=== 处理完成 ===")
print("2015年7月各波段反射率中位数 (已过滤云/影):")
print(result)

# 如果想看具体数值的 DataFrame 格式：
df_result = result.to_dataframe()
print("\nDataFrame 预览:")
print(df_result)

# 提示：计算 NDVI (如果需要)
# NDVI = (NIR - Red) / (NIR + Red)
# 注意：DEA数据的数值通常放大了10000倍 (0-10000)，计算NDVI时不需要缩放，直接除即可
# 这里的 Red 是 nbart_red, NIR 是 nbart_nir_1
ndvi = (result.nbart_nir_1 - result.nbart_red) / (result.nbart_nir_1 + result.nbart_red)
print("\nNDVI 每日中位数:")
print(ndvi)