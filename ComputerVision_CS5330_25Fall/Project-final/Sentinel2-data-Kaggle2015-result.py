''' 2015 NSW Only
Requirement already satisfied: odc-stac in /usr/local/lib/python3.11/dist-packages (0.5.0)
Requirement already satisfied: pystac-client in /usr/local/lib/python3.11/dist-packages (0.9.0)
Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.3)
Requirement already satisfied: xarray in /usr/local/lib/python3.11/dist-packages (2025.7.1)
Requirement already satisfied: affine in /usr/local/lib/python3.11/dist-packages (from odc-stac) (2.4.0)
Requirement already satisfied: odc-geo>=0.4.7 in /usr/local/lib/python3.11/dist-packages (from odc-stac) (0.5.0)
Requirement already satisfied: odc-loader>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from odc-stac) (0.6.0)
Requirement already satisfied: rasterio!=1.3.0,!=1.3.1,>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from odc-stac) (1.4.3)
Requirement already satisfied: dask[array] in /usr/local/lib/python3.11/dist-packages (from odc-stac) (2024.12.1)
Requirement already satisfied: numpy>=1.20.0 in /usr/local/lib/python3.11/dist-packages (from odc-stac) (1.26.4)
Requirement already satisfied: pystac<2,>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from odc-stac) (1.14.1)
Requirement already satisfied: toolz in /usr/local/lib/python3.11/dist-packages (from odc-stac) (1.1.0)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.11/dist-packages (from odc-stac) (4.15.0)
Requirement already satisfied: requests>=2.28.2 in /usr/local/lib/python3.11/dist-packages (from pystac-client) (2.32.5)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pystac-client) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.2)
Requirement already satisfied: packaging>=24.1 in /usr/local/lib/python3.11/dist-packages (from xarray) (25.0)
Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy>=1.20.0->odc-stac) (1.3.8)
Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy>=1.20.0->odc-stac) (1.2.4)
Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy>=1.20.0->odc-stac) (0.1.1)
Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy>=1.20.0->odc-stac) (2025.3.0)
Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy>=1.20.0->odc-stac) (2022.3.0)
Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy>=1.20.0->odc-stac) (2.4.1)
Requirement already satisfied: cachetools in /usr/local/lib/python3.11/dist-packages (from odc-geo>=0.4.7->odc-stac) (6.2.1)
Requirement already satisfied: pyproj>=3.0.0 in /usr/local/lib/python3.11/dist-packages (from odc-geo>=0.4.7->odc-stac) (3.7.1)
Requirement already satisfied: shapely in /usr/local/lib/python3.11/dist-packages (from odc-geo>=0.4.7->odc-stac) (2.1.2)
Requirement already satisfied: jsonschema~=4.18 in /usr/local/lib/python3.11/dist-packages (from pystac[validation]>=1.10.0->pystac-client) (4.25.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pystac-client) (1.17.0)
Requirement already satisfied: attrs in /usr/local/lib/python3.11/dist-packages (from rasterio!=1.3.0,!=1.3.1,>=1.0.0->odc-stac) (25.4.0)
Requirement already satisfied: certifi in /usr/local/lib/python3.11/dist-packages (from rasterio!=1.3.0,!=1.3.1,>=1.0.0->odc-stac) (2025.10.5)
Requirement already satisfied: click>=4.0 in /usr/local/lib/python3.11/dist-packages (from rasterio!=1.3.0,!=1.3.1,>=1.0.0->odc-stac) (8.3.0)
Requirement already satisfied: cligj>=0.5 in /usr/local/lib/python3.11/dist-packages (from rasterio!=1.3.0,!=1.3.1,>=1.0.0->odc-stac) (0.7.2)
Requirement already satisfied: click-plugins in /usr/local/lib/python3.11/dist-packages (from rasterio!=1.3.0,!=1.3.1,>=1.0.0->odc-stac) (1.1.1.2)
Requirement already satisfied: pyparsing in /usr/local/lib/python3.11/dist-packages (from rasterio!=1.3.0,!=1.3.1,>=1.0.0->odc-stac) (3.0.9)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.28.2->pystac-client) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.28.2->pystac-client) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.28.2->pystac-client) (2.5.0)
Requirement already satisfied: cloudpickle>=3.0.0 in /usr/local/lib/python3.11/dist-packages (from dask[array]->odc-stac) (3.1.2)
Requirement already satisfied: fsspec>=2021.09.0 in /usr/local/lib/python3.11/dist-packages (from dask[array]->odc-stac) (2025.10.0)
Requirement already satisfied: partd>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from dask[array]->odc-stac) (1.4.2)
Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.11/dist-packages (from dask[array]->odc-stac) (6.0.3)
Requirement already satisfied: importlib_metadata>=4.13.0 in /usr/local/lib/python3.11/dist-packages (from dask[array]->odc-stac) (8.7.0)
Requirement already satisfied: zipp>=3.20 in /usr/local/lib/python3.11/dist-packages (from importlib_metadata>=4.13.0->dask[array]->odc-stac) (3.23.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema~=4.18->pystac[validation]>=1.10.0->pystac-client) (2025.4.1)
Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.11/dist-packages (from jsonschema~=4.18->pystac[validation]>=1.10.0->pystac-client) (0.36.2)
Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema~=4.18->pystac[validation]>=1.10.0->pystac-client) (0.26.0)
Requirement already satisfied: locket in /usr/local/lib/python3.11/dist-packages (from partd>=1.4.0->dask[array]->odc-stac) (1.0.0)
Requirement already satisfied: onemkl-license==2025.3.0 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy>=1.20.0->odc-stac) (2025.3.0)
Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy>=1.20.0->odc-stac) (2024.2.0)
Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy>=1.20.0->odc-stac) (2022.3.0)
Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy>=1.20.0->odc-stac) (1.4.0)
Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy>=1.20.0->odc-stac) (2024.2.0)
Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy>=1.20.0->odc-stac) (2024.2.0)
Connecting to DEA STAC API...
Searching for scenes...
Found 22 scenes.
Loading and resampling data (Anonymous Mode)...
Cleaning cloud cover...
Calculating median (this triggers the download)...

=== Final Result Preview ===
         Date  Latitude  Longitude  nbart_blue  nbart_green  nbart_red  \
0   2015/7/31     -31.2      145.9       362.0        568.0      758.0   
1   2015/8/10     -31.2      145.9       338.0        578.0      808.0   
2   2015/8/20     -31.2      145.9       372.0        594.0      856.0   
3   2015/8/30     -31.2      145.9       392.0        627.0      890.0   
4    2015/9/9     -31.2      145.9       430.0        654.0      937.0   
5   2015/9/19     -31.2      145.9       442.0        664.0      975.0   
6   2015/9/29     -31.2      145.9       464.0        686.0     1055.0   
7  2015/11/18     -31.2      145.9       432.0        658.0     1078.0   
8  2015/11/28     -31.2      145.9       463.0        708.0     1162.0   
9  2015/12/28     -31.2      145.9       434.0        660.0     1104.0   

   nbart_red_edge_1  nbart_red_edge_2  nbart_red_edge_3  nbart_nir_1  \
0            1123.0            1651.0            1740.0       1803.0   
1            1164.0            1696.0            1812.0       1874.0   
2            1214.0            1740.0            1856.0       1927.0   
3            1272.0            1824.0            1942.0       2027.0   
4            1305.0            1840.0            1990.0       2067.0   
5            1320.0            1782.0            1963.0       2038.0   
6            1361.0            1764.0            1981.0       2051.0   
7            1349.0            1709.0            1974.0       2057.0   
8            1444.0            1796.0            2060.0       2156.0   
9            1375.0            1700.0            1957.0       2057.0   

   nbart_nir_2  nbart_swir_2  nbart_swir_3  
0       1910.0        2410.0        1693.0  
1       1981.0        2547.0        1819.0  
2       2024.0        2603.0        1908.0  
3       2111.0        2665.0        1954.0  
4       2161.0        2747.0        2015.0  
5       2131.0        2767.0        2032.0  
6       2154.0        2867.0        2111.0  
7       2145.0        2971.0        2260.0  
8       2234.0        3092.0        2371.0  
9       2141.0        3011.0        2255.0  
CSV saved successfully.
'''