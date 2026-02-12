'''
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
Found 2 scenes.
Loading and resampling data (Anonymous Mode)...
Cleaning cloud cover...
Calculating median (this triggers the download)...

=== Final Result Preview ===
        Date  Latitude  Longitude  nbart_blue  nbart_green  nbart_red  \
0  2015/7/31     -31.2      145.9       362.0        568.0      758.0   

   nbart_red_edge_1  nbart_red_edge_2  nbart_red_edge_3  nbart_nir_1  \
0            1123.0            1651.0            1740.0       1803.0   

   nbart_nir_2  nbart_swir_2  nbart_swir_3  
0       1910.0        2410.0        1693.0  
CSV saved successfully.
'''