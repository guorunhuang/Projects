'''
Collecting odc-stac
  Downloading odc_stac-0.5.0-py3-none-any.whl.metadata (5.7 kB)
Collecting pystac-client
  Downloading pystac_client-0.9.0-py3-none-any.whl.metadata (3.1 kB)
Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.3)
Requirement already satisfied: xarray in /usr/local/lib/python3.11/dist-packages (2025.7.1)
Collecting affine (from odc-stac)
  Downloading affine-2.4.0-py3-none-any.whl.metadata (4.0 kB)
Collecting odc-geo>=0.4.7 (from odc-stac)
  Downloading odc_geo-0.5.0-py3-none-any.whl.metadata (5.3 kB)
Collecting odc-loader>=0.6.0 (from odc-stac)
  Downloading odc_loader-0.6.0-py3-none-any.whl.metadata (1.4 kB)
Collecting rasterio!=1.3.0,!=1.3.1,>=1.0.0 (from odc-stac)
  Downloading rasterio-1.4.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.1 kB)
Requirement already satisfied: dask[array] in /usr/local/lib/python3.11/dist-packages (from odc-stac) (2024.12.1)
Requirement already satisfied: numpy>=1.20.0 in /usr/local/lib/python3.11/dist-packages (from odc-stac) (1.26.4)
Collecting pystac<2,>=1.0.0 (from odc-stac)
  Downloading pystac-1.14.1-py3-none-any.whl.metadata (4.7 kB)
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
Downloading odc_stac-0.5.0-py3-none-any.whl (43 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.2/43.2 kB 1.3 MB/s eta 0:00:00
Downloading pystac_client-0.9.0-py3-none-any.whl (41 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.8/41.8 kB 2.2 MB/s eta 0:00:00
Downloading odc_geo-0.5.0-py3-none-any.whl (159 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 159.0/159.0 kB 4.8 MB/s eta 0:00:00
Downloading odc_loader-0.6.0-py3-none-any.whl (57 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.3/57.3 kB 3.0 MB/s eta 0:00:00
Downloading pystac-1.14.1-py3-none-any.whl (207 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.7/207.7 kB 10.3 MB/s eta 0:00:00
Downloading rasterio-1.4.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (22.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 22.2/22.2 MB 29.6 MB/s eta 0:00:0000:0100:01m
Downloading affine-2.4.0-py3-none-any.whl (15 kB)
Installing collected packages: affine, pystac, pystac-client, rasterio, odc-geo, odc-loader, odc-stac
Successfully installed affine-2.4.0 odc-geo-0.5.0 odc-loader-0.6.0 odc-stac-0.5.0 pystac-1.14.1 pystac-client-0.9.0 rasterio-1.4.3
Connecting to DEA STAC API...
Found 2 scenes.
Loading and resampling data (this might take a moment)...
Cleaning cloud cover...
Calculating median...
Aborting load due to failure while reading: s3://dea-public-data/baseline/ga_s2am_ard_3/55/JDF/2015/07/31/20160822T043304/ga_s2am_nbart_3-2-1_55JDF_2015-07-31_final_band07.tif:1
Aborting load due to failure while reading: s3://dea-public-data/baseline/ga_s2am_ard_3/55/JDF/2015/07/31/20160822T043304/ga_s2am_oa_3-2-1_55JDF_2015-07-31_final_fmask.tif:1
Aborting load due to failure while reading: s3://dea-public-data/baseline/ga_s2am_ard_3/55/JDF/2015/07/31/20160822T043304/ga_s2am_nbart_3-2-1_55JDF_2015-07-31_final_band06.tif:1
Aborting load due to failure while reading: s3://dea-public-data/baseline/ga_s2am_ard_3/55/JDF/2015/07/31/20160822T043304/ga_s2am_nbart_3-2-1_55JDF_2015-07-31_final_band11.tif:1
---------------------------------------------------------------------------
CPLE_AWSInvalidCredentialsError           Traceback (most recent call last)
rasterio/_base.pyx in rasterio._base.DatasetBase.__init__()

rasterio/_base.pyx in rasterio._base.open_dataset()

rasterio/_err.pyx in rasterio._err.exc_wrap_pointer()

CPLE_AWSInvalidCredentialsError: AWS_SECRET_ACCESS_KEY and AWS_NO_SIGN_REQUEST configuration options not defined, and /root/.aws/credentials not filled

During handling of the above exception, another exception occurred:

RasterioIOError                           Traceback (most recent call last)
/tmp/ipykernel_47/4160417868.py in <cell line: 0>()
     68 print("Calculating median...")
     69 
---> 70 median_reflectance = ds_clean.median(dim=['x', 'y']).compute()
     71 
     72 

/usr/local/lib/python3.11/dist-packages/xarray/core/dataset.py in compute(self, **kwargs)
    713         """
    714         new = self.copy(deep=False)
--> 715         return new.load(**kwargs)
    716 
    717     def _persist_inplace(self, **kwargs) -> Self:

/usr/local/lib/python3.11/dist-packages/xarray/core/dataset.py in load(self, **kwargs)
    540 
    541             # evaluate all the chunked arrays simultaneously
--> 542             evaluated_data: tuple[np.ndarray[Any, Any], ...] = chunkmanager.compute(
    543                 *lazy_data.values(), **kwargs
    544             )

/usr/local/lib/python3.11/dist-packages/xarray/namedarray/daskmanager.py in compute(self, *data, **kwargs)
     83         from dask.array import compute
     84 
---> 85         return compute(*data, **kwargs)  # type: ignore[no-untyped-call, no-any-return]
     86 
     87     def persist(self, *data: Any, **kwargs: Any) -> tuple[DaskArray | Any, ...]:

/usr/local/lib/python3.11/dist-packages/dask/base.py in compute(traverse, optimize_graph, scheduler, get, *args, **kwargs)
    658 
    659     with shorten_traceback():
--> 660         results = schedule(dsk, keys, **kwargs)
    661 
    662     return repack([f(r, *a) for r, (f, a) in zip(results, postcomputes)])

/usr/local/lib/python3.11/dist-packages/odc/loader/_builder.py in _dask_loader_tyx(srcs, gbt, iyx, prefix_dims, postfix_dims, cfg, rdr, env, load_state, selection)
    494     with rdr.restore_env(env, load_state):
    495         for ti, ti_srcs in enumerate(srcs):
--> 496             _fill_nd_slice(
    497                 ti_srcs, gbox, cfg, chunk[ti], ydim=ydim, selection=selection
    498             )

/usr/local/lib/python3.11/dist-packages/odc/loader/_builder.py in _fill_nd_slice(srcs, dst_gbox, cfg, dst, ydim, selection)
    571 
    572     src, *rest = srcs
--> 573     yx_roi, pix = src.read(cfg, dst_gbox, dst=dst, selection=selection)
    574     assert len(yx_roi) == 2
    575     assert pix.ndim == dst.ndim

/usr/local/lib/python3.11/dist-packages/odc/loader/_rio.py in read(self, cfg, dst_geobox, dst, selection)
    138         selection: Optional[ReaderSubsetSelection] = None,
    139     ) -> tuple[tuple[slice, slice], np.ndarray]:
--> 140         return rio_read(self._src, cfg, dst_geobox, dst=dst, selection=selection)
    141 
    142 

/usr/local/lib/python3.11/dist-packages/odc/loader/_rio.py in rio_read(src, cfg, dst_geobox, dst, selection)
    565                 src.band,
    566             )
--> 567             raise e
    568     except rasterio.errors.RasterioError as e:
    569         if cfg.fail_on_error:

/usr/local/lib/python3.11/dist-packages/odc/loader/_rio.py in rio_read(src, cfg, dst_geobox, dst, selection)
    551     try:
    552         return fixup_out(
--> 553             _rio_read(src, cfg, dst_geobox, prep_dst(dst), selection=selection)
    554         )
    555     except (

/usr/local/lib/python3.11/dist-packages/odc/loader/_rio.py in _rio_read(src, cfg, dst_geobox, dst, selection)
    599     ttol = 0.9 if cfg.nearest else 0.05
    600 
--> 601     with rasterio.open(src.uri, "r", sharing=False) as rdr:
    602         assert isinstance(rdr, rasterio.DatasetReader)
    603         ovr_idx: Optional[int] = None

/usr/local/lib/python3.11/dist-packages/rasterio/env.py in wrapper(*args, **kwds)
    461 
    462         with env_ctor(session=session):
--> 463             return f(*args, **kwds)
    464 
    465     return wrapper

/usr/local/lib/python3.11/dist-packages/rasterio/__init__.py in open(fp, mode, driver, width, height, count, crs, transform, dtype, nodata, sharing, opener, **kwargs)
    354 
    355             if mode == "r":
--> 356                 dataset = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
    357             elif mode == "r+":
    358                 dataset = get_writer_for_path(path, driver=driver)(

rasterio/_base.pyx in rasterio._base.DatasetBase.__init__()

RasterioIOError: AWS_SECRET_ACCESS_KEY and AWS_NO_SIGN_REQUEST configuration options not defined, and /root/.aws/credentials not filled
'''