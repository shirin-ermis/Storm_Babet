import numpy as np
import xarray as xr
import pandas as pd
import os
import seaborn as sns
import random 
import dask
import babet as bb

random.seed(10)
dask.config.set(**{'array.slicing.split_large_chunks': True})

def preproc_ds_v2(ds):
    """
    Main pre-processing function
    Writtten by Nick Leach and Shirin Ermis.

    Input:
    ------
    ds: xarray dataset

    Output:
    -------
    ds1: xarray dataset with inidate dimension added
    """
    # remove any dimensions of length 1
    ds1 = ds.squeeze()
    # set up aux data
    inidate = pd.to_datetime(ds1.time[0].values)
    # expand dimensions to include extra info
    if not 'inidate' in ds1:
        ds1 = ds1.expand_dims({'inidate': [inidate]}).copy()

    if not 'number' in ds1:
        ds1 = ds1.expand_dims({'number': [0]}).copy()

    # put time dimension at front
    ds1 = ds1.transpose('time', ...)
    return ds1

def bootstrap_sample(data, n_iterations=10000):
    """Bootstrap resampling with replacement."""
    means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_iterations)
    ])
    return np.percentile(means, [2.5, 97.5])  # 95% confidence interval

if __name__ == '__main__':

    uk = [-11, 10, 45, 65] # longitude min, longitude max, latitude min, latitude max

    # Load data ---------------------------

    # PGW 
    # pgw_ens = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_clean_ensemble.nc')
    # pgw_ens['tp'] = (((pgw_ens.tp.sel(time=slice('2023-10-19 00', '2023-10-22 00'))*3*3600).sum(dim='time'))/1e5).sel(lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1]))

    # ERA5 analogues
    # era5_analogues = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues_72hour_mean.nc')
    # era5_analogues['tp'] = era5_analogues['tp'].sel(lat=slice(uk[3], uk[2]), lon=slice(uk[0], uk[1]))

    # RACMO analogues - no analoues available atm
    # racmo_tp = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_tp_72hour_mean.nc')
    # racmo_tp['tp'] = racmo_tp['tp'].sel(lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1]))

    # PGW - no ens members available atm
    # pgw = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_clean_ensemble.nc')
    # pgw['tp'] = (((pgw.tp.sel(time=slice('2023-10-19 00', '2023-10-22 00'))*3*3600).sum(dim='time'))/1e5).sel(lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1]))

    # IFS
    # ifs = bb.data.Data.get_fba_ifs()
    # ifs = ifs.sel(latitude=slice(uk[3], uk[2]), longitude=slice(uk[0], uk[1]))
    # ifs['tp'] = ((ifs.tp.sel(time='2023-10-22 00') - ifs.tp.sel(time='2023-10-19 00'))*1000)

    # MICAS
    # micas = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_clean.nc')
    # micas['tp'] = micas.tp.sel(time=slice('2023-10-19 12', '2023-10-21 12'), lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1])).sum(dim='time')*24*3600

    # RACMO
    # racmo_tp = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_tp_72hour_mean.nc')
    # print(racmo_tp.dims)

    # Bootstrapping ------------------------------
    
    # ERA5
    # print('Now calculating for ERA5')
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample, 
    #     era5_analogues.tp-era5_analogues.tp.sel(climate='present'), 
    #     input_core_dims=[['member']],  # Modify based on your data dimensions
    #     vectorize=True,  # Ensures element-wise computation
    #     dask="parallelized",  # Enables parallel execution if data is chunked
    #     output_core_dims=[["percentile"]],  # Output contains percentiles
    # )
    # era5_analogues_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5]).to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/significance/era5_analogues_tp_sign_map.nc')
    # # print(f"Size in memory: {era5_analogues_sign.nbytes / 1024**2:.2f} MB")

    # IFS
    # rechunked = (ifs.tp-ifs.tp.sel(climate='present')).chunk({'number': -1})
    # print('Rechunk successful')
    # print('Now calculating for IFS')
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample,
    #     rechunked,
    #     input_core_dims=[['number']],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_core_dims=[["percentile"]],
    #     dask_gufunc_kwargs={"output_sizes": {"percentile": 2}}
    #     )
    # ifs_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5]).to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/significance/ifs_tp_sign_map.nc')
    # print(f"Size in memory: {ifs_sign.nbytes / 1024**2:.2f} MB")

    # MICAS
    # print('Now calculating for MICAS')
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample,
    #     micas.tp-micas.tp.sel(climate='present'),
    #     input_core_dims=[['member']],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_core_dims=[["percentile"]],
    # )
    # micas_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5]).to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/significance/micas_tp_sign_map.nc')
    # # print(f"Size in memory: {micas_sign.nbytes / 1024**2:.2f} MB")

    # PGW
    # print('Now calculating for PGW')
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample,
    #     pgw_ens.tp-pgw_ens.tp.sel(climate='present'),
    #     input_core_dims=[['member']],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_core_dims=[["percentile"]],
    # )
    # pgw_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5]).to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/significance/pgw_tp_sign_map.nc')

    # RACMO
    # print('Now calculating for RACMO')
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample,
    #     racmo_tp.tp-racmo_tp.tp.sel(climate='present'),
    #     input_core_dims=[['member']],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_core_dims=[["percentile"]],
    # )