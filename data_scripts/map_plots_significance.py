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

def bootstrap_sample(data, n_iterations=1000):
    """Bootstrap resampling with replacement."""
    means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_iterations)
    ])
    return np.percentile(means, [2.5, 97.5])  # 95% confidence interval

if __name__ == '__main__':

    uk = [-11, 10, 45, 65] # longitude min, longitude max, latitude min, latitude max

    # Load data ---------------------------

    # ERA5 analogues
    era5_analogues = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues_72hour_mean.nc')
    era5_analogues['tp'] = era5_analogues['tp'].sel(lat=slice(uk[3], uk[2]), lon=slice(uk[0], uk[1]))

    # RACMO analogues

    # PGW - no ens members available atm
    # pgw = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_clean.nc')
    # pgw['tp'] = (((pgw.tp.sel(time=slice('2023-10-19 00', '2023-10-22 00'))*3*3600).sum(dim='time'))/1e5).sel(lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1]))

    # IFS
    tmp = []
    base_dir='/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/MED-R/EXP/{}/EU025/sfc/pf'
    climates = ['1870', '1950', 'present', 'future1']
    for e, exp in enumerate(['pi', 'pi_1950', 'curr', 'incr']):
        tmp.append(xr.open_mfdataset(os.path.join(base_dir.format(exp), '*.nc'), engine='netcdf4', preprocess=preproc_ds_v2).expand_dims(climate=[climates[e]]))
    ifs = xr.concat(tmp, dim='climate')
    ifs['tp'] = ((ifs.tp.sel(time='2023-10-22 00') - ifs.tp.sel(time='2023-10-19 00'))*1000).sel(latitude=slice(uk[3], uk[2]), longitude=slice(uk[0], uk[1]))

    # MICAS
    micas = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_clean.nc')
    micas['tp'] = micas.tp.sel(time=slice('2023-10-19 12', '2023-10-21 12'), lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1])).sum(dim='time')*24*3600

    # Bootstrapping ------------------------------
    
    # ERA5
    print('Now calculating for ERA5')
    bootstrapped = xr.apply_ufunc(
        bootstrap_sample, 
        era5_analogues.tp-era5_analogues.tp.sel(climate='present'), 
        input_core_dims=[['member']],  # Modify based on your data dimensions
        vectorize=True,  # Ensures element-wise computation
        dask="parallelized",  # Enables parallel execution if data is chunked
        output_core_dims=[["percentile"]],  # Output contains percentiles
    )
    era5_analogues_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5]).to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/significance/era5_analogues_tp_sign_map.nc')


    # IFS
    print('Now calculating for IFS')
    bootstrapped = xr.apply_ufunc(
        bootstrap_sample,
        ifs.tp-ifs.tp.sel(climate='present'),
        input_core_dims=[['number']],
        vectorize=True,
        dask="parallelized",
        output_core_dims=[["percentile"]],
        dask_gufunc_kwargs={"output_sizes": {"percentile": 2}}
        )
    ifs_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5]).to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/significance/ifs_tp_sign_map.nc')

    # MICAS
    print('Now calculating for MICAS')
    bootstrapped = xr.apply_ufunc(
        bootstrap_sample,
        micas.tp-micas.tp.sel(climate='present'),
        input_core_dims=[['member']],
        vectorize=True,
        dask="parallelized",
        output_core_dims=[["percentile"]],
    )
    micas_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5]).to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/significance/micas_tp_sign_map.nc')