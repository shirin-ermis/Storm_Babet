import numpy as np
import xarray as xr
import pandas as pd
import os
import seaborn as sns
import random 
import dask
import babet as bb

from map_plots_significance import preproc_ds_v2, bootstrap_sample

random.seed(10)
dask.config.set(**{'array.slicing.split_large_chunks': True})

def calc_precip_scaling(tp, t2m):
    """
    Calculate precipitation scaling factor.

    Input:
    ------
    tp: xarray DataArray, precipitation, should include dimension climate
    t2m: xarray DataArray, 2m temperature, should include dimension climate
    
    Output:
    -------
    scaling: xarray DataArray, precipitation scaling factor for each climate
    """
    # Calculate fractional change in precipitation for each memer
    d_tp = tp - tp.sel(climate='present')
    d_t2m = t2m - t2m.sel(climate='present')
    scaling = (d_tp/tp.sel(climate='present'))/d_t2m
    return scaling

if __name__ == '__main__':

    uk = [-11, 10, 45, 65] # longitude min, longitude max, latitude min, latitude max

    # Load data ---------------------------

    # ERA5 analogues
    era5_analogues = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues_72hour_mean.nc')
    era5_analogues['tp'] = era5_analogues['tp'].sel(lat=slice(uk[3], uk[2]), lon=slice(uk[0], uk[1]))
    era5_analogues['t2m'] = era5_analogues['t2m'].sel(lat=slice(uk[3], uk[2]), lon=slice(uk[0], uk[1]))

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
    ifs = ifs.sel(latitude=slice(uk[3], uk[2]), longitude=slice(uk[0], uk[1]))
    ifs['tp'] = ((ifs.tp.sel(time='2023-10-22 00') - ifs.tp.sel(time='2023-10-19 00'))*1000)

    # MICAS
    micas = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_clean.nc')
    micas['tp'] = micas.tp.sel(time=slice('2023-10-19 12', '2023-10-21 12'), lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1])).sum(dim='time')*24*3600

    # Calculate precip scaling ---------------------------
    
    # ERA5
    print('Now calculating for ERA5')
    era5_precip_scaling = calc_precip_scaling(era5_analogues.tp, era5_analogues.t2m.mean(dim='time'))
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample, 
    #     era5_precip_scaling, 
    #     input_core_dims=[['member']],  # Modify based on your data dimensions
    #     vectorize=True,  # Ensures element-wise computation
    #     dask="parallelized",  # Enables parallel execution if data is chunked
    #     output_core_dims=[["percentile"]],  # Output contains percentiles
    # )
    # era5_analogues_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5])