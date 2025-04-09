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
    ifs = bb.data.Data.get_fba_ifs()
    ifs['tp'] = ((ifs.tp.sel(time='2023-10-22 00') - ifs.tp.sel(time='2023-10-19 00'))*1000)
    ifs['t2m'] = ifs.t2m.sel(time=slice('2023-10-19 00', '2023-10-22 00'), latitude=slice(uk[3], uk[2]), longitude=slice(uk[0], uk[1])).mean(dim='time')

    # MICAS
    micas = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_clean.nc')
    micas['tp'] = micas.tp.sel(time=slice('2023-10-19 12', '2023-10-21 12'), lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1])).sum(dim='time')*24*3600
    micas['tas'] = micas.tas.sel(time=slice('2023-10-19 12', '2023-10-21 12'), lat=slice(uk[2], uk[3]), lon=slice(uk[0], uk[1])).mean(dim='time')

    # Calculate precip scaling ---------------------------
    
    # Bootstrap samples etc
    n = 10000
    aberdeen = [-4, -2, 55.5, 57.2] # longitude min, longitude max, latitude min, latitude max

    # ERA5
    # print('Now calculating for ERA5')
    # era5_precip_scaling = calc_precip_scaling(era5_analogues.tp.sel(lat=slice(aberdeen[3], aberdeen[2]), lon=slice(aberdeen[0], aberdeen[1])).mean(['lat', 'lon']), 
    #                                       era5_analogues.t2m.sel(lat=slice(aberdeen[3], aberdeen[2]), lon=slice(aberdeen[0], aberdeen[1])).mean(['lat', 'lon']))
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample, 
    #     era5_precip_scaling, 
    #     input_core_dims=[['member']],  # Modify based on your data dimensions
    #     vectorize=True,  # Ensures element-wise computation
    #     dask="parallelized",  # Enables parallel execution if data is chunked
    #     output_core_dims=[["percentile"]],  # Output contains percentiles
    #     kwargs={"n_iterations": n}  # Pass n=10 to bootstrap_sample
    # )
    # era5_analogues_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5])
    # print(f"ERA5 precip scaling: {era5_analogues_sign.sel(percentile=2.5, climate='1950').values, era5_analogues_sign.sel(percentile=97.5, climate='1950').values}")

    # IFS
    print('Now calculating for IFS')
    ifs_precip_scaling = calc_precip_scaling(ifs.tp.sel(latitude=slice(aberdeen[3], aberdeen[2]), longitude=slice(aberdeen[0], aberdeen[1])).mean(['latitude', 'longitude']), 
                                         ifs.t2m.sel(latitude=slice(aberdeen[3], aberdeen[2]), longitude=slice(aberdeen[0], aberdeen[1])).mean(['latitude', 'longitude']))
    bootstrapped = xr.apply_ufunc(
        bootstrap_sample,
        ifs_precip_scaling,
        input_core_dims=[['number']],
        vectorize=True,
        dask="parallelized",
        output_core_dims=[["percentile"]],
        dask_gufunc_kwargs={"output_sizes": {"percentile": 2}},
        kwargs={"n_iterations": n}
        )
    ifs_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5])
    print(f"IFS precip scaling: {ifs_sign.sel(percentile=2.5, climate='1950').values, ifs_sign.sel(percentile=97.5, climate='1950').values}")

    # MICAS
    # print('Now calculating for MICAS')
    # micas_precip_scaling = calc_precip_scaling(micas.tp.sel(lat=slice(aberdeen[2], aberdeen[3]), lon=slice(aberdeen[0], aberdeen[1])).mean(['lat', 'lon']), 
    #                                        micas.tas.sel(lat=slice(aberdeen[2], aberdeen[3]), lon=slice(aberdeen[0], aberdeen[1])).mean(['lat', 'lon']))
    # bootstrapped = xr.apply_ufunc(
    #     bootstrap_sample,
    #     micas_precip_scaling,
    #     input_core_dims=[['member']],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_core_dims=[["percentile"]],
    #     kwargs={"n_iterations": n}
    # )
    # micas_sign = bootstrapped.assign_coords(percentile=[2.5, 97.5])
    # print(f"MICAS precip scaling: {micas_sign.sel(percentile=2.5, climate='1870').values, micas_sign.sel(percentile=97.5, climate='1870').values}")