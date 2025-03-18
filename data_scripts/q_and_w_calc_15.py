import xarray as xr
import os
import babet as bb
import dask
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from 09_vert_integrated_advection import combine_xarray_dict
dask.config.set(**{'array.slicing.split_large_chunks': True})

def combine_xarray_dict(exp, dim_name="experiment"):
    """
    Combine a dictionary of xarray datasets into a single dataset along a new dimension.

    Parameters
    ----------
    exp : dict of {str: xr.Dataset}
        Dictionary where keys are experiment names and values are xarray datasets.
    dim_name : str, optional
        Name of the new dimension to combine along (default is "experiment").

    Returns
    -------
    xr.Dataset
        A combined dataset with an additional dimension corresponding to dictionary keys.
    """
    # Convert dictionary to a list of datasets and extract keys
    datasets = list(exp.values())
    keys = list(exp.keys())

    # Add the new dimension as a coordinate
    expanded_datasets = [ds.assign_coords({dim_name: key}) for ds, key in zip(datasets, keys)]

    # Concatenate along the new dimension
    combined_ds = xr.concat(expanded_datasets, dim=dim_name)

    return combined_ds

def calc_average_w(ds, lower=850, upper=250):
    
    ds = ds.copy(deep=True).sel(level=slice(upper, lower))
    
    # Compute the pressure thickness (ΔP) between levels
    # The last dimension should be the pressure dimension
    delta_p = ds.level.diff('level').rename('delta_p')

    # Align ΔP with the original dataset (it will be one less in size)
    delta_p = delta_p.assign_coords(p=ds['level'].isel(level=slice(1, None)))

    # Compute the mass-weighted vertical velocity
    weighted_w = ds['w'].isel(level=slice(1, None)) * delta_p

    # Compute the mass-weighted average
    mass_weighted_average = (weighted_w.sum(dim='level') / delta_p.sum(dim='level')).rename('mass_weighted_w')

    # Add the result to your dataset
    ds['mass_weighted_w'] = mass_weighted_average
    
    return ds

def calc_qs(ds):

    # Constants
    epsilon = 0.622  # Ratio of gas constants (R_v / R_d)
    R_d = 287.05  # Gas constant for dry air (J/kg/K)
    R_v = 461.5  # Gas constant for water vapor (J/kg/K)

    # Assume your dataset `ds` contains:
    # - 'T': temperature (Kelvin)
    # - 'p': pressure (Pa)

    # Convert temperature from Kelvin to Celsius
    T_c = ds['t'] - 273.15

    # Compute saturation vapor pressure (e_s) in hPa
    e_s_hpa = 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))

    # Convert saturation vapor pressure to Pa
    e_s = e_s_hpa * 100  # Convert hPa to Pa

    # Compute saturation specific humidity (q_s)
    q_s = (epsilon * e_s) / (ds['level'] - (1 - epsilon) * e_s)

    # Add q_s to the dataset
    ds['q_s'] = q_s
    
    return ds

def integate_qs(ds, lower=850, upper=250):
    # Constants
    g = 9.81  # gravitational acceleration in m/s^2

    # Assume `ds` is your xarray Dataset with:
    # - 'q_s': saturation specific humidity (kg/kg)
    # - 'p': pressure levels (Pa), sorted from top to bottom

    # Calculate saturation specific humidity at each level
    ds = calc_qs(ds).copy(deep=True).sel(level=slice(upper, lower))

    # Compute pressure thickness (Δp) for each layer
    delta_p = ds['level'].diff('level').rename('delta_p')

    # Align Δp to match dimensions of q_s (it will have one fewer element)
    delta_p = delta_p.assign_coords(p=ds['level'].isel(level=slice(1, None)))

    # Compute the weighted saturation specific humidity (q_s * Δp)
    weighted_qs = ds['q_s'].isel(level=slice(1, None)) * delta_p

    # Integrate vertically by summing over the pressure dimension
    vertically_integrated_qs = (weighted_qs.sum(dim='level') / g).rename('Q_s')

    # Add the result to the dataset for further use
    ds['Q_s'] = vertically_integrated_qs

    return ds

if __name__ == '__main__':
    
    experiments = ['pi', 'curr', 'incr']

    # Import forecast data 
    base_dir = '/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/MED-R/EXP/{}/EU025/pl/pf'
    exp = {}
    for experiment in experiments:
        exp[experiment] = xr.open_mfdataset(os.path.join(base_dir.format(experiment), '*.nc'), preprocess=bb.Data.preproc_ds)

    # Calculate the mass-weighted average vertical velocity
    starttime = '2023-10-19 00'
    endtime = '2023-10-22 00'
    exp = {exp_key: integate_qs(calc_average_w(exp[exp_key])) for exp_key in exp.keys()}
    Q_s_dict = {exp_key: {ini_key: (exp[exp_key]-exp['curr']).Q_s.sel(inidate=ini_key, time=slice(starttime, endtime)).mean(dim=['time', 'number']).squeeze() for ini_key in ['2023-10-15', '2023-10-17']} for exp_key in exp.keys()}
    w_dict = {exp_key: {ini_key: (exp[exp_key]-exp['curr']).mass_weighted_w.sel(inidate=ini_key, time=slice(starttime, endtime)).mean(dim=['time', 'number']).squeeze() for ini_key in ['2023-10-15', '2023-10-17']} for exp_key in exp.keys()}


    q_s = combine_xarray_dict({experiment: Q_s_dict[experiment]['2023-10-15'] for experiment in experiments}).compute()
    w = combine_xarray_dict({experiment: w_dict[experiment]['2023-10-15'] for experiment in experiments}).compute()

    # combine in one dataset
    ds = xr.merge([q_s, w])

    # Save q_s and w to netCDF
    save_dir = '/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/q_s_and_w/'
    fpath = os.path.join(save_dir, f'11_q_s_and_w_2023-10-15.nc')
    ds.to_netcdf(fpath)