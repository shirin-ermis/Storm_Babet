import xarray as xr
import os
import babet as bb
import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})

def calc_advection_q_all_fast(ds):
    """
    Calculate scalar horizontal advection of q on a pressure level.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing u, v, and q variables.

    Returns
    -------
    xarray.DataArray
        Scalar advection of q.
    """
    # Compute grid spacing using haversine distance
    dx = bb.met.Met.haversine(ds.latitude, ds.longitude.shift(longitude=-1), ds.latitude, ds.longitude.shift(longitude=1))
    dy = bb.met.Met.haversine(ds.latitude.shift(latitude=-1), ds.longitude, ds.latitude.shift(latitude=1), ds.longitude)
    
    # Calculate gradients using xarray's built-in differentiation
    dqdlon = ds.q.differentiate('longitude') / dx
    dqdlat = ds.q.differentiate('latitude') / dy

    # Compute advection
    adv = (ds.u * dqdlon) + (ds.v * dqdlat)

    return adv

def vert_average_q_advection_fast(ds, upper=250, lower=1000):
    """Calculate the vertically averaged moisture advection using mass-weighting.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing u, v, and q variables.
    upper : int
        Upper pressure level for the integration.
    lower : int
        Lower pressure level for the integration.

    Returns
    -------
    xarray.Dataset
        Dataset containing the mass-weighted moisture advection.
    """
    
    # Select the relevant pressure levels
    ds_levels = ds.sel(level=slice(upper, lower))
    
    # Compute moisture advection at all levels
    ds_levels['q_advection'] = calc_advection_q_all_fast(ds_levels)

    # Compute pressure thickness (ΔP)
    delta_p = ds_levels.level.diff('level')

    # Compute mass-weighted moisture advection
    weighted_vdq = ds_levels['q_advection'].isel(level=slice(1, None)) * delta_p

    # Compute the mass-weighted vertical average
    mass_weighted_average = (weighted_vdq.sum(dim='level') / delta_p.sum(dim='level')).rename('mass_weighted_vdq')

    # Assign the computed value to the original dataset without deep copying
    ds = ds.assign(mass_weighted_vdq=mass_weighted_average)

    return ds

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

if __name__ == '__main__':
    
    experiments = ['pi', 'curr', 'incr']
    
    # Pressure levels for the integration
    upper = 250
    lower = 1000
    
    # Import forecast data 
    print('Importing forecast data...')
    base_dir = '/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/MED-R/EXP/{}/EU025/pl/pf'
    exp = {}
    for experiment in experiments:
        exp[experiment] = xr.open_mfdataset(os.path.join(base_dir.format(experiment), '*.nc'), preprocess=bb.Data.preproc_ds)

    # Calculate the mass-weighted moisture advection
    print('Calculating moisture advection...')
    adv = {experiment: vert_average_q_advection_fast(exp[experiment], upper=upper, lower=lower) for experiment in ['pi', 'curr', 'incr']}
    adv_combined = combine_xarray_dict(adv)

    # Calculate mean over event
    print('Calculating event mean...')
    adv_combined_mean = adv_combined.sel(time=slice('2023-10-19 00', '2023-10-22 00'), inidate=['2023-10-15', '2023-10-17']).mean(dim='time').mass_weighted_vdq.compute()

    # Save the dataset to a netCDF file
    save_dir = '/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/advection/'
    fpath = os.path.join(save_dir, f'09_vert_integrated_adv_{lower}hPa_to_{upper}hPa_event-mean.nc')
    print(f'Saving the dataset to {fpath}...')
    adv_combined_mean.to_netcdf(os.path.join(save_dir, f'09_vert_integrated_adv_{lower}hPa_to_{upper}hPa_event-mean.nc'))

