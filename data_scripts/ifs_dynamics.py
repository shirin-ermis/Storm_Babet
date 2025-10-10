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

if __name__ == '__main__':

    uk = [-11, 10, 40, 70]
    tmp = []
    base_dir='/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/MED-R/EXP/{}/EU025/pl/{}'
    climates = ['1870', '1950', 'present', 'future1']
    for e, exp in enumerate(['pi', 'pi_1950', 'curr', 'incr']):
        tmp.append([])
        for c in ['pf']:
            dir_path = os.path.join(base_dir.format(exp, c), '*.nc')
            ds = xr.open_mfdataset(dir_path, engine='netcdf4', preprocess=bb.data.Data.preproc_ds_v2).get(['t', 'q', 'w'])
            tmp[e].append(ds.expand_dims(climate=[climates[e]]))
        tmp[e] = xr.concat(tmp[e], dim='number')
    ifs_pl = xr.concat(tmp, dim='climate')

    # calculate averaged vertical velocity
    ifs_pl['w'] = ifs_pl.w.sel(time=slice('2023-10-19 00', '2023-10-22 00'), latitude=slice(uk[3], uk[2]), longitude=slice(uk[0], uk[1])).mean(dim='time')
    ifs_pl['q'] = ifs_pl.q.sel(time=slice('2023-10-19 00', '2023-10-22 00'), latitude=slice(uk[3], uk[2]), longitude=slice(uk[0], uk[1])).mean(dim='time')
    ifs_pl['t'] = ifs_pl.t.sel(time=slice('2023-10-19 00', '2023-10-22 00'), latitude=slice(uk[3], uk[2]), longitude=slice(uk[0], uk[1])).mean(dim='time')
    ifs_pl['av_w'] = bb.met.Met.calc_vert_velocity_average(ifs_pl['w'], ifs_pl.level, ifs_pl['t'])

    ifs_pl = ifs_pl.chunk({})
    ifs_pl.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/av_w/ifs_dynamics.nc', mode='w')