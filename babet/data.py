'''
Functions to import data and metadata
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob
import xesmf as xe


class Data():
    """
    Class to import data files and pre-process them.
    """

    def __init__(self):
        self.status = None

    def preproc_ds(ds):
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
        ds1 = ds.copy().squeeze()
        # set up aux data
        inidate = pd.to_datetime(ds1.time[0].values)
        # expand dimensions to include extra info
        if not 'hDate' in ds1:
            ds1 = ds1.expand_dims({'inidate': [inidate]}).copy()

        if not 'number' in ds1:
            ds1 = ds1.expand_dims({'number': [0]}).copy()

        # put time dimension at front
        ds1 = ds1.transpose('time', ...)
        return ds1
    
    def accum2rate(ds):
        """
        Function to convert accumulated variables to conventional ones.
        Definition to convert accumulated variables to instantaneous.
        Written by Nick Leach.

        Input:
        ------

        Output:
        -------
        """
        # accumulated variables & scaling factors
        accumulated_vars = {'tp': 60 * 60 * 24 * 1e3,
                            'ttr': 1,
                            'tsr': 1,
                            'str': 1,
                            'ssr': 1,
                            'e': 1}
        accumulated_var_newunits = {'tp': 'mm day$^{-1}$',
                                    'ttr': 'W m$^{-2}$',
                                    'tsr': 'W m$^{-2}$',
                                    'str': 'W m$^{-2}$',
                                    'ssr': 'W m$^{-2}$',
                                    'e':'m s$^{-1}$'}

        ds = ds.copy()
        oindex = ds.time
        inidate = pd.to_datetime(oindex[0].values)
        ds = ds.diff('time') / (ds.time.diff('time').astype(float) / 1e9 )
        ds = ds.reindex(time=oindex)
        return ds[1:]
    
    def preproc_mclim(ds):
        """
        A couple more steps for pre-processing m-climate
        Written by Nick Leach.

        Input:
        ------
        ds: xarray

        Output:
        -------
        ds: xarray
        """

        ds = ds.copy().squeeze()
        ds = Data.preproc_ds(ds)
        # create index of hours from initialisation
        ds_hours = ((ds.time - ds.time.isel(time=0)) / 1e9 / 3600).astype(int)
        # change time coord to hours coord + rename
        ds = ds.assign_coords(time=ds_hours).rename(dict(time='hour'))
        return ds
    
    def hourly2accum(ds, start_day='2023-10-18 09', end_day='2023-10-22 00', m2mm=True):
        """
        Function to convert hourly precipitation to accumulated precipitation in mm.
        Also truncates the data to the desired time period.

        Input:
        ------
        ds: xarray dataset

        Output:
        -------
        ds_out: xarray dataset with precipitation accumulated in mm.
        """
        if m2mm:
            factor = 1000
        else:
            factor = 1
        ds_out = ds.copy(deep=True).sel(time=slice(start_day, end_day))
        ds_out['tp'] = ds_out.tp.cumsum(dim='time')*factor  # sum and convert to mm

        return ds_out
    
    def get_era5_analogues():
        """
        Function that imports ERA5 analogue data in a cleaned version.
        """


        # check if file exists
        if not os.path.exists('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues_72hour_mean.nc'):
            # precip
            tmp1 = xr.open_mfdataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues72_tp_past2.nc').expand_dims(climate=["1950"])
            tmp2 = xr.open_mfdataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues72_tp_prst2.nc').expand_dims(climate=["present"])

            tmp = xr.concat([tmp1, tmp2], dim="climate")

            # Find all variables that start with "unknown"
            precip_vars = sorted([var for var in tmp.data_vars if var.startswith("unknown")])

            # Stack all precipitation variables along the new 'member' dimension
            tp = xr.concat([tmp[var] for var in precip_vars], dim="member")

            # Assign member values from 1 to 27
            tp = tp.assign_coords(member=np.arange(1, len(precip_vars) + 1))

            # Create a new dataset with the combined variable
            era5_analogues = xr.Dataset({"tp": tp}, coords={"lat": tmp.lat, "lon": tmp.lon, "member": tp.member})


            # mean sea level pressure
            tmp1 = xr.open_mfdataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues72_slp_past2.nc').expand_dims(climate=["1950"])
            tmp2 = xr.open_mfdataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues72_slp_prst2.nc').expand_dims(climate=["present"])

            tmp = xr.concat([tmp1, tmp2], dim="climate")

            # Find all variables that start with "unknown"
            slp_vars = sorted([var for var in tmp.data_vars if var.startswith("msl")])

            # Stack all precipitation variables along the new 'member' dimension
            msl = xr.concat([tmp[var] for var in slp_vars], dim="member")

            # Assign member values from 1 to 27
            msl = msl.assign_coords(member=np.arange(1, len(slp_vars) + 1))

            # Create a new dataset with the combined variable
            era5_analogues = xr.merge([era5_analogues,
                                    xr.Dataset({"msl": msl}, coords={"lat": tmp.lat, "lon": tmp.lon, "member": msl.member})], compat="override")

            # 2m temperature
            tmp1 = xr.open_mfdataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues72_t2m_past2.nc').expand_dims(climate=["1950"])
            tmp2 = xr.open_mfdataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues72_t2m_prst2.nc').expand_dims(climate=["present"])

            tmp = xr.concat([tmp1, tmp2], dim="climate")

            # Find all variables that start with "unknown"
            t2m_vars = sorted([var for var in tmp.data_vars if var.startswith("unknown")])

            # Stack all precipitation variables along the new 'member' dimension
            t2m = xr.concat([tmp[var] for var in t2m_vars], dim="member")

            # Assign member values from 1 to 27
            t2m = t2m.assign_coords(member=np.arange(1, len(t2m_vars) + 1))

            # Create a new dataset with the combined variable
            era5_analogues = xr.merge([era5_analogues,
                                    xr.Dataset({"t2m": t2m}, coords={"lat": tmp.lat, "lon": tmp.lon, "member": t2m.member})], compat="override")

            # Save to netcdf
            era5_analogues.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues_72hour_mean.nc')
        else:
            print('Importing data from pre-existing file')
            era5_analogues = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/ERA5_analogues/analogues_72hour_mean.nc')
        
        return era5_analogues
    
    def get_racmo_analogues():
        """
        Function to import RACMO analogue data in a cleaned version.
        """

        # RACMO analogues

        # check if file exists
        if not os.path.exists('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_msl_72hour_mean.nc'):

            # mslp
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_RACMO_2023-10-20_use27_stat_AmeanprSCOT_NC_mslp_1951-1980.nc').expand_dims(climate=["1950"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_RACMO_2023-10-20_use27_stat_AmeanprSCOT_NC_mslp_1991-2020.nc').expand_dims(climate=["present"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_RACMO_2023-10-20_use27_stat_AmeanprSCOT_NC_mslp_2071-2100.nc').expand_dims(climate=["future2"])
            racmo_msl = xr.concat([tmp1, tmp2, tmp3], dim="climate").rename({"mslp": "msl"})

            # precip
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_RACMO_2023-10-20_use27_stat_AmeanprSCOT_NC_precip_1951-1980.nc').expand_dims(climate=["1950"])
            tmp1_ = xr.Dataset({"tp": (("climate", "lat", "lon"), tmp1.precip.values)},
                        coords={"lat": tmp1.lat.values[:,0], 
                                "lon": tmp1.lon.values[0,:],
                                "climate": tmp1.climate.values})
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_RACMO_2023-10-20_use27_stat_AmeanprSCOT_NC_precip_1991-2020.nc').expand_dims(climate=["present"])
            tmp2_ = xr.Dataset({"tp": (("climate", "lat", "lon"), tmp2.precip.values)},
                        coords={"lat": tmp2.lat.values[:,0], 
                                "lon": tmp2.lon.values[0,:],
                                "climate": tmp2.climate.values})
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_RACMO_2023-10-20_use27_stat_AmeanprSCOT_NC_precip_2071-2100.nc').expand_dims(climate=["future2"])
            tmp3_= xr.Dataset({"tp": (("climate", "lat", "lon"), tmp3.precip.values)},
                        coords={"lat": tmp3.lat.values[:,0], 
                                "lon": tmp3.lon.values[0,:],
                                "climate": tmp3.climate.values})
            racmo_tp = xr.concat([tmp1_, tmp2_, tmp3_], dim="climate")
            
            # t2m
            # missing so far

            # Save file in two parts because lat lon are not compatible
            racmo_msl.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_msl_72hour_mean.nc')
            racmo_tp.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_tp_72hour_mean.nc')
        else:
            print('Importing data from pre-existing file')
            racmo_msl = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_msl_72hour_mean.nc')
            racmo_tp = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_tp_72hour_mean.nc')

        return racmo_msl, racmo_tp
    
    def get_pgw():
        """
        Function to import PGW data in a cleaned version.
        """

        # check if file exists
        if not os.path.exists('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_clean.nc'):
            # mean sea level pressure
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_slp_past.nc').expand_dims(climate=["1870"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_slp_prst.nc').expand_dims(climate=["present"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_slp_fut1.nc').expand_dims(climate=["future1"])
            tmp4 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_slp_fut2.nc').expand_dims(climate=["future2"])
            tmp = xr.concat([tmp1, tmp2, tmp3, tmp4], dim="climate").rename({"unknown": "msl"})

            ds = xr.Dataset(
                data_vars=dict(
                    msl=(["climate", "time", "lat", "lon"], tmp.msl.values)),
                coords=dict(
                    lon=tmp.lon.values[0,:],
                    lat=tmp.lat.values[:,0],
                    time=tmp.time.values,
                    climate=tmp.climate.values),
                attrs=dict(description="PGW data"))

            # precip
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_tp_past.nc').expand_dims(climate=["1870"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_tp_prst.nc').expand_dims(climate=["present"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_tp_fut1.nc').expand_dims(climate=["future1"])
            tmp4 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_tp_fut2.nc').expand_dims(climate=["future2"])
            tmp = xr.concat([tmp1, tmp2, tmp3, tmp4], dim="climate").rename({"unknown": "tp"})

            ds = xr.merge([ds,
                            xr.Dataset(data_vars=dict(tp=(["climate", "time", "lat", "lon"], tmp.tp.values)),
                                    coords=dict(lon=tmp.lon.values[0,:],
                                            lat=tmp.lat.values[:,0],
                                            time=tmp.time.values,
                                            climate=tmp.climate.values),
                                    attrs=dict(description="PGW data"))], compat="override")

            # temperature
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_t2m_past.nc').expand_dims(climate=["1870"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_t2m_prst.nc').expand_dims(climate=["present"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_t2m_fut1.nc').expand_dims(climate=["future1"])
            tmp4 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_t2m_fut2.nc').expand_dims(climate=["future2"])
            tmp = xr.concat([tmp1, tmp2, tmp3, tmp4], dim="climate").rename({"unknown": "t2m"})
            pgw = xr.merge([ds,
                            xr.Dataset(data_vars=dict(t2m=(["climate", "time", "lat", "lon"], tmp.t2m.values)),
                                    coords=dict(lon=tmp.lon.values[0,:],
                                            lat=tmp.lat.values[:,0],
                                            time=tmp.time.values,
                                            climate=tmp.climate.values),
                                    attrs=dict(description="PGW data"))], 
                            compat="override")

            # Save to netcdf
            pgw.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_clean.nc')
        else:
            print('Importing data from pre-existing file')
            pgw = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW/pgw_clean.nc')

        return pgw
    
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
    
    def get_fba_ifs():
        """
        Function to import FBA IFS data more easily, same as other methods with dimension "climate" nit .
        """

        tmp = []
        base_dir='/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/MED-R/EXP/{}/EU025/sfc/{}'
        climates = ['1870', '1950', 'present', 'future1']
        for e, exp in enumerate(['pi', 'pi_1950', 'curr', 'incr']):
            tmp.append([])
            for c in ['cf', 'pf']:
                dir_path = os.path.join(base_dir.format(exp, c), '*.nc')
                ds = xr.open_mfdataset(dir_path, engine='netcdf4', preprocess=Data.preproc_ds_v2).get(['tp', 't2m', 'msl', 'tcwv'])
                tmp[e].append(ds.expand_dims(climate=[climates[e]]))
            tmp[e] = xr.concat(tmp[e], dim='number')
        ifs = xr.concat(tmp, dim='climate')

        return ifs
    
    def get_global_ifs():
        """
        Function to import FBA IFS data more easily, same as other methods with dimension "climate" nit .
        """

        tmp = []
        base_dir='/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/MED-R/EXP/{}/GLO100/sfc/{}'
        climates = ['1870', '1950', 'present', 'future1']
        for e, exp in enumerate(['pi', 'pi_1950', 'curr', 'incr']):
            tmp.append([])
            for c in ['cf', 'pf']:
                dir_path = os.path.join(base_dir.format(exp, c), '*.nc')
                print(dir_path)
                ds = xr.open_mfdataset(dir_path, engine='netcdf4', preprocess=Data.preproc_ds_v2).get(['t2m'])
                tmp[e].append(ds.expand_dims(climate=[climates[e]]))
            tmp[e] = xr.concat(tmp[e], dim='number')
        ifs = xr.concat(tmp, dim='climate')

        return ifs
    
    def get_fba_micas():
        # FBA ACCESS

        # check if file exists
        if not os.path.exists('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_clean.nc'):
            # precip
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_pr_highGHG.nc').expand_dims(climate=["future1"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_pr_lowGHG.nc').expand_dims(climate=["1870"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_pr_ctrl.nc').expand_dims(climate=["present"])
            ds = xr.concat([tmp1, tmp2, tmp3], dim="climate").rename({"pr": "tp"})

            # temperature
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_tas_highGHG.nc').expand_dims(climate=["future1"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_tas_lowGHG.nc').expand_dims(climate=["1870"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_tas_ctrl.nc').expand_dims(climate=["present"])
            ds = xr.merge([ds, xr.concat([tmp1, tmp2, tmp3], dim="climate")], compat="override")

            # mean sea level pressure
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_psl_highGHG.nc').expand_dims(climate=["future1"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_psl_lowGHG.nc').expand_dims(climate=["1870"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_psl_ctrl.nc').expand_dims(climate=["present"])
            micas = xr.merge([ds, xr.concat([tmp1, tmp2, tmp3], dim="climate").rename({'psl': 'msl'})], compat="override")

            # Save to netcdf
            micas.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_clean.nc')
        else:
            print('Importing data from pre-existing file')
            micas = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/access-micas/micas_clean.nc')

        return micas


    def regrid_racmo(ds):
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], ds.lat[:,0].values),
                "lon": (["lon"], ds.lon[0,:].values),
            }
        )

        regridder = xe.Regridder(ds, ds_out, "bilinear")

        regridded_data = regridder(ds)
        return regridded_data
    
    def clean_array_racmo(tmp1, var_name, var=None):

        # If var is not provided, use var_name
        if var is None:
            var = var_name
        
        # Find all variables that start with "unknown"
        slp_vars = sorted([data_var for data_var in tmp1.data_vars if data_var.startswith(var)])

        # Rename 'mem' dimension to 'member' if it exists
        if 'mem' in tmp1.dims:
            tmp1 = tmp1.rename({'mem': 'member'})

        # Stack all precipitation variables along the new 'member' dimension, if already done, do nothing
        if len(slp_vars) == 1:
            msl = tmp1[slp_vars[0]]
        else:
            msl = xr.concat([tmp1[var] for var in slp_vars], dim="member")
            msl = msl.assign_coords(member=np.arange(1, len(slp_vars) + 1))
        
        # remove height dimension if it exists
        if 'height' in msl.dims:
            msl = msl.squeeze('height', drop=True)

        # Create a new dataset with the combined variable
        if 'time' in tmp1.coords:
            test = xr.Dataset({var_name: msl}, coords={"lat": tmp1.lat, "lon": tmp1.lon, "member": msl.member, "time": tmp1.time})
        else:
            test = xr.Dataset({var_name: msl}, coords={"rlat": tmp1.rlat, "rlon": tmp1.rlon, "member": msl.member})

        tmp1_ = Data.regrid_racmo(test)
        return tmp1_
    
    def get_pgw_ensemble():
            # check if file exists
            if not os.path.exists('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_clean_ensemble.nc'):
                # mean sea level pressure
                tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_slp_PGW-1.nc').expand_dims(climate=["1870"])
                tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_slp_CTL.nc').expand_dims(climate=["present"])
                tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_slp_PGW+1.nc').expand_dims(climate=["future1"])
                
                tmp4dry = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_slp_PGWdry+3.nc').expand_dims(dummy=["dry"])
                tmp4wet = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_slp_PGWwet+3.nc').expand_dims(dummy=["wet"])
                tmp4 = xr.concat([tmp4dry, tmp4wet], dim="dummy").mean("dummy").squeeze().expand_dims(climate=["future2"])

                tmp1_ = Data.clean_array_racmo(tmp1, var_name="msl")
                tmp2_ = Data.clean_array_racmo(tmp2 , var_name="msl")
                tmp3_ = Data.clean_array_racmo(tmp3, var_name="msl")
                tmp4_ = Data.clean_array_racmo(tmp4, var_name="msl")
                pgw_ens = xr.concat([tmp1_, tmp2_, tmp3_, tmp4_], dim="climate")

                # precip
                tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_pr_PGW-1.nc').expand_dims(climate=["1870"])
                tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_pr_CTL.nc').expand_dims(climate=["present"])
                tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_pr_PGW+1.nc').expand_dims(climate=["future1"])
                
                tmp4dry = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_pr_PGWdry+3.nc').expand_dims(dummy=["dry"])
                tmp4wet = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_pr_PGWwet+3.nc').expand_dims(dummy=["wet"])
                tmp4 = xr.concat([tmp4dry, tmp4wet], dim="dummy").mean("dummy").squeeze().expand_dims(climate=["future2"])

                tmp1_ = Data.clean_array_racmo(tmp1, var_name="unknown")
                tmp2_ = Data.clean_array_racmo(tmp2, var_name="unknown")
                tmp3_ = Data.clean_array_racmo(tmp3, var_name="unknown")
                tmp4_ = Data.clean_array_racmo(tmp4, var_name="unknown")
                pgw_ens = xr.merge([pgw_ens, 
                                    xr.concat([tmp1_, tmp2_, tmp3_, tmp4_], dim="climate").rename({"unknown": "tp"})], 
                                    compat="override")
                
                # temperature
                tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_t2m_PGW-1.nc').expand_dims(climate=["1870"])
                tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_t2m_CTL.nc').expand_dims(climate=["present"])
                tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_t2m_PGW+1.nc').expand_dims(climate=["future1"])
                
                tmp4dry = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_t2m_PGWdry+3.nc').expand_dims(dummy=["dry"])
                tmp4wet = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_t2m_PGWwet+3.nc').expand_dims(dummy=["wet"])
                tmp4 = xr.concat([tmp4dry, tmp4wet], dim="dummy").mean("dummy").squeeze().expand_dims(climate=["future2"])
                
                tmp1_ = Data.clean_array_racmo(tmp1, var_name="t2m")
                tmp2_ = Data.clean_array_racmo(tmp2, var_name="t2m")
                tmp3_ = Data.clean_array_racmo(tmp3, var_name="t2m")
                tmp4_ = Data.clean_array_racmo(tmp4, var_name="t2m")
                pgw_ens = xr.merge([pgw_ens, 
                                    xr.concat([tmp1_, tmp2_, tmp3_, tmp4_], dim="climate")], 
                                    compat="override")

                # Save to netcdf
                pgw_ens.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_clean_ensemble.nc')
            else:
                print('Importing data from pre-existing file')
                pgw_ens = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_ensemble/pgw_clean_ensemble.nc')

            return pgw_ens
    
    def get_racmo_indiv_analogues():
        # check if file exists
        if not os.path.exists('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_msl_72hour_analogues.nc'):
            # MSLP
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_ECEARTH__mslp__1951-1980_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["1950"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_ECEARTH__mslp__1991-2020_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["present"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_ECEARTH__mslp__2071-2100_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["future1"])
            tmp4 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_ECEARTH__mslp__2071-2100_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["future2"])


            racmo_msl = xr.concat([tmp1, tmp2, tmp3, tmp4], dim='climate').rename_dims({'ana': 'member'}).rename({"mslp": "msl"})

            # precipitation
            tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_RACMO__precip__1951-1980_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["1950"])
            tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_RACMO__precip__1991-2020_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["present"])
            tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_RACMO__precip__2071-2100_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["future1"])
            tmp4 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogs_Babet_A20231020_KNMI_RACMO__precip__2071-2100_3dy_ave.nc').swap_dims({'time': 'ana'}).expand_dims(climate=["future2"])

            tmp1_ = Data.regrid_racmo(tmp1).rename_dims({'ana': 'member'}).rename({"precip": "tp"})
            tmp2_ = Data.regrid_racmo(tmp2).rename_dims({'ana': 'member'}).rename({"precip": "tp"})
            tmp3_ = Data.regrid_racmo(tmp3).rename_dims({'ana': 'member'}).rename({"precip": "tp"})
            tmp4_ = Data.regrid_racmo(tmp4).rename_dims({'ana': 'member'}).rename({"precip": "tp"})
            racmo_tp = xr.concat([tmp1_, tmp2_, tmp3_, tmp4_], dim="climate")

            # Save file in two parts because lat lon are not compatible
            racmo_msl.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_msl_72hour_analogues.nc')
            racmo_tp.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_tp_72hour_analogues.nc')
        else:
            print('Importing data from pre-existing file')
            racmo_msl = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_msl_72hour_analogues.nc')
            racmo_tp = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/RACMO_analogues/analogues_tp_72hour_analogues.nc')
        
        return racmo_msl, racmo_tp
    
    def get_pgw_new_runs():
        
            # check if file exists
            if not os.path.exists('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/pgw_clean_ensemble_new_runs.nc'):
                # mean sea level pressure
                tmp1 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/single_level_jexp4_expid_-1.5K.nc').expand_dims(climate=["1870"])
                tmp2 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/single_level_jexp0_expid_CTL.nc').expand_dims(climate=["present"])
                tmp3 = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/single_level_jexp1_expid_+1.5K.nc').expand_dims(climate=["future1"])
                
                tmp4dry = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/single_level_jexp2_expid_+3.0Kd.nc').expand_dims(dummy=["dry"])
                tmp4wet = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/single_level_jexp3_expid_+3.0Kw.nc').expand_dims(dummy=["wet"])
                tmp4 = xr.concat([tmp4dry, tmp4wet], dim="dummy").mean("dummy").squeeze().expand_dims(climate=["future2"])

                tmp1_ = Data.clean_array_racmo(tmp1, var_name="msl")
                tmp2_ = Data.clean_array_racmo(tmp2 , var_name="msl")
                tmp3_ = Data.clean_array_racmo(tmp3, var_name="msl")
                tmp4_ = Data.clean_array_racmo(tmp4, var_name="msl")
                pgw_ens = xr.concat([tmp1_, tmp2_, tmp3_, tmp4_], dim="climate", coords="minimal", compat="override")

                # precip
                tmp1_ = Data.clean_array_racmo(tmp1, var_name="tp", var="precip")
                tmp2_ = Data.clean_array_racmo(tmp2, var_name="tp", var="precip")
                tmp3_ = Data.clean_array_racmo(tmp3, var_name="tp", var="precip")
                tmp4_ = Data.clean_array_racmo(tmp4, var_name="tp", var="precip")
                pgw_ens = xr.merge([pgw_ens, 
                                    xr.concat([tmp1_, tmp2_, tmp3_, tmp4_], dim="climate", coords="minimal", compat="override")],
                                    compat="override")
                
                # qvi
                tmp1_ = Data.clean_array_racmo(tmp1, var_name="qvi")
                tmp2_ = Data.clean_array_racmo(tmp2, var_name="qvi")
                tmp3_ = Data.clean_array_racmo(tmp3, var_name="qvi")
                tmp4_ = Data.clean_array_racmo(tmp4, var_name="qvi")
                pgw_ens = xr.merge([pgw_ens, 
                                    xr.concat([tmp1_, tmp2_, tmp3_, tmp4_], dim="climate", coords="minimal", compat="override")], 
                                    compat="override")

                # Save to netcdf
                pgw_ens.to_netcdf('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/pgw_clean_ensemble_new_runs.nc')
            else:
                print('Importing data from pre-existing file')
                pgw_ens = xr.open_dataset('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/PGW_new_runs/pgw_clean_ensemble_new_runs.nc')
            
            return pgw_ens