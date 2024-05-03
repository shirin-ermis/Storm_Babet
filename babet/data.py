'''
Functions to import data and metadata
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob


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
