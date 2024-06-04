'''
Functions to perform lagragian analysis
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob


class Lagrange():
    """
    Class to perform lagragian analysis.
    """

    def __init__(self):
        self.status = None
    
    def lagrangian_frame(ds):
        '''
        Function to calculate Lagrangian frame from tracks
        Written by Nick Leach.

        Input:
        ------
        ds: xarray dataset

        Output:
        -------
        '''
        ds = ds.squeeze()
        ds = ds.assign_coords(latitude=ds.latitude - ds.centroid_lat,
                              longitude=ds.longitude - ds.centroid_lon)
        ds = ds.rename(latitude='storm_lat', longitude='storm_lon')
        ds = ds.sel(storm_lon=slice(-10, 10), storm_lat=slice(10, -10))
        return ds

    def import_medr_tracks_TE(fpath):
        '''
        Funtion to import medium range tracks from
        Tempest Extremes algorithm
        Written by Nick Leach.

        Input:
        ------
        fpath: string, file path

        Output:
        -------
        '''

        df = pd.read_csv(fpath, skipinitialspace=True)

        expdict = {'b2rc': 'curr',
                   'b2re': 'incr',
                   'b2rr': 'pi'}

        fname = fpath.split('/')[-1]
        _, expid, inidate, mem = fname.split('_')

        df['expid'] = expid
        df['experiment'] = expdict[expid]
        df['inidate'] = pd.to_datetime(inidate)
        df['number'] = int(mem)
        return df
