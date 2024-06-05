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

    def babet_dist(df):
        '''
        Function to calculate distance between Babet and tracks
        Written by Nick Leach and Shirin Ermis.

        Input:
        ------
        df: pandas dataframe, dataframe of tracks in IFS

        Output:
        ----
        distances of tracks in IFS to Babet track in ERA5
        '''
        
        # import tracks from ERA 5
        era_track = pd.read_csv('/gf5/predict/AWH019_ERMIS_ATMICP/Babet/DATA/postproc/tracks/TEStitch_2023_0',skipinitialspace=True)
        era_track['expid'] = 'era5'
        era_track['experiment'] = 'era5'
        era_track['inidate'] = pd.to_datetime('2023-10-01')
        era_track['number'] = 0
        era_track['date'] = pd.to_datetime(era_track.loc[:,['year','month','day','hour']])

        # select Babet track from ERA5, get properties
        babet_track = era_track.query('track_id==1')
        babet_lons = babet_track.lon.values
        babet_lats = babet_track.lat.values
        
        track_lons = df.lon.values
        track_lats = df.lat.values
        
        # calculate distance between Babet and tracks in IFS
        minsize = min(babet_lons.size,track_lons.size)
        
        return np.sqrt((track_lons[:minsize]-babet_lons[:minsize])**2+(track_lats[:minsize]-babet_lats[:minsize])**2).sum()
