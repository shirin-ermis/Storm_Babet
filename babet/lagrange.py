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
    
    def preproc_to_stormframe(ds, ifs_storm_list=None, sfc=True):
        '''
        Funtion for pre-processing to Lagrangian fields for tracked storms.
        Written by Nick Leach and Shirin Ermis.

        Input:
        ------
        ds: xarray dataset
        ifs_storm_list: pandas dataframe of track information
        sfc: bool, whether surface data or pressure level data is needed
        level:

        Output:
        -------
        LG_fields: xarray dataset with Lagrangian fileds for storms similar to the storm of interest
        '''

        ds = ds.copy()

        if 'number' not in ds.coords:
            ds = ds.expand_dims({'number': [0]})

        fpath = ds.encoding['source']
        if sfc:
            exp = fpath.split('/')[-5]
        else:
            exp = fpath.split('/')[-6]
        inidate = fpath.split('/')[-1].split('_')[-1].split('.')[0]
        ds_tracks = ifs_storm_list.query('experiment=="{}" & inidate=="{}"'.format(exp, inidate))
        LG_fields = []

        for num in set(ds.number.values).intersection(ds_tracks.number.unique()):

            mem_track = ds_tracks.loc[ds_tracks.number == num]
            mem_fields = ds.sel(number=num)
            time_intersection = sorted(list(set(mem_fields.time.values).intersection(mem_track.date.values)))

            resample_freq = 3  # resampling frequency in hours
            if inidate == '2022-02-10':  # this was used for Eunice preproc
                resample_freq = 6

            # get start / end times for properly calculating the maximum
            # fields (taking into account the different preproc times in IFS)
            time_start = time_intersection[0] - pd.Timedelta('{}h 59m'.format(resample_freq - 1))
            time_end = time_intersection[-1]

            # get the instantaneous fields + wind speeds
            if sfc:
                mem_fields_out = mem_fields.get(['u10',
                                                 'v10',
                                                 'msl',
                                                 'u100',
                                                 'v100',
                                                 'fg10',
                                                 't2m',
                                                 'tp',
                                                 'tcwv']).sel(time=time_intersection)
                mem_fields_out['ws10'] = np.sqrt(mem_fields_out.u10**2 + mem_fields_out.v10**2)
                mem_fields_out['ws100'] = np.sqrt(mem_fields_out.u100**2 + mem_fields_out.v100**2)

                # get the maximum fields, taking into account the different preproc times
                mxtpr_field_out = mem_fields.mxtpr.sel(time=slice(time_start, time_end)).resample(time='{}h'.format(resample_freq), label='right', closed='right', base=0).max()
                mem_fields_out['mxtpr'] = mxtpr_field_out
            else:
                mem_fields_out = mem_fields.get(['z',
                                                 'q',  # spec humidity
                                                 'w',
                                                 't',
                                                 'u',
                                                 'v',
                                                 'vo']).sel(time=time_intersection)
                mem_fields_out['ws'] = np.sqrt(mem_fields_out.u**2 + mem_fields_out.v**2)

            # add in the mslp centroid lon/lats for Lagrangian analysis
            mem_track_out = mem_track.loc[mem_track.date.isin(time_intersection)]
            mem_fields_out['centroid_lon'] = ('time', (mem_track_out.lon * 4).round() / 4)
            mem_fields_out['centroid_lat'] = ('time', (mem_track_out.lat * 4).round() / 4)

            # convert to storm frame fields
            mem_fields_out = mem_fields_out.groupby('time').apply(Lagrange.lagrangian_frame)
            mem_fields_out = mem_fields_out.assign(datetime=mem_fields_out.time).drop('time').rename(time='timestep')

            # compute the time of peak vorticity (include moving average to
            # smooth) for storm composites
            peak_vo = mem_track.rolling(3, center=True).mean().vo.idxmax()
            peak_vo_datetime = mem_track.date.loc[peak_vo]
            peak_vo_relative_time = (mem_fields_out.datetime.squeeze().to_pandas() - peak_vo_datetime).dt.total_seconds().values / (3600 * 24)

            # set the storm frame fields timestep relative to peak vorticity time
            mem_fields_out = mem_fields_out.assign_coords(timestep=peak_vo_relative_time)

            LG_fields += [mem_fields_out]

        LG_fields = xr.concat(LG_fields, 'number')
        LG_fields = LG_fields.expand_dims(dict(
            inidate=[pd.to_datetime(inidate)],
            experiment=[exp]))

        return LG_fields

