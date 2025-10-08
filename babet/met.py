'''
Functions to calculate meteorological variables
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob
import metpy.calc as mpcalc
from metpy.units import units


class Met():
    """
    Class to calculate meteorological variables.
    """

    def __init__(self):
        self.status = None
    
    def calc_ivt(q, u, v, toplevel=250):
        """
        Function to calculate integrated vapor transport.
        Definition to calculate integrated vapor transport.
        Written by Shirin Ermis.

        Input:
        ------
        q: xarray dataset
            Specific humidity
        u: xarray dataset
            Zonal wind
        v: xarray dataset
            Meridional wind
        lat: xarray dataset
            Latitude
        toplevel: int
            Top level for integration

        Output:
        -------
        ivt: xarray dataset
            Integrated vapor transport
        """

        # Constants
        g = 9.81

        # Calculate IVT
        vt = ((q * v)**2 + (q * u)**2)**(1/2) # vapour transport on all pressure levels
        ivt = vt.sel(level=slice(toplevel, 1000)).integrate('level') / g
        return ivt
    
    def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
        """
        https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows
        slightly modified version: of http://stackoverflow.com/a/29546836/2901002

        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees or in radians)

        All (lat, lon) coordinates must have numeric dtypes and be of equal length.

        """
        if to_radians:
            lat1 = np.radians(lat1)
            lat2 = np.radians(lat2)
            lon1 = np.radians(lon1)
            lon2 = np.radians(lon2)

        a = np.sin((lat2-lat1)/2.0)**2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

        return earth_radius * 2 * np.arcsin(np.sqrt(a))
    
    def calc_vert_velocity_average(w, levels, temperature, Pa_to_mm=True,
                                   upper=200, lower=850): 
        
        # look out for unites of temperature and pressure!
        assert(temperature.mean().values > 200, 'Temperature needs to be in K')
        assert(levels.mean().values < 1000, 'Pressure levels need to be in hPa')

        # convert to mm
        if Pa_to_mm:
            vert_w = mpcalc.vertical_velocity(w*units('Pa/s'), levels*units('hPa'), temperature*units('K'))

        # mass weighted mean
        delta_p = vert_w.level.diff('level').rename('delta_p')
        delta_p = delta_p.assign_coords(p=vert_w['level'].isel(level=slice(1, None)))
        weighted_w  = vert_w.sel(level=slice(upper, lower)).isel(level=slice(1, None)) * delta_p
        av_w = (weighted_w.sum(dim='level') / delta_p.sum(dim='level')).rename('av_w') * 100

        av_w.assign_attrs(units='mm/s', description=f'Mass averaged vertical velocity between {upper}hPa and {lower}hPa')

        return av_w