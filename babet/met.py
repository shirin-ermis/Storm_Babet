'''
Functions to calculate meteorological variables
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob


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