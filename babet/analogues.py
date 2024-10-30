'''
Functions to create flow analogues from ERA5
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob


class Analogues():
    """
    Class to import data files and pre-process them.
    """

    def __init__(self):
        self.status = None
