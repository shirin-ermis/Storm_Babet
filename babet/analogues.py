'''
Functions to create flow analogues from ERA5
'''

import xarray as xr
import os
import numpy as np
import pandas as pd
import glob
import subprocess
import iris
import iris.coord_categorisation as icc
from iris.coord_categorisation import add_season_membership
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cart
import matplotlib.cm as mpl_cm
import sys
import scipy.stats as sps
from scipy.stats import genextreme as gev
import random
import scipy.io
import netCDF4 as nc
import iris.coords
import iris.util
from iris.util import equalise_attributes
from iris.util import unify_time_units
from scipy.stats.stats import pearsonr
import scipy.stats as stats
import calendar


class Analogues():
    """
    Class to import data files and pre-process them.
    """

    def __init__(self):
        self.status = None

    def era5_data(var, years=np.arange(1950,2023)):
        '''
        input: variable as string, years as list,
        var options include: t2m, msl, z500, tmax ...
        defaults: years 1950-2022.
        returns list of cubes (one per year) of daily variable, May-Sep
        '''
        all_yr_list = iris.cube.CubeList()
        for yr in years:
            print(yr)
            files = glob.glob('/net/pc230042/nobackup/users/sager/nobackup_2_old/ERA5/'+str(yr)+'/day/era5_????0[56789]_'+var+'.nc') # numbers indicate months extracted
            yr_cubes = iris.load(files)
            rm_att = equalise_attributes(yr_cubes)
            unify_time_units(yr_cubes)
            new_cubes = iris.cube.CubeList()
            for each in yr_cubes:
                if var == 'z500':
                    each.coord('air_pressure').var_name = 'lev'
                new_cubes.append(iris.util.squeeze(each))
                if len(new_cubes)>1 and new_cubes[-1].metadata!=new_cubes[0].metadata:
                    new_cubes[-1].metadata = new_cubes[0].metadata
            yr_cube = new_cubes.concatenate_cube()
            if len(all_yr_list)>1 and all_yr_list[-1].metadata!=all_yr_list[0].metadata:
                all_yr_list[-1].metadata = all_yr_list[0].metadata
            all_yr_list.append(yr_cube)
        return all_yr_list


    def era5_mydata(var, years=np.arange(1950,2023)):
        '''
        input: variable as string, years as list,
        var options include: t2m, msl, z500, tmax ...
        defaults: years 1950-2022.
        returns list of cubes (one per year) of daily variable, May-Sep
        '''
        if var == 'psi250':
            file = '/net/pc200023/nobackup/users/thompson/ERA5/'+var+'/ERA5_'+var+'_day_1deg_jja.nc' # numbers indicate months extracted
            cube = iris.load(file)[0]
            iris.coord_categorisation.add_year(cube, 'time')
            MyConstraint = iris.Constraint(year=lambda y: years[0] <= y <= years[-1])
            all_yr_list = cube.extract(MyConstraint)
        elif var == 'z500':
            file = '/net/pc200023/nobackup/users/thompson/ERA5/ERA5_'+var+'.nc' # numbers indicate months extracted
            cube = iris.load(file)[0]
            iris.coord_categorisation.add_year(cube, 'time')
            MyConstraint = iris.Constraint(year=lambda y: years[0] <= y <= years[-1])
            all_yr_list = cube.extract(MyConstraint)
            all_yr_list.coord('air_pressure').var_name = 'lev'
            all_yr_list = iris.util.squeeze(all_yr_list)
        elif var == 'msl':
            file = '/net/pc200023/nobackup/users/thompson/ERA5/ERA5_'+var+'.nc' # numbers indicate months extracted
            cube = iris.load(file)[0]
            iris.coord_categorisation.add_year(cube, 'time')
            MyConstraint = iris.Constraint(year=lambda y: years[0] <= y <= years[-1])
            all_yr_list = cube.extract(MyConstraint)
        else:
            all_yr_list = iris.cube.CubeList()
            for yr in years:
                print(yr)
                files = glob.glob('/net/pc200023/nobackup/users/thompson/ERA5/'+var+'/ERA5_'+var+'_day_'+str(yr)+'*.nc') # numbers indicate months extracted
                yr_cubes = iris.load(files)
                rm_att = equalise_attributes(yr_cubes)
                unify_time_units(yr_cubes)
                new_cubes = iris.cube.CubeList()
                for each in yr_cubes:
                    if var == 'z500':
                        each.coord('air_pressure').var_name = 'lev'
                    new_cubes.append(iris.util.squeeze(each))
                    if len(new_cubes)>1 and new_cubes[-1].metadata!=new_cubes[0].metadata:
                        new_cubes[-1].metadata = new_cubes[0].metadata
                yr_cube = new_cubes.concatenate_cube()
                if len(all_yr_list)>1 and all_yr_list[-1].metadata!=all_yr_list[0].metadata:
                    all_yr_list[-1].metadata = all_yr_list[0].metadata
                all_yr_list.append(yr_cube)
        return all_yr_list


    def era5_mon_data(var, years=np.arange(1959,2023)):
        '''
        input: variable as string, years as list,
        var options include: t2m, msl, z500, tmax ...
        defaults: years 1950-2022.
        returns list of cubes (one per year) of daily variable, May-Sep
        '''
        all_yr_list = iris.cube.CubeList()
        for yr in years:
            print(yr)
            files = glob.glob('/net/pc230042/nobackup/users/sager/nobackup_2_old/ERA5/'+str(yr)+'/mon/era5_'+var+'_*') # numbers indicate months extracted
            yr_cubes = iris.load_cube(files) # should just be one file, so one cube
            all_yr_list.append(yr_cubes)
            #rm_att = equalise_attributes(yr_cubes)
            #unify_time_units(yr_cubes)
        return all_yr_list


    def lentis_data(Pth='/net/pc200021/nobackup_1/users/muntjewe/LENTIS/', run='PD', var='pr'):
        '''
        #Pth = '/net/pc200272/nobackup/users/wiel/LENTIS/PD/day/pr'
        run = PD or 2K, string
        var = pr or ...  , string
        '''
        cubes = iris.cube.CubeList([]) # empty cube list
        files_list = glob.glob(Pth+run+'/day/'+var+'/*')
        for each in range(len(files_list)):
            #print(each)
            cubes.append(iris.load(files_list[each])[0])   
        # Add realization coord
        for n, each in enumerate(cubes):
            #print(n)
            N = each.attributes['realization_index']
            new_coord = iris.coords.AuxCoord(N, 'realization')
            each.add_aux_coord(new_coord)
        # Fix variation in coord bounds
        for n, each in enumerate(cubes):
            if cubes[0].coord('latitude') != cubes[n].coord('latitude'):
                print(n)
                cubes[n].coord('latitude').bounds = cubes[0].coord('latitude').bounds
                cubes[n].coord('latitude').points = cubes[0].coord('latitude').points
        # Merge to one cube (real x time x lat x lon)
        equalise_attributes(cubes)
        return cubes.merge_cube()


    def lentis_mydata(run='PD'):
        '''
        Pth = '/net/pc200272/nobackup/users/wiel/LENTIS/PD/day/pr'
        run = PD or 2D, string
        var = pr or ...  , string
        '''
        if run == 'PD':
            run_pwd = 'psi250'
        elif run == 'F':
            run_pwd = 's_psi250'
        cubes = iris.cube.CubeList([]) # empty cube list
        files_list = glob.glob('/net/pc200023/nobackup/users/thompson/LENTIS/'+run_pwd+'/*')
        for each in range(len(files_list)):
            #print(each)
            cubes.append(iris.load(files_list[each])[0])   
        # Add realization coord
        for n, each in enumerate(cubes):
            #print(n)
            N = each.attributes['realization_index']
            new_coord = iris.coords.AuxCoord(N, 'realization')
            each.add_aux_coord(new_coord)
        # Fix variation in coord bounds
        for n, each in enumerate(cubes):
            if cubes[0].coord('latitude') != cubes[n].coord('latitude'):
                print(n)
                cubes[n].coord('latitude').bounds = cubes[0].coord('latitude').bounds
                cubes[n].coord('latitude').points = cubes[0].coord('latitude').points
        # Merge to one cube (real x time x lat x lon)
        equalise_attributes(cubes)
        x = cubes.merge_cube()
        return iris.util.squeeze(x)

    def regrid(original, new):
        ''' Regrids onto a new grid '''
        mod_cs = original.coord_system(iris.coord_systems.CoordSystem)
        new.coord(axis='x').coord_system = mod_cs
        new.coord(axis='y').coord_system = mod_cs
        new_cube = original.regrid(new, iris.analysis.Linear())
        return new_cube

    def extract_JJA(cube_list):
        '''
        Extract JJA, works for single cube or list
        '''
        if isinstance(cube_list, iris.cube.Cube):
            if len(cube_list.coords('season')) > 0:
                pass
            else:
                iris.coord_categorisation.add_season(cube_list, 'time')
            jja_cubes = cube_list.extract(iris.Constraint(season='jja'))
        elif isinstance(cube_list, iris.cube.CubeList):
            jja_cubes = iris.cube.CubeList([])
            for each in range(len(cube_list)):
                print(each)
                if len(cube_list[each].coords('season')) > 0:
                    pass
                else:
                    iris.coord_categorisation.add_season(cube_list[each], 'time')
                jja_cubes.append(cube_list[each].extract(iris.Constraint(season='jja')))
        return jja_cubes


    def extract_SON(cube_list):
        '''
        Extract SON, works for single cube or list
        '''
        if isinstance(cube_list, iris.cube.Cube):
            if len(cube_list.coords('season')) > 0:
                pass
            else:
                iris.coord_categorisation.add_season(cube_list, 'time')
            jja_cubes = cube_list.extract(iris.Constraint(season='son'))
        elif isinstance(cube_list, iris.cube.CubeList):
            jja_cubes = iris.cube.CubeList([])
            for each in range(len(cube_list)):
                print(each)
                if len(cube_list[each].coords('season')) > 0:
                    pass
                else:
                    iris.coord_categorisation.add_season(cube_list[each], 'time')
                jja_cubes.append(cube_list[each].extract(iris.Constraint(season='son')))
        return jja_cubes



    def extract_DJF(cube_list):
        '''
        Extract DJF, works for single cube or list
        '''
        if isinstance(cube_list, iris.cube.Cube):
            if len(cube_list.coords('season')) > 0:
                pass
            else:
                iris.coord_categorisation.add_season(cube_list, 'time')
            djf_cubes = cube_list.extract(iris.Constraint(season='djf'))
        elif isinstance(cube_list, iris.cube.CubeList):
            djf_cubes = iris.cube.CubeList([])
            for each in range(len(cube_list)):
                print(each)
                if len(cube_list[each].coords('season')) > 0:
                    pass
                else:
                    iris.coord_categorisation.add_season(cube_list[each], 'time')
                djf_cubes.append(cube_list[each].extract(iris.Constraint(season='djf')))
        return djf_cubes


    def extract_region(cube_list, R1):
        '''
        Extract Region (defaults to Europe)
        '''
        const_lat = iris.Constraint(latitude = lambda cell:R1[1] < cell < R1[0])
        if isinstance(cube_list, iris.cube.Cube):
            reg_cubes_lat = cube_list.extract(const_lat)
            reg_cubes = reg_cubes_lat.intersection(longitude=(R1[3], R1[2]))
        elif isinstance(cube_list, iris.cube.CubeList):
            reg_cubes = iris.cube.CubeList([])
            for each in range(len(cube_list)):
                print(each)
                subset = cube_list[each].extract(const_lat)
                reg_cubes.append(subset.intersection(longitude=(R1[3], R1[2])))
        return reg_cubes


    def extract_Plev(cube_list, lev=250):
        '''
        Extract specific vertical level. based on pressue, works for single cube or list
        lev = pressure in hPa, default 250 hPa
        '''
        lev = lev*100
        if isinstance(cube_list, iris.cube.Cube):
            lev_cubes = cube_list.extract(iris.Constraint(air_pressure=lev))
        elif isinstance(cube_list, iris.cube.CubeList):
            lev_cubes = iris.cube.CubeList([])
            for each in range(len(cube_list)):
                print(each)
                lev_cubes.append(cube_list[each].extract(iris.Constraint(air_pressure=lev)))
        return lev_cubes

    def extract_date(cube, yr, mon, day):
        '''
        Extract specific day from cube of a single year
        '''
        if len(cube.coords('year')) > 0:
            pass
        else:
            iris.coord_categorisation.add_year(cube, 'time')
        if len(cube.coords('month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_month(cube, 'time')
        if len(cube.coords('day_of_month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_day_of_month(cube, 'time')
        return cube.extract(iris.Constraint(year=yr, month=mon, day_of_month=day))


    def cube_date(cube):
        '''
        Returns date of cube (assumes cube single day)
        '''
        if len(cube.coords('year')) > 0:
            pass
        else:
            iris.coord_categorisation.add_year(cube, 'time')
        if len(cube.coords('month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_month(cube, 'time')
        if len(cube.coords('day_of_month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_day_of_month(cube, 'time')
        if len(cube.coords('day_of_year')) > 0:
            pass
        else:
            iris.coord_categorisation.add_day_of_year(cube, 'time')
        YY = cube.coord('time').units.num2date(cube.coord('time').points)[0].year
        MM = cube.coord('time').units.num2date(cube.coord('time').points)[0].month
        DD = cube.coord('time').units.num2date(cube.coord('time').points)[0].day
        T = cube.coord('time').points[0]
        return YY, MM, DD, T

    def pull_out_day_era(psi, sel_year, sel_month, sel_day):
        if type(psi)==iris.cube.Cube:
            psi_day = Analogues.extract_date(psi, sel_year, sel_month, sel_day)
        else:
            for each in psi:
                if len(each.coords('year')) > 0:
                    pass
                else:
                    iris.coord_categorisation.add_year(each, 'time')
                if each.coord('year').points[0]==sel_year:
                    print(each.coord('year').points[0])
                    psi_day = Analogues.extract_date(each, sel_year, sel_month, sel_day)
                else:
                    pass
        try:
            return psi_day
        except NameError:
            print('ERROR: Date not in data')
            return


    def pull_out_day_lentis(psi, sel_R, sel_year, sel_month, sel_day):
        psi_real = psi.extract(iris.Constraint(realization=sel_R)) 
        psi_day = Analogues.extract_date(psi_real, sel_year, sel_month, sel_day)
        try:
            return psi_day
        except NameError:
            print('ERROR: Date not in data')
            return


    def var_event_data(var, R1, date):
        '''
        For specified variable, region, and date, returns the spatial field
        '''
        my_var = ['psi250', 'u250', 'v250']
        if var in my_var:
            var_cube = Analogues.era5_mydata(var, range(date[0],date[0]+1))
        else:
            var_cube = Analogues.era5_data(var, range(date[0],date[0]+1))
        var_reg = Analogues.extract_region(var_cube, R1)
        if var == 'psi250':
            iris.coord_categorisation.add_month(var_reg, 'time')
            iris.coord_categorisation.add_day_of_month(var_reg, 'time')
        else: pass
        return Analogues.pull_out_day_era(var_reg, date[0], date[1], date[2])

    #
    ##
    ### Composites
    ##
    #

    def composite_dates(psi, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''
        n = len(date_list)
        FIELD = 0
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_era(psi, year, month, day)
            if NEXT_FIELD == None:
                print('Field failure for: ',+each)
                n = n-1
            else:
                if FIELD == 0:
                    FIELD = NEXT_FIELD
                else:
                    FIELD = FIELD + NEXT_FIELD
        return FIELD/n



    def composite_dates_significance(psi, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''
        n = len(date_list)
        field_list = iris.cube.CubeList([])
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            field_list.append(Analogues.pull_out_day_era(psi, year, month, day))
        sig_field = field_list[0].data
        a, b = np.shape(field_list[0].data)
        for i in range(a):
            print(i)
            for j in range(b):
                loc_list = []
                for R in range(n):
                    loc_list.append(field_list[R].data[i,j])
                if np.abs(np.mean(loc_list)) > np.abs(np.std(loc_list)):
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list[0]
        result_cube.data = sig_field
        return result_cube



    def composite_dates_change_significance(psi1, dates1, psi2, dates2):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''    
        n = len(dates1)
        field_list1 = iris.cube.CubeList([])
        for each in range(n):
            year = int(dates1[each][:4])
            month = calendar.month_abbr[int(dates1[each][4:-2])]
            day = int(dates1[each][-2:])
            field_list1.append(Analogues.pull_out_day_era(psi1, year, month, day))
        n = len(dates2)
        field_list2 = iris.cube.CubeList([])
        for each in range(n):
            year = int(dates2[each][:4])
            month = calendar.month_abbr[int(dates2[each][4:-2])]
            day = int(dates2[each][-2:])
            field_list2.append(Analogues.pull_out_day_era(psi2, year, month, day))    
        sig_field = field_list1[0].data
        a, b = np.shape(field_list1[0].data)
        for i in range(a):
            print(i)
            for j in range(b):
                loc_list1 = []; loc_list2 = []
                for R in range(n):
                    loc_list1.append(field_list1[R].data[i,j])
                    loc_list2.append(field_list2[R].data[i,j])
                    #u, p = stats.mannwhitneyu(loc_list1,loc_list2)
                    u, p = stats.ttest_ind(loc_list1, loc_list2, equal_var=False, alternative='two-sided')
                if p < 0.05:
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list1[0]
        result_cube.data = sig_field
        return result_cube


    def composite_dates_change_significance_lentis(psi1, dates1, psi2, dates2):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''    
        n = len(dates1)
        field_list1 = iris.cube.CubeList([])
        for each in range(n):
            real = int(dates1[each][:4])
            year = int(dates1[each][4:8])
            month = calendar.month_abbr[int(dates1[each][8])]
            day = int(dates1[each][-2:])
            field_list1.append(Analogues.pull_out_day_lentis(psi1, real, year, month, day))
        n = len(dates2)
        field_list2 = iris.cube.CubeList([])
        for each in range(n):
            real = int(dates2[each][:4])
            year = int(dates2[each][4:8])
            month = calendar.month_abbr[int(dates2[each][8])]
            day = int(dates2[each][-2:])
            field_list2.append(Analogues.pull_out_day_lentis(psi2, real, year, month, day))
        sig_field = field_list1[0].data
        a, b = np.shape(field_list1[0].data)
        for i in range(a):
            print(i)
            for j in range(b):
                loc_list1 = []; loc_list2 = []
                for R in range(n):
                    loc_list1.append(field_list1[R].data[i,j])
                    loc_list2.append(field_list2[R].data[i,j])
                    #u, p = stats.mannwhitneyu(loc_list1,loc_list2)
                    u, p = stats.ttest_ind(loc_list1, loc_list2)
                if p < 0.05:
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list1[0]
        result_cube.data = sig_field
        return result_cube

    def composite_max_dates(psi, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''
        n = len(date_list)
        field_list = iris.cube.CubeList([])
        for each in range(n):
            real = int(date_list[each][:4])
            year = int(date_list[each][4:8])
            month = calendar.month_abbr[int(date_list[each][8])]
            day = int(date_list[each][-2:])
            field_list.append(Analogues.pull_out_day_era(psi, year, month, day))
        max_field = field_list[0].data
        a, b = np.shape(field_list[0].data)
        for i in range(a):
            print(i)
            for j in range(b):
                loc_list = []
                for R in range(n):
                    loc_list.append(field_list[R].data[i,j])
                max_field[i,j] = np.max(loc_list)
        result_cube = field_list[0]
        result_cube.data = max_field
        return result_cube


    def composite_max_dates_lentis(psi, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''
        n = len(date_list)
        field_list = iris.cube.CubeList([])
        for each in range(n):
            real = int(date_list[each][:4])
            year = int(date_list[each][4:8])
            month = calendar.month_abbr[int(date_list[each][8])]
            day = int(date_list[each][-2:])
            field_list.append(Analogues.pull_out_day_lentis(psi, real, year, month, day))
        max_field = field_list[0].data
        a, b = np.shape(field_list[0].data)
        for i in range(a):
            print(i)
            for j in range(b):
                loc_list = []
                for R in range(n):
                    loc_list.append(field_list[R].data[i,j])
                max_field[i,j] = np.max(loc_list)
        result_cube = field_list[0]
        result_cube.data = max_field
        return result_cube


    def composite_dates_lentis(psi, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''
        n = len(date_list)
        FIELD = 0
        for each in range(n):
            real = int(date_list[each][:4])
            year = int(date_list[each][4:8])
            month = calendar.month_abbr[int(date_list[each][8])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_lentis(psi, real, year, month, day)
            if FIELD == 0:
                FIELD = NEXT_FIELD
            else:
                FIELD = FIELD + NEXT_FIELD
        return FIELD/n

    def composite_dates_significance_lentis(psi, date_list):
        '''
        Returns single composite of all dates
        Inputs required:
        psi = list of cubes, 1 per year - as used to calc D/date_list
        date_list = list of events to composite
        '''
        n = len(date_list)
        field_list = iris.cube.CubeList([])
        for each in range(n):
            real = int(date_list[each][:4])
            year = int(date_list[each][4:8])
            month = calendar.month_abbr[int(date_list[each][8])]
            day = int(date_list[each][-2:])
            field_list.append(Analogues.pull_out_day_lentis(psi, real, year, month, day))
        sig_field = field_list[0].data
        a, b = np.shape(field_list[0].data)
        for i in range(a):
            print(i)
            for j in range(b):
                loc_list = []
                for R in range(n):
                    loc_list.append(field_list[R].data[i,j])
                if np.abs(np.mean(loc_list)) > np.abs(np.std(loc_list)):
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list[0]
        result_cube.data = sig_field
        return result_cube


    #
    ##
    ### Euclidean Distance Functions
    ##
    #

    def euclidean_distance(psi, event):
        '''
        Returns list of D
        Inputs required:
        psi = single cube of JJA psi.
        event = cube of single day of event to match.
        BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
        '''
        D = [] # to be list of all euclidean distances
        if type(psi) is iris.cube.CubeList:
            for each in psi:
                a, b, c = np.shape(each)
                XA = event.data.reshape(b*c,1)
                #yrs.append(each.coord('year').points[0])
                XB = each.data.reshape(np.shape(each.data)[0], b*c, 1)
                for Xb in XB:
                    D.append(np.sqrt(np.sum(np.square(XA - Xb))))
        else:
            a, b, c = np.shape(psi)
            XA = event.data.reshape(b*c,1)
            #yrs.append(each.coord('year').points[0])
            XB = psi.data.reshape(np.shape(psi.data)[0], b*c, 1)
            for Xb in XB:
                D.append(np.sqrt(np.sum(np.square(XA - Xb))))
        return D


    def correlation_coeffs(field, event):
        '''
        Returns list of C, calc using linear pearsonr
        Inputs required:
        psi = single cube of JJA psi.
        event = cube of single day of event to match.
        BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
        '''
        C = [] # to be list of all correlation coefficients
        a, b, c = np.shape(field)
        for i in range(a):
            day = field[i,:,:]
            XA = event.data.reshape(b*c)
            XB = day.data.reshape(b*c)
            C.append(pearsonr(XA, XB)[0])
        # C become 1-C so that smallest value is closest fit (matching D)
        return [1-x for x in C]

    def euclidean_dates(psi, D, N):
        '''
        Returns list of dates of the analogs
        Inputs required:
        psi = single cube
        D = array of euclidean distance, yr x day.
        N = how many events to identify
        '''             
        date_list = []
        time_list = []
        for i in np.arange(N):
            a1 = np.where(D == np.sort(D)[i])[0][0] 
            YY, MM, DD, T = Analogues.cube_date(psi[a1,...])
            #print(YY, MM, DD)
            if DD<10:
                date = str(YY)+str(MM)+str(0)+str(DD)
            else:
                date = str(YY)+str(MM)+str(DD)
            date_list.append(date)
            time_list.append(T)
        return date_list, time_list

    def date_list_checks(time_list, date_list, event, days=5):
        '''
        Takes date_list and removes:
        1) the original event (if present)
        2) any days within 5 days of another event
        '''
        new_date_list = date_list.copy()
        # Remove event date (if present)
        if event.coord('time').points[0] in time_list:
            new_date_list.remove(date_list[time_list.index(event.coord('time').points[0])])
        else: pass
        # Remove neighbours (if present). Auto 120hrs = 5 days
        hrs = days*24
        for i in np.arange(1,len(time_list)):
            for i_earlier in np.arange(i):
                if (time_list[i]-hrs)<=time_list[i_earlier]<=(time_list[i]+hrs):
                    new_date_list.remove(date_list[i])
                    break
                else:
                    pass
        return new_date_list


    def list_of_analogs(var, R1, event_date, N, Y1, Y2):
        '''
        Returns a list of the dates of the top N analogs calculated by Euclidean distance over region R1, eliminating duplicates
        '''
        my_var = ['psi250', 'u250', 'v250', 'z500']
        if var in my_var:
            var_cube = Analogues.era5_mydata(var, range(Y1,Y2))
        else:
            var_cube = Analogues.era5_data(var, range(Y1,Y2))
            # this is a list - needs merging to one
        var_reg = Analogues.extract_region(var_cube, R1)  
        event = Analogues.var_event_data(var, R1, event_date)
        D = Analogues.euclidean_distance(var_reg, event)
        date_list_present, T_present = Analogues.euclidean_dates(var_reg, D, N*5)
        new_date_list = Analogues.date_list_checks(T_present, date_list_present, event)
        if len(new_date_list)<N:
            print('NOT ENOUGH PRESENT DAY ANALOGS IDENTIFIED')
        else:
            pass
        return new_date_list

    def list_of_analogs_corrcoeff(var, R1, event_date, N, Y1, Y2):
        '''
        Returns a list of the dates of the top N analogs calculated by correlation ceoff over region R1, eliminating duplicates
        '''
        my_var = ['psi250', 'u250', 'v250', 'z500']
        if var in my_var:
            var_cube = Analogues.era5_mydata(var, range(Y1,Y2))
        else:
            var_cube = Analogues.era5_data(var, range(Y1,Y2))
            # this is a list - needs merging to one
        var_reg = Analogues.extract_region(var_cube, R1)  
        event = Analogues.var_event_data(var, R1, event_date)
        D = Analogues.correlation_coeffs(var_reg, event)
        date_list_present, T_present = Analogues.euclidean_dates(var_reg, D, N*5)
        new_date_list = Analogues.date_list_checks(T_present, date_list_present, event)
        if len(new_date_list)<N:
            print('NOT ENOUGH PRESENT DAY ANALOGS IDENTIFIED')
        else:
            pass
        return new_date_list



    def analogs_datelist(psi, event, N=50):
        '''
        Returns a list of the dates of the top N analogs calculated by Euclidean distance over region R1, eliminating duplicates
        '''
        D = Analogues.euclidean_distance(psi, event)
        date_list_present, T_present = Analogues.euclidean_dates(psi, D, N*5)
        new_date_list = Analogues.date_list_checks(T_present, date_list_present, event)
        if len(new_date_list)<N:
            print('NOT ENOUGH PRESENT DAY ANALOGS IDENTIFIED')
        else:
            pass
        return new_date_list


    def list_of_analogs_lentis(var_cube, event, R, N):
        '''
        Returns a list of the dates of the top N analogs calculated by Euclidean distance over region R1, eliminating duplicates
        '''
        d = []
        for each in range(R):   # 160 realisations, 20 for testing
            print(each)
            D = Analogues.euclidean_distance(var_cube[each, ...], event)
            d.append(D)
        # identify limit level
        a, b = np.shape(d)
        d_limit = np.sort(np.asarray(d).reshape(a*b,))[N*5]
        date_list = []; time_list = []; euc_dist = []
        count = 0
        for R in np.arange(a):
            for day in np.arange(b):
                if d[R][day] <= d_limit:
                    count+=1; print(count); print(d[R][day])
                    YY, MM, DD, T = Analogues.cube_date(var_cube[R,day,...])
                    date = "{0:0=4d}".format(var_cube.coord('realization').points[R])+str(YY)+str(MM)+"{0:0=2d}".format(DD)
                    date_list.append(date)
                    time_list.append(T)
                    euc_dist.append(d[R][day])
                    print('Ensemble complete:',R)
        new_date_list, new_euc_dist = Analogues.date_list_checks_lentis(time_list, date_list, euc_dist)
        if len(new_date_list)<N:
            print('NOT ENOUGH PRESENT DAY ANALOGS IDENTIFIED')
        else:
            pass
        return new_date_list, new_euc_dist


    def date_list_checks_lentis(time_list, date_list, euc_dist, days=5):
        '''
        Takes date_list and removes:
        1) the original event (if present)
        2) any days within 5 days of another event
        '''
        new_date_list = date_list.copy()
        new_euc_dist = euc_dist.copy()
        # Remove neighbours (if present). Auto 120hrs = 5 days
        hrs = days*24
        for i in np.arange(1,len(time_list)):
            for i_earlier in np.arange(i):
                if (time_list[i]-hrs)<=time_list[i_earlier]<=(time_list[i]+hrs) and date_list[i_earlier][:4] == date_list[i][:4]:
                    new_date_list.remove(date_list[i])
                    new_euc_dist.remove(euc_dist[i])
                    break
                else:
                    pass
        return new_date_list, new_euc_dist


    def list_of_corr_analogs(var, R1, event_date, N, Y1, Y2):
        '''
        Returns a list of the dates of the top N analogs calculated by Euclidean distance over region R1, eliminating duplicates
        '''
        my_var = ['psi250', 'u250', 'v250', 'z500']
        if var in my_var:
            var_cube = Analogues.era5_mydata(var, range(Y1,Y2))
        else:
            var_cube = Analogues.era5_data(var, range(Y1,Y2))
            # this is a list - needs merging to one
        var_reg = Analogues.extract_region(var_cube, R1)  
        event = Analogues.var_event_data(var, R1, event_date)
        C = Analogues.correlation_coeffs(var_reg, event)
        date_list_present, T_present = Analogues.euclidean_dates(var_reg, C, N*5)
        new_date_list = Analogues.date_list_checks(T_present, date_list_present, event)
        if len(new_date_list)<N:
            print('NOT ENOUGH PRESENT DAY ANALOGS IDENTIFIED')
        else:
            pass
        return new_date_list

    def events_in_both_lists(dates1, dates2):
        '''
        Identify dates that are in both lists (+/- 2 days)
        '''
        date_list = []
        for each in dates1:
            if each in dates2 or str(int(each)-1) in dates2 or str(int(each)+1) in dates2:
                date_list.append(each)
            elif str(int(each)-2) in dates2 or str(int(each)+2) in dates2:
                date_list.append(each)
            else: pass
        return date_list
    #
    ##
    ### Analogue quality (from euclidean distance)
    ##
    #

    def euclidean_quality(psi, N=30):
        L = np.shape(psi.data)[0]-1
        Q = []
        for each_day in np.arange(L):
            #print(each_day)
            event_day = psi[each_day,:,:]
            D = Analogues.euclidean_distance(psi, event_day) # calc euclidean distances
            Q.append(np.sum(np.sort(D)[:N])) # sum of closest 30 events
        return Q


    def euclidean_quality_event(psi, event, N=30):
        D = Analogues.euclidean_distance(psi, event) # calc euclidean distances
        # sum of closest 30 events
        return np.sum(np.sort(D)[:N])

    def euclidean_quality_analogs(psi, date_list, N=30):
        '''
        For the 30 closest analogs of the event day, calculates analogue quality
        '''
        Q = []
        for i, each in enumerate(date_list):
            YY = int(each[:4])
            MM = calendar.month_abbr[int(each[4:-2])]
            DD = int(each[-2:])
            analog_event = Analogues.pull_out_day_era(psi, YY, MM, DD)
            D = Analogues.euclidean_distance(psi, analog_event) # calc euclidean distances
            Q.append(np.sum(np.sort(D)[:N]))
        return Q


    def euclidean_quality_lentis(psi, N=30):
        '''
        Takes far too long so instead of assessing against all days, just use 10 realsiations for each one
        '''
        L = np.shape(psi)[1]-1
        Q = []
        for R in np.arange(np.shape(psi)[0]): 
            print(R)
            for each_day in np.arange(L):
                print(each_day)
                event_day = psi[R, each_day,:,:]
                d = []
                reals = []
                t = np.arange(0, 160)
                for _ in range(10):
                    chosen_real = random.choice(t)
                    reals.append(chosen_real)
                    t.remove(chosen_real)
                for each in reals:   # 10 out of 160 reals
                    #print(R, each)
                    D = Analogues.euclidean_distance(psi[each, ...], event_day)
                    d.append(D)
                Q.append(np.sum(np.sort(d)[:N])) # sum of closest 30 events
        return Q

    def euclidean_quality_event_lentis(psi, event, N=30):
        '''
        Takes far too long so instead of assessing against all days, just use 10 realsiations for each one
        '''
        d = []
        reals = []
        t = np.arange(0, np.shape(psi)[0])
        for _ in range(9):
            chosen_real = random.choice(t)
            if chosen_real in reals:
                pass
            else:
                reals.append(chosen_real)
        for each in reals:   # 160 realisations, 20 for testing
            print(each)
            D = Analogues.euclidean_distance(psi[each, ...], event)
            d.append(D)
        a, b = np.shape(d)
        # sum of closest 30 events
        return np.sum(np.sort(np.asarray(d).reshape(a*b,))[:N])


    def euclidean_quality_analogs_lentis(psi, date_list, N=30):
        '''
        For the 30 closest analogs of the event day, calculates analogue quality
        '''
        Q = []
        for i, each in enumerate(date_list):
            print(each)
            RR = int(each[:4])
            YY = int(each[4:8])
            MM = calendar.month_abbr[int(each[8])]
            DD = int(each[-2:])
            analog_event = Analogues.pull_out_day_lentis(psi, RR, YY, MM, DD)
            Q.append(Analogues.euclidean_quality_event_lentis(psi, analog_event, N))
            print(len(Q))
        return Q





    #
    ##
    ###
    # Persistence Functions
    ###
    ##
    #

    def euclidean_persistence(psi, Dquanti):
        '''
        For each day in psi, calculates how long event persists
        Persists if next day within [quanti] percent of event
        '''
        L = np.shape(psi.data)[0]-1
        P = []
        for each_day in np.arange(L):
            #print(each_day)
            event_day = psi[each_day,:,:]
            D = Analogues.euclidean_distance(psi, event_day) # calc euclidean distances
            Pval = 0
            for i in np.arange(1,20):
                if D.index(0)+(i) == len(D):
                    break
                elif D[D.index(0)+i]<Dquanti:
                    Pval += 1
                else:
                    break
            P.append(Pval) 
        return P


    def euclidean_persistence_analogs(psi, event, date_list, Dquanti):
        '''
        For the 30 closest analogs of the event day, calculates how long event persists
        Persists if next day within [quanti] percent of event
        '''
        P = []
        for i, each in enumerate(date_list):
            print(i)
            YY = np.int(each[:4])
            MM = calendar.month_abbr[int(each[4])]
            DD = int(each[-2:])
            analog_event = Analogues.pull_out_day_era(psi, YY, MM, DD)
            D = Analogues.euclidean_distance(psi, analog_event) # calc euclidean distances
            Pval = 0
            for j in np.arange(1,20):
                if D.index(0)+j>=219:
                    pass
                elif D[D.index(0)+j]<Dquanti:
                    Pval += 1
                else:
                    break
            P.append(Pval)    
        return P


    def euclidean_persistence_event(psi, event, quanti=5):
        '''
        For event day, calculates how long event persists
        Persists if next day within [quanti] percent of event
        '''
        D = Analogues.euclidean_distance(psi, event) # calc euclidean distances
        percentile = int(len(D)/100*quanti)
        Dquanti = np.sort(D)[percentile] # closest X% of events
        Pval = 0
        for i in np.arange(1,20):
            if D[D.index(0)+i]<Dquanti:
                Pval += 1
            else:
                break
        return Pval, Dquanti


    def euclidean_persistence_analogs_lentis(psi, event, date_list, quanti=5):
        '''
        For the 30 closest analogs of the event day, calculates how long event persists
        Persists if next day within [quanti] percent of event
        '''
        P = []
        for i, each in enumerate(date_list):
            print(i)
            YY = np.int(each[4:8])
            MM = calendar.month_abbr[int(each[8])]
            DD = int(each[-2:])
            RR = int(each[:4])
            analog_event = Analogues.pull_out_day_lentis(psi, RR, YY, MM, DD)
            D = Analogues.euclidean_distance(psi[RR,...], analog_event) # calc euclidean distanc
            percentile = int(len(D)/100*quanti)
            Dquanti = np.sort(D)[percentile]
            Pval = 0
            for j in np.arange(1,20):
                if D.index(0)+j>=219:
                    pass
                elif D[D.index(0)+j]<Dquanti:
                    Pval += 1
                else:
                    break
            P.append(Pval)    
        return P