'''
15 Sep 2022 

Loading ssp_pop data and masking for all regions
@vikki.thompson
'''

# Load neccessary libraries
import iris
import iris.coord_categorisation as icc
from iris.coord_categorisation import add_season_membership
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
import cartopy.crs as ccrs
import cartopy as cart
import glob
import matplotlib.cm as mpl_cm
import sys
from iris.experimental.equalise_cubes import equalise_attributes
import scipy.stats as sps
from iris.analysis import geometry
import os
import cdsapi
import subprocess


def region_mask(cube, region_value):
    '''
    Masks except region
    '''
    new_cube = cube[0,:,:].copy()
    for i in range(np.shape(cube.data)[1]):
        print(i, 'out of', np.shape(cube.data)[1])
        for j in range(np.shape(cube.data)[2]):
            if cube.coord('regions')[i, j].points == region_value:
                new_cube.data[i,j] = 1
            else:
                new_cube.data[i, j] = np.NaN 
    masked = cube * new_cube
    return masked



############

## Get regions, put on obs grid
file =  "/user/home/hh21501/farfrom/tmp/region_fx-WRAF05-v4-1_WRAF_All-Hist_est1_v4-1_4-1-0_000000-000000.nc"   # 200+ regions
reg_grid = iris.load(file, 'Region_Index')[0]
print(reg_grid)

def regridded(original, new):
    ''' Regrids onto a new grid '''
    mod_cs = original.coord_system(iris.coord_systems.CoordSystem)
    new.coord(axis='x').coord_system = mod_cs
    new.coord(axis='y').coord_system = mod_cs
    new_cube = original.regrid(new, iris.analysis.Linear())
    return new_cube

def region_mask(cube, region_value):
    '''
    Masks except region
    '''
    new_cube = cube[0,:,:].copy()
    for i in range(np.shape(cube.data)[1]):
        print(i, 'out of', np.shape(cube.data)[1])
        for j in range(np.shape(cube.data)[2]):
            if cube.coord('regions')[i, j].points == region_value:
                new_cube.data[i,j] = 1
            else:
                new_cube.data[i, j] = np.NaN 
    masked = cube * new_cube
    return masked


###
file_pop = "/user/work/hh21501/ssp_pop/ssp5_2010.nc"
pop_2050 = iris.load(file_pop)[0]


new_reg = regridded(reg_grid, pop_2050)

## Add regions as an aux_coord 
pop_2050.add_aux_coord(iris.coords.AuxCoord(new_reg.data, long_name='regions'), data_dims=[0, 1])   

reg_data = []
for reg in np.arange(237):
    print(reg)
    x = np.where(pop_2050.coord('regions').points == reg)
    single_reg = []
    for i, j in zip(x[0], x[1]):
        #print(i,j)
        single_reg.append(pop_2050.data[i,j])
    reg_data.append(np.ma.sum(single_reg))


