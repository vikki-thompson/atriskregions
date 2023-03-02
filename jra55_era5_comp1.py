'''
Created 20 Jul 2022
Comparison of JRA55 and ERA5 annual max data
@vikki.thompson
'''

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
from scipy.stats import genextreme as gev
import random
import scipy.io
import xarray as xr
import netCDF4 as nc

def annmax_era(reg):
    return np.load('/user/work/hh21501/era5_regions/ann_list_19592021.npy')[:,reg]-273.15

def annmax_jra(reg):
    x = np.load('/user/work/hh21501/jra55_regions/reg_data.npy')-273.15
    reg_data = x[reg][1:-1]
    #year = np.arange(1958, 2022)
    return reg_data


def compare_annmax(reg):
    plt.figure()
    data_era = annmax_era(reg)
    data_jra = annmax_jra(reg)
    #year_era = np.arange(1959, 2022)
    plt.plot(year, data_era, 'b', label='ERA5')
    plt.plot(year, data_jra, 'r', label='JRA55')
    plt.legend()
    plt.title('Region: '+str(reg))


def adjust_obs(ann_max): 
    offset = []
    a, b = np.polyfit(GMST, ann_max, 1)
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1.2*a+b)) #1*a+b means adjusted for world with 1deg warming  
    return offset

def year_max_adjust(ann_max):
    x = adjust_obs(ann_max)
    year = np.arange(1959, 2022)
    yr = year[np.where(x==np.amax(x))[0]][0]
    return yr

def year_max(ann_max):
    x = ann_max
    year = np.arange(1959, 2022)
    yr = year[np.where(x==np.amax(x))[0]][0]
    return yr

def year_max_multiple_adjust(ann_max, how_many):
    x = adjust_obs(ann_max)
    year = np.arange(1959, 2022)
    yr = []
    for each in np.sort(x)[-how_many:]:
        yr.append(year[np.where(x==each)[0]][0])
    return yr

def compare_annmax_offset(reg):
    plt.figure()
    data_era = adjust_obs(annmax_era(reg))
    data_jra = adjust_obs(annmax_jra(reg))
    #year_era = np.arange(1959, 2022)
    plt.plot(year, data_era, 'b', label='ERA5')
    plt.plot(year, data_jra, 'r', label='JRA55')
    plt.legend()
    plt.title('Offset, Region: '+str(reg))


#####

# NASA GISS
GMST = [0.03, -0.03, 0.06, 0.03, 0.05, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05, 0.03, -0.08, 0.01, 0.16, -0.07, -0.01, -0.10, 0.18, 0.07, 0.17, 0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27, 0.45, 0.41, 0.22, 0.23, 0.32, 0.45, 0.33, 0.46, 0.61, 0.38, 0.39, 0.54, 0.63, 0.62, 0.54, 0.68, 0.64, 0.67, 0.54, 0.66, 0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.02, 0.92, 0.85, 0.98, 1.02, 0.85]

year = np.arange(1959, 2022)

##
year = year[31:]
GMST=GMST[31:]

list = []
for reg in range(237):
    jra_yr = year_max_multiple_adjust(annmax_jra(reg)[31:], 5)
    era_yr = year_max_adjust(annmax_era(reg)[31:])
    if era_yr in jra_yr:
        list.append(1)
    else:
        list.append(0)
    print(reg)

# era in top 5 of jra: 170/237

# how many agree? or is top for era in top 5 for jra?

mask_regs = list


plt.ion()
plt.show()
compare_annmax(202)

# Riskiest eliminated region: Mexico. 221
compare_annmax(221)
compare_annmax_offset(221)

reg = 74
np.max(adjust_obs(annmax_era(reg)))

# global
def offset_higher(reg):
    ''' loads data, 
    applies offset to present day, 
    calcs difference between observed max and 1-in-10000 event'''
    ann_max = annmax_era(reg)
    # Adjust ann max to distance from best fit
    offset = []
    a, b = np.polyfit(GMST, ann_max, 1)
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
    val = how_much_higher(offset)
    return val

def offset_higher_removeevent(reg):
    ''' loads data, 
    applies offset to present day, 
    calcs difference between observed max and 1-in-10000 event'''
    ann_max = annmax_era(reg)
    ann_max_new, GMST_new = remove_max(ann_max, GMST)
    # Adjust ann max to distance from best fit
    offset = []
    a, b = np.polyfit(GMST_new, ann_max_new, 1)
    for i, each in enumerate(ann_max_new):
        actual = each
        predicted = GMST_new[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
    tenthos = one_in_tenthousand(offset)
    offset_full = []
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset_full.append(actual-predicted+(1*a+b))
    max_val = np.max(offset_full)
    return tenthos - max_val

def obs_max_ret_per_removeevent(reg):
    ' Returns the highest event in terms of return period, yrs '
    ann_max = annmax_era(reg)
    ann_max_new, GMST_new = remove_max(ann_max, GMST)
    offset = []
    a, b = np.polyfit(GMST_new, ann_max_new, 1)
    for i, each in enumerate(ann_max_new):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
    #max_val = np.max(offset) # Maximum observed
    shape, loc, scale = gev.fit(offset) # EVT distribution
    x_val = np.linspace(np.min(offset)-.5, np.max(offset)+2, 1000)
    dist_pdf = gev.pdf(x_val, shape, loc, scale)
    offset_full = []
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset_full.append(actual-predicted+(1*a+b))
    max_val = np.max(offset_full)
    chance = []
    for i, _ in enumerate(dist_pdf):
        P = []
        for a, b in zip(dist_pdf[i:-1], dist_pdf[i+1:]):
            P.append(((a+b) / 2) * (x_val[1] - x_val[0])) 
        chance.append(sum(P)*100)
    x = chance[np.abs(x_val - max_val).argmin()] # chance of observed max
    if x == 0:
        result = 99999
    else:
        result = 100.*1/x
    return result

def obs_max_ret_per(reg):
    ' Returns the highest event in terms of return period, yrs '
    ann_max = annmax_era(reg)
    offset = []
    a, b = np.polyfit(GMST, ann_max, 1)
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
    max_val = np.max(offset) # Maximum observed
    shape, loc, scale = gev.fit(offset) # EVT distribution
    x_val = np.linspace(np.min(offset)-.5, np.max(offset)+2, 1000)
    dist_pdf = gev.pdf(x_val, shape, loc, scale)
    chance = []
    for i, _ in enumerate(dist_pdf):
        P = []
        for a, b in zip(dist_pdf[i:-1], dist_pdf[i+1:]):
            P.append(((a+b) / 2) * (x_val[1] - x_val[0])) 
        chance.append(sum(P)*100)
    x = chance[np.abs(x_val - max_val).argmin()] # chance of observed max
    return 100.*1/x

def offset_maxval(reg):
    ann_max = annmax_era(reg)
    offset = []
    a, b = np.polyfit(GMST, ann_max, 1)
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
    max_val = np.max(offset) # Maximum observed
    return max_val

def abs_max(reg):
    ann_max = annmax_era(reg)
    return np.max(ann_max)

def remove_max(ann_max, GMST):
    ' Remove the max value from region, and corresponding GMST value '
    ann_max_new = np.delete(ann_max, np.abs(ann_max - np.max(ann_max)).argmin())
    GMST_new = np.delete(GMST, np.abs(ann_max - np.max(ann_max)).argmin())
    return ann_max_new, GMST_new

def one_in_tenthousand(data_array):
    shape, loc, scale = gev.fit(data_array)
    x_val = np.linspace(np.min(data_array)-.5, np.max(data_array)+2, 1000)
    dist_pdf = gev.pdf(x_val, shape, loc, scale)
    ret_lev, chance = return_levels_plot(dist_pdf, x_val)
    return ret_lev[np.abs(np.asarray(chance)-0.01).argmin()]

def how_much_higher(data_array):
    mod = one_in_tenthousand(data_array)
    obs = np.max(data_array)
    return mod-obs


### GEV functions
def return_levels_plot(distribution_pdf, x_values):
    '''
    Calculates probability of return levels 
    '''
    chance = []
    return_level = []
    for i, _ in enumerate(distribution_pdf):
        width = x_values[1] - x_values[0]
        P = []
        for a, b in zip(distribution_pdf[i:-1], distribution_pdf[i+1:]):
            P.append(((a+b) / 2) * width) 
        chance.append(sum(P)*100)
        return_level.append(x_values[i])
    return return_level, chance



## Calc vals for each region
vals_retper_without = []
vals_abs_without = []
#vals_retper_with = []
#vals_abs_with = []
for each in np.arange(237):
    print(each)
    vals_retper_without.append(obs_max_ret_per_removeevent(each))
    vals_abs_without.append(offset_higher_removeevent(each))
    #vals_retper_with.append(obs_max_ret_per(each))
    #vals_abs_with.append(offset_higher(each))

vals_retper_without = [np.nan if x == 99999 else x for x in vals_retper_without]


# nan regions to be included (they are where era in top 5 of jra)
mask_regs = [np.nan if x == 1 else x for x in mask_regs] 




## Transfer vals to regional data
region_fp = 'region_fx-WRAF05-v4-1_WRAF_All-Hist_est1_v4-1_4-1-0_000000-000000.nc'
region_ds = xr.open_mfdataset(region_fp, parallel=True)
lats = region_ds.lat
lons = region_ds.lon
region_abs = region_ds.region
region_retper = region_ds.region
region_mask = region_ds.region
for region in range(237):
    print('region',region)
    region_abs = region_abs.where(region_ds.region.values != region, vals_abs_without[region])
    region_retper = region_retper.where(region_ds.region.values != region, vals_retper_without[region])
    region_mask = region_mask.where(region_ds.region.values != region, mask_regs[region])



## Figure of two maps
fig, axs = plt.subplots(2, 1, figsize=(10., 7.), dpi=80, num=None, subplot_kw={'projection': ccrs.PlateCarree()})
# upper plot
c = axs[0].contourf(lons,lats,region_abs,11,transform=ccrs.PlateCarree(),
                    cmap=mpl_cm.get_cmap('brewer_RdBu_11'), linestyle='solid',
                    vmin=-3.5, vmax=3.5)
cbar = plt.colorbar(c, shrink=0.7, ax=axs[0])
a = axs[0].contourf(lons,lats,region_mask,11,transform=ccrs.PlateCarree(),
                    colors='k',
                    vmin=-1, vmax=1)
cbar.set_label(u"\u2103")
cbar.outline.set_linewidth(0.5)
cbar.axs[0].tick_params(labelsize=6,width=0.5)
axs[0].set_title('Difference between 10000yr event & current record')
axs[0].text(-190, 95, 'a')

# lower plot
from matplotlib import ticker, cm
e = axs[1].contourf(lons,lats,region_abs,11,transform=ccrs.PlateCarree(),
                    colors='c',
                    vmin=-10, vmax=30)
d = axs[1].contourf(lons,lats,region_retper,11,transform=ccrs.PlateCarree(),
                    locator=ticker.LogLocator(),
                    cmap=mpl_cm.get_cmap('Reds'),
                    vmin=0.1, vmax=10000, extend='max')
a = axs[1].contourf(lons,lats,region_mask,11,transform=ccrs.PlateCarree(),
                    colors='k',
                    vmin=-1, vmax=1)
axs[1].set_title('Return period of current record')
axs[1].text(-190, 95, 'b')
cbar = plt.colorbar(d, shrink=0.7, ax=axs[1])
cbar.set_label('Years')
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=6,width=0.5)

for axes in axs:
    axes.coastlines()
    axes.set_ylim([-60, 90])
    axes.outline_patch.set_linewidth(1)

plt.tight_layout()
#plt.savefig('GEV_maps1_160222.png', dpi=300)
#plt.close()
