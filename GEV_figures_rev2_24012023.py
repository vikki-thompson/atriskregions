'''
Created 14 Dec 2021
Editted 28 Feb 2023
GEV for ERA5 Stone regions
Final figures for at-risk paper

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
from scipy.stats import genextreme as gev
import random
import scipy.io
import xarray as xr
import netCDF4 as nc
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
#import seaborn as sns


### Generic functions
def time_slice(cube, year1, year2):
    year_cons = iris.Constraint(time=lambda cell: year1 <= cell.point.year <= year2)
    return cube.extract(year_cons)

def cube_to_array(cube):
    return cube.data.reshape(np.shape(cube.data)[0]*np.shape(cube.data)[1])

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

def plot_points(axs_sel, data_array):
    obs_sort = np.sort(data_array)
    chance = []
    for i in range(len(obs_sort)):
        chance.append(((i+1) / len(obs_sort)) * 100)
    ret_per = []
    for each in chance:
        ret_per.append(100.*1/each)
    ret_per.reverse()
    axs_sel.plot(ret_per, obs_sort, '+b', alpha=.5, label='Observed TXx')

def plot_gev(axs_sel, data_array, min_val, max_val):
    shape, loc, scale = gev.fit(data_array)
    print('shape: ',str(shape))
    print('loc: ',str(loc))
    print('scale: ',str(scale))
    x_val = np.linspace(min_val, max_val, 1000)
    dist_pdf = gev.pdf(x_val, shape, loc, scale)
    ret_lev, chance = return_levels_plot(dist_pdf, x_val)
    chance2 = []
    for each in chance[:-1]:
        chance2.append(100.*1/each)
    axs_sel.plot(chance2[10:], ret_lev[10:-1], color='dodgerblue', label='GEV fit')
    
def bootstrap_uncertainties(axs_sel, data_array):
    # measure of uncertainty in distribution
    # 100 bootstraps, plot 5th-95th% range
    chance_list = []
    for i in np.arange(100):
        bootstrap_data = random.choices(data_array, k=len(data_array)) # allows double dipping
        shape, loc, scale = gev.fit(bootstrap_data)
        x_val = np.linspace(np.min(data_array)-.5, np.max(data_array)+2, 1000)
        dist_pdf = gev.pdf(x_val, shape, loc, scale)
        ret_lev, chance = return_levels_plot(dist_pdf, x_val)
        chance_list.append(chance)
    chance_5 = []
    chance_95 = []
    for i in np.arange(1000):
        new_val = []
        for each in chance_list:
            new_val.append(each[i])
        chance_5.append(np.sort(new_val)[5])
        chance_95.append(np.sort(new_val)[95])
    chanceyear_5 = []
    chanceyear_95 = []
    for i, each in enumerate(chance_5[:-1]):
        chanceyear_5.append(100.*1/each)
        chanceyear_95.append(100.*1/chance_95[i])
    axs_sel.plot(chanceyear_5[10:], ret_lev[10:-1], '--', color='dodgerblue')
    axs_sel.plot(chanceyear_95[10:], ret_lev[10:-1], '--', color='dodgerblue', label='Uncertainty (5th-95th%)')

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


                    



# 1970-2021 global mean surface temp (nasa giss)


GMST = [0.03, -0.03, 0.06, 0.03, 0.05, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05, 0.03, -0.08, 0.01, 0.16, -0.07, -0.01, -0.10, 0.18, 0.07, 0.17, 0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27, 0.45, 0.41, 0.22, 0.23, 0.32, 0.45, 0.33, 0.46, 0.61, 0.38, 0.39, 0.54, 0.63, 0.62, 0.54, 0.68, 0.64, 0.67, 0.54, 0.66, 0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.02, 0.92, 0.85, 0.98, 1.02, 0.85]
GMST_yr = np.arange(1959, 2022, 1)


############
plt.ion()
plt.show()

# Antarctica regions: 
regs_in_ant = np.arange(170, 191)

def plot_data_GMST(axs_sel, ann_max, GMST):
    axs_sel.plot(GMST, ann_max, 'b+')
    a, b = np.polyfit(GMST, ann_max, 1)
    bestfit = []
    for each in np.sort(GMST): bestfit.append(a*each+b)
    axs_sel.plot([np.min(GMST), np.max(GMST)], [np.min(bestfit), np.max(bestfit)], color='dodgerblue')
    #axs_sel.set_ylabel('TXx')
    #axs_sel.set_xlabel('GMST')

def adjust_obs(ann_max, GMST):
    offset = []
    a, b = np.polyfit(GMST, ann_max, 1)
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b)) #1*a+b means adjusted for world with 1deg warming  
    return offset

def plot_EVT(axs_sel, data_array):
    axs_sel.set_xscale('log')    
    plot_points(axs_sel, data_array)
    plot_gev(axs_sel, data_array, np.min(data_array)-.5, np.max(data_array)+2)

def load_annmax(reg):
    return np.load('/user/work/hh21501/era5_regions/ann_list_19592021.npy')[:,reg]-273.15

def remove_max(ann_max, GMST):
    offset = adjust_obs(ann_max, GMST)
    ' Remove the max value from region, and corresponding GMST value '
    idx_max = np.where(offset == np.max(offset))[0][0]
    ann_max_new = np.delete(ann_max, idx_max)
    GMST_new = np.delete(GMST, idx_max)
    return ann_max_new, GMST_new

def one_in_hundred(data_array):
    shape, loc, scale = gev.fit(data_array)
    x_val = np.linspace(np.min(data_array)-.5, np.max(data_array)+2, 1000)
    dist_pdf = gev.pdf(x_val, shape, loc, scale)
    ret_lev, chance = return_levels_plot(dist_pdf, x_val)
    return ret_lev[np.abs(np.asarray(chance)-1).argmin()]


def plot_data_GMST(axs_sel, ann_max, GMST):
    axs_sel.plot(GMST, ann_max, '+', label='Observed TXx')
    a, b = np.polyfit(GMST, ann_max, 1)
    bestfit = []
    for each in np.sort(GMST): bestfit.append(a*each+b)
    axs_sel.plot([np.min(GMST), np.max(GMST)], [np.min(bestfit), np.max(bestfit)], label='Fit')




'''
## For table
reg = [74, 59, 127, 211, 139, 36, 200, 44, 168, 149]
hund_list = []
max_list = []
for each in np.arange(237):
    hund_list.append(one_in_hundred(adjust_obs(load_annmax(each), GMST)))
    max_list.append(np.max(adjust_obs(load_annmax(each), GMST)))
'''


### Fig.1 
#
#
reg = 9 # Alberta, Canada
ann_max = load_annmax(reg)
offset = adjust_obs(ann_max, GMST)
ann_max_new, GMST_new = remove_max(ann_max, GMST)
#Prior to 2021 event 
offset_whatif = adjust_obs(ann_max_new, GMST_new)
ann_max_whatif, GMST_whatif = remove_max(ann_max_new, GMST_new)
 
fig, axs = plt.subplots(2, 2, figsize=(10., 7.), dpi=80, num=None)
axs[0,0].plot(GMST, ann_max, 'r+')
plot_data_GMST(axs[0,0], ann_max_new, GMST_new)
axs[0,0].set_ylabel('$\it{TXx}$, $^\circ$C')
axs[0,0].set_title('Record event in 2021', loc='right')
axs[0,0].set_ylim([26.5, 36.5])
axs[0,0].text(-.35, 37, 'a', weight='bold')

axs[1,0].plot(GMST_new, ann_max_new, 'r+', label='Record event')
plot_data_GMST(axs[1,0], ann_max_whatif, GMST_whatif)
axs[1,0].set_title('Record event in 1981', loc='right')
axs[1,0].set_ylim([26.5, 36.5])
axs[1,0].text(-.35, 37, 'c', weight='bold')
axs[1,0].set_ylabel('$\it{TXx}$, $^\circ$C')
axs[1,0].set_xlabel('GMST, $^\circ$C')
axs[1,0].legend(loc='upper left')

offset = adjust_obs(ann_max, GMST)
offset_new = adjust_obs(ann_max_new, GMST_new)
offset_whatif = adjust_obs(ann_max_whatif, GMST_whatif)
plot_EVT(axs[0,1], offset_new)
bootstrap_uncertainties(axs[0,1], offset_new)
axs[0,1].axhline(np.max(offset), color='r', label='Record event')
axs[0,1].set_ylim([26.5, 36.5])
axs[0,1].text(1, 37, 'b', weight='bold')

plot_EVT(axs[1,1], offset_whatif)
axs[1,1].set_xlabel('Return Period, Yr')
bootstrap_uncertainties(axs[1,1], offset_whatif)
axs[1,1].axhline(np.max(offset_new), color='r', label='Record event')
axs[1,1].set_ylim([26.6, 36.5])
axs[1,1].text(1, 37, 'd', weight='bold')
axs[1,1].legend(loc='lower right')

for each in [axs[0,1], axs[1,1]]:
    each.set_xlim([.9, 10000])
    each.set_xticks([2, 5, 10, 50, 100, 1000, 10000])
    each.set_xticklabels([2, 5, 10, 50, 100, 1000, 10000])

plt.tight_layout()
plt.savefig('fig1_final.png', dpi=300)
plt.savefig('fig1_final.pdf')

#plt.close()







##### GLOBAL ######
def offset_higher(reg):
    ''' loads data, 
    applies offset to present day, 
    calcs difference between observed max and 1-in-10000 event'''
    ann_max = load_annmax(reg)
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
    ann_max = load_annmax(reg)
    offset = []
    a, b = np.polyfit(GMST, ann_max, 1)
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
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
    ann_max = load_annmax(reg)
    ann_max_new, GMST_new = remove_max(ann_max, GMST)
    offset = []
    a, b = np.polyfit(GMST_new, ann_max_new, 1)
    for i, each in enumerate(ann_max_new):
        actual = each
        predicted = GMST_new[i]*a +b
        offset.append(actual-predicted+(1*a+b))
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
    ann_max = load_annmax(reg)
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
    ann_max = load_annmax(reg)
    offset = []
    a, b = np.polyfit(GMST, ann_max, 1)
    for i, each in enumerate(ann_max):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
    max_val = np.max(offset) # Maximum observed
    return max_val

def abs_max(reg):
    ann_max = load_annmax(reg)
    return np.max(ann_max)

def offset_100(reg):
    ann_max = load_annmax(reg)
    ann_max_new, GMST_new = remove_max(ann_max, GMST)
    offset = []
    a, b = np.polyfit(GMST_new, ann_max_new, 1)
    for i, each in enumerate(ann_max_new):
        actual = each
        predicted = GMST[i]*a +b
        offset.append(actual-predicted+(1*a+b))  # np.mean to make 'present day'
    shape, loc, scale = gev.fit(offset) # EVT distribution
    x_val = np.linspace(np.min(offset)-.5, np.max(offset)+2, 1000)
    dist_pdf = gev.pdf(x_val, shape, loc, scale)
    ret_lev, chance = return_levels_plot(dist_pdf, x_val)
    chance2 = []
    for each in chance[:-1]:
        chance2.append(100.*1/each)
    return ret_lev[(np.abs(np.asarray(chance2)-100)).argmin()]


## Fig 2
#
#
vals_retper_without = []
vals_abs_without = []
for each in np.arange(237):
    print(each)
    vals_retper_without.append(obs_max_ret_per_removeevent(each))
    vals_abs_without.append(offset_higher_removeevent(each))

vals_retper_without = [np.nan if x == 99999 else x for x in vals_retper_without]
# Transfer vals to regional data
region_fp = 'region_fx-WRAF05-v4-1_WRAF_All-Hist_est1_v4-1_4-1-0_000000-000000.nc'
region_ds = xr.open_mfdataset(region_fp, parallel=True)
lats = region_ds.lat
lons = region_ds.lon
region_abs = region_ds.region
region_retper = region_ds.region
for region in range(237):
    print('region',region)
    region_abs = region_abs.where(region_ds.region.values != region, vals_abs_without[region])
    region_retper = region_retper.where(region_ds.region.values != region, vals_retper_without[region])

# mask from 1990, comparison data from excel file
reg_mask = [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
reg_mask = [np.nan if x == 1 else x for x in reg_mask] 
region_mask = region_ds.region
for region in range(237):
    print('region',region)
    region_mask = region_mask.where(region_ds.region.values != region, reg_mask[region])

# figure
fig, axs = plt.subplots(2, 1, figsize=(10., 7.), dpi=80, num=None, subplot_kw={'projection': ccrs.PlateCarree()})
# Statistical maximum difference plot
c = axs[0].contourf(lons,lats,region_abs,11,transform=ccrs.PlateCarree(),
                    cmap=mpl_cm.get_cmap('bwr'), linestyle='solid',
                    vmin=-3.5, vmax=3.5) # or try 'bwr''coolwarm'
cbar = plt.colorbar(c, shrink=0.7, ax=axs[0])
b = axs[0].contourf(lons,lats,region_mask,11,transform=ccrs.PlateCarree(), colors='darkgrey', vmin=-1, vmax=1)
cbar.set_label(u"\u2103", fontsize=12)
cbar.outline.set_linewidth(0.5)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), ha='right')
cbar.ax.tick_params(labelsize=12,pad=40)
axs[0].set_title('Statistical maximum minus current record')
axs[0].text(-190, 95, 'a')
# Current return period plot
e = axs[1].contourf(lons,lats,region_abs,11,transform=ccrs.PlateCarree(),
                    colors='r',
                    vmin=-10, vmax=30)
levels=np.logspace(np.log10(1),np.log10(10000),5)
d = axs[1].contourf(lons,lats,region_retper,11,transform=ccrs.PlateCarree(),
                    cmap=mpl_cm.get_cmap('Blues'),
                    levels=levels, locator=ticker.LogLocator(base=10))
b = axs[1].contourf(lons,lats,region_mask,11,transform=ccrs.PlateCarree(), colors='darkgrey', vmin=-1, vmax=1)

axs[1].set_title('Current record return period')
axs[1].text(-190, 95, 'b')
cbar = plt.colorbar(d, shrink=0.7, ax = axs[1], ticks=ticker.LogLocator(base=10))
cbar.set_label('Years', fontsize=12)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=12,width=0.5)

axs[1].text(-170, -35, 'Exceptional', fontsize = 10, 
         bbox = dict(facecolor = 'red', alpha = 0.5))
axs[1].text(-170, -50, 'Inconsistent data', fontsize = 10, 
         bbox = dict(facecolor = 'darkgrey', alpha = 0.5))

for axes in axs:
    axes.coastlines()
    axes.set_ylim([-60, 90])
    axes.outline_patch.set_linewidth(1)

plt.tight_layout()
plt.savefig('fig2_final.png', dpi=300)
plt.savefig('fig2_final.pdf')
#plt.close()




## Fig 3
#
#	
reg_retper = vals_retper_without.copy()
reg_abs = vals_abs_without.copy()
# Delete the Antarctic regions
for each in regs_in_ant[::-1]:
    reg_retper.pop(each)
    reg_abs.pop(each)
    reg_mask.pop(each)

# apply reg_mask
retper_mask = []
abs_mask = []
for i, each in enumerate(reg_mask):
    if np.isnan(each):
        retper_mask.append(reg_retper[i])
        abs_mask.append(reg_abs[i])
    else: pass

## Scatter of two measures - where nan
def obs_max_year(reg):
    ' Returns the highest event in terms of return period, yrs '
    ann_max = load_annmax(reg)
    max = np.where(ann_max == np.max(ann_max))[0][0]
    return max+1959 # index -> year

vals_wherenan = []
year_wherenan = []
for i, each in enumerate(reg_retper):
    print(i)
    if np.isnan(reg_mask[i]):
        if np.isnan(each):
            vals_wherenan.append(reg_abs[i])
            year_wherenan.append(obs_max_year(i))

## pop data, from excel sheet
pop2020 = [35058, 9362.0, 36778.0, 4244.0, 7085, 155.0, 7784.0, 123699, 3768962, 3838695, 1275069, 1463330.0, 280330.0, 14180861.0, 39153.0, 7285067, 2248624, 6525, 573490.0, 11999838, 2697617, 8547714, 24082918.0, 11842203, 39273866.0, 40107493, 8384732, 24665718, 12264077, 63378369.0, 35493739, 33887421, 1776601, 443288, 4961768.0, 1375779, 6042843, 31150375, 11773314, 619124, 1984691, 10275139, 39445384, 80088461, 4127100, 4561119, 2973924, 30302763, 495720, 6688702, 29372864, 1777633, 32093533, 173174, 3626562, 35863592, 686558, 185412, 1058910, 1833886, 5124811, 9067462, 4789973, 28279744, 86327318, 1185522, 75910653, 13776194, 14810497, 5318217, 20417405, 29934246, 7901607, 32427030, 2057012, 17175842, 25703520, 64430782, 498206, 19633883, 73703476.0, 114829191, 228727, 13651192.0, 22070496.0, 4668836, 6345789, 17137553, 17125552.0, 28038064, 18433425.0, 11323533, 12411156, 35074065.0, 35005949.0, 17973787.0, 25106274, 25749291.0, 2580356, 2124047, 12522729, 39674125, 3300780, 884918, 40451661, 22175302, 23963980, 19123720, 302242.0, 1516770, 4522402, 9248667, 182013.0, 513, 3062310, 1289081.0, 2773401.0, 1054825, 281, 114.0, 104012, 9724, 293647, 132290, 17011, 356977, 2011401, 1432453, 74370541, 5351986, 27571533.0, 53411661, 14602262, 12217481, 1233655, 4359458, 3241555.0, 3239493.0, 5006886.0, 38755552, 197725186, 10665078, 7519179, 1034334, 31269113, 4808374, 11947485, 13950923, 67500374, 250297823.0, 704816, 1048482, 115213313, 75085565, 176968847, 267801195, 142034915, 144861962, 21875, 46092, 601818, 48198, 6191, 1245302, 7596027, 42121, 49716, 316674, 395350, 2590528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 47199454, 2288772, 22792175, 1392269, 3761272, 7087663, 75580294.0, 9126748, 10695544, 110276787.0, 65017756, 51612076.0, 64729722, 68988873, 34947609, 153658397, 427316337, 173377780, 258059259, 279816308, 44303465, 110458041, 69831788, 114120493, 203805030, 12323013, 17031880, 2039.0, 0.0, 872.0, 8044559.0, 16474411, 73425997, 21716337, 10139072, 84906918, 14565540, 91121993.0, 600857, 2097275, 349775, 45466417.0, 17321153.0, 41271482, 153926924, 7491016]

for each in regs_in_ant[::-1]:
    pop2020.pop(each)

pop_mask = []
for i, each in enumerate(reg_mask):
    if np.isnan(each):
        pop_mask.append(pop2020[i])
    else: pass

fig, axs = plt.subplots(1, 2, figsize=(10., 4.), dpi=80)
axs[0].plot(retper_mask, pop_mask, '+', label='Regional records')
axs[0].plot(166.4, pop2020[9], 'rx', label='Alberta record prior to 2021') # reg9 if 2021 hadn't happened
axs[0].set_xlabel('Return Period of current record (Years)', fontsize=12)
axs[0].set_ylabel('Population in 2020', fontsize=12)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid()
#axs[0].plt.arrow(10**6, 10**3, 10**5, 10**
axs[0].text(20, 8E8, 'a')
axs[0].legend(loc='lower right')
for i in range(len(vals_wherenan)): vals_wherenan[i] = -vals_wherenan[i]

axs[1].plot(year_wherenan, vals_wherenan, '+', label='Regional records')
axs[1].set_xlabel('Year', fontsize=12)
axs[1].set_ylabel('How far beyond statistical maximum, '+u"\u2103", fontsize=12)
axs[1].grid()
axs[1].text(1954, 4.7, 'b')
plt.tight_layout()
plt.savefig('fig3_final.png', dpi=300)
plt.savefig('fig3_final.pdf')
#plt.close()







## FigS1
#
#
a = []; b=[]; sc=[] #shape, loc, scale arrays
for each in np.arange(237):
    print(each)
    ann_max = load_annmax(each)
    shape, loc, scale = gev.fit(ann_max)
    a.append(shape)
    b.append(loc)
    sc.append(scale)

#sns.set_style("white")
region_fp = 'region_fx-WRAF05-v4-1_WRAF_All-Hist_est1_v4-1_4-1-0_000000-000000.nc'
region_ds = xr.open_mfdataset(region_fp, parallel=True)
lats = region_ds.lat
lons = region_ds.lon
shape_val = region_ds.region
loc_val = region_ds.region
scale_val = region_ds.region

for region in range(237):
    print('region',region)
    shape_val = shape_val.where(region_ds.region.values != region, a[region])
    loc_val = loc_val.where(region_ds.region.values != region, b[region])
    scale_val = scale_val.where(region_ds.region.values != region, sc[region])

fig, axs = plt.subplots(3, 1, figsize=(7., 10.), dpi=80, num=None, subplot_kw={'projection': ccrs.PlateCarree()})
levels = np.linspace(0.0, 0.6, 20)
c = axs[0].contourf(lons,lats,shape_val,levels = levels,transform=ccrs.PlateCarree(),extend='min')
cbar = plt.colorbar(c, shrink=0.54, ax = axs[0], ticks=[0, 0.2, 0.4])
cbar.set_label(u"\u2103", fontsize=12)
cbar.outline.set_linewidth(0.5)
cbar.axs[0].tick_params(labelsize=12,width=0.5)
axs[0].title.set_text('Shape')
axs[0].text(-190, 95, 'a')

levels = np.linspace(0.0, 50, 20)
c = axs[1].contourf(lons,lats,loc_val,levels = levels,transform=ccrs.PlateCarree(),extend='min')
cbar = plt.colorbar(c, shrink=0.54, ax = axs[1], ticks=[0, 20, 40])
cbar.set_label(u"\u2103", fontsize=12)
cbar.outline.set_linewidth(0.5)
cbar.axs[1].tick_params(labelsize=12,width=0.5)
axs[1].title.set_text('Location')
axs[1].text(-190, 95, 'b')

levels = np.linspace(0.0, 2.5, 20)
c = axs[2].contourf(lons,lats,scale_val,levels = levels,transform=ccrs.PlateCarree(),extend='min')
cbar = plt.colorbar(c, shrink=0.54, ax = axs[2], ticks=[0, 1, 2])
cbar.set_label(u"\u2103", fontsize=12)
cbar.outline.set_linewidth(0.5)
cbar.axs[2].tick_params(labelsize=12,width=0.5)
axs[2].title.set_text('Scale')
axs[2].text(-190, 95, 'c')

for axes in axs:
    axes.coastlines()
    axes.set_ylim([-60, 90])
    axes.outline_patch.set_linewidth(1)

plt.tight_layout()
plt.savefig('gev_Sfig1.png', dpi=300)





