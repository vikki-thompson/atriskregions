'''
Created 24 March 2022
Editted 28 Feb 2023
GEV for MODEL Stone regions

Figures for at-risk paper, using models

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

print('starting script')

### Generic functions
def time_slice(cube, year1, year2):
    year_cons = iris.Constraint(time=lambda cell: year1 <= cell.point.year <= year2)
    return cube.extract(year_cons)

def cube_to_array(cube):
    return cube.data.reshape(np.shape(cube.data)[0]*np.shape(cube.data)[1])

def model_ann_max(reg, model):
    #Load model data, model = canesm or miroc (str)
    filename = '/user/work/hh21501/'+model+'_regions/reg'+str(reg)+'_*'
    x= iris.load(filename)
    top = []
    for each in x:
        icc.add_year(each, 'time')
        top.append(each.aggregated_by(('year'), iris.analysis.MAX)-273.15)
    if model=='canesm':
        if reg < 208:
            ann_max = iris.cube.CubeList([top[0], top[2]]).concatenate()[0]
        if reg > 207:
            ann_max = iris.cube.CubeList([top[0], top[1]]).concatenate()[0]
    elif model == 'miroc':
        if reg < 208:
            ann_max = iris.cube.CubeList([top[0], top[1], top[2], top[3], top[4], top[5], top[6]]).concatenate()[0]
        else:
            ann_max = iris.cube.CubeList([top[0], top[1], top[2], top[3], top[4], top[5], top[6], top[7], top[8], top[9], top[10], top[11], top[12], top[13], top[14], top[15]]).concatenate()[0][:, :65]
    return ann_max

def remove_max(ann_max, GMST):
    ' Remove the max value from region, and corresponding GMST value '
    ann_max_new = np.delete(ann_max, np.abs(ann_max - np.max(ann_max)).argmin())
    GMST_new = np.delete(GMST, np.abs(ann_max - np.max(ann_max)).argmin())
    return ann_max_new, GMST_new

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
    axs_sel.plot(ret_per, obs_sort, '+', color='dodgerblue', alpha=.5)
    return

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
    axs_sel.plot(chance2[10:], ret_lev[10:-1], color='dodgerblue', linewidth=1)
    return

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
    axs_sel.plot(chanceyear_5[10:], ret_lev[10:-1], '--')
    axs_sel.plot(chanceyear_95[10:], ret_lev[10:-1], '--')
    return

def remove_max_mod(ann_max, GMST):
    ' Remove the max value from region, and corresponding GMST value '
    ann_max_new = []
    GMST_new = []
    for ens in np.arange(np.shape(ann_max)[0]):
        ann_max_new.append(np.delete(ann_max[ens], np.abs(ann_max[ens] - np.max(ann_max[ens])).argmin()))
        GMST_new.append(np.delete(GMST, np.abs(ann_max[ens] - np.max(ann_max[ens])).argmin()))
    return ann_max_new, GMST_new


### Global map functions
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

## Global mean surface temp timeseries
def load_CanESM5_GMST(y1, y2):
    '''
    Returns cube of tas GMST
    '''
    file_list = []
    for filepath in glob.iglob('/bp1store/geog-tropical/data/CMIP6/CMIP/CCCma/CanESM5/historical/*/day/tas/gn/latest/*'):
        file_list.append(filepath)
    cube_list = iris.cube.CubeList([])
    for count, each in enumerate(file_list):
        print(count, each)
        cube = iris.load_cube(each)  # dim = 31390x64x128
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.collapsed(('longitude', 'latitude'), iris.analysis.MEAN)
        cube = cube.aggregated_by('year', iris.analysis.MEAN)
        realization_coord = iris.coords.AuxCoord(count, 'realization')
        cube.add_aux_coord(realization_coord)
        cube_list.append(cube)
        print(cube.shape)
    equalise_attributes(cube_list) # to allow merge to one cube 
    x = cube_list.merge_cube()
    GMST = x.collapsed('realization', iris.analysis.MEAN)
    GMST = GMST.extract(iris.Constraint(time=lambda cell: y1 <= cell.point.year <= y2))
    return GMST


## CanESM5 GMST - ensemble mean, daily tas, 1950-2014 (historical CMIP6)
## Relative to 2014 (hottest) 
GMST = load_CanESM5_GMST(1950, 2014).data
GMST_yr = np.arange(1950, 2014, 1)


def calc_fit(data, GMST):
    # data & GMST must be same shape
    A, B = np.shape(data)
    offset = []
    model = np.reshape(data, [A*B])
    gmst = np.reshape(GMST, [A*B])
    a, b = np.polyfit(gmst, model, 1)
    return a, b

def detrend(data, gm, a, b):
    A, B = np.shape(data)
    offset_data = []
    actual = data
    predicted = np.reshape(gm*a+b, [50, 65])
    offset = actual-predicted+(1*a+b)
    return offset

def plot_EVT(axs_sel, data_array):
    axs_sel.set_xscale('log')    
    plot_points(axs_sel, data_array)
    plot_gev(axs_sel, data_array, np.min(data_array)-.5, np.max(data_array)+2)
    axs_sel.set_ylabel('maxT')
    axs_sel.set_xlabel('Return period, yr')
    axs_sel.set_xlim([1, 100000])
    return

def return_period(GEV_data, extreme):
    shape, loc, scale = gev.fit(GEV_data) # EVT distribution
    x_val = np.linspace(np.min(GEV_data)-.5, np.max(GEV_data)+2, 1000)
    dist_pdf = gev.pdf(x_val, shape, loc, scale)
    chance = []
    for i, _ in enumerate(dist_pdf):
        P = []
        for a, b in zip(dist_pdf[i:-1], dist_pdf[i+1:]):
            P.append(((a+b) / 2) * (x_val[1] - x_val[0])) 
        chance.append(sum(P)*100)
    x = chance[np.abs(x_val - extreme).argmin()] # chance of observed max
    if x == 0:
        result = np.nan
    else:
        result = 100.*1/x
    return result

plt.ion()
plt.show()

## Fig.3 global maps

impossible_count = [] # each ensemble member - how many impossible
ret_per_full = []   # full ensemble return period of record
for reg in range(237):
    ann_max = model_ann_max(reg, 'canesm')
    a, b = calc_fit(ann_max.data, np.tile(GMST, 50)) # Trend
    offset = detrend(ann_max.data, np.tile(GMST, 50), a, b)
    ann_max_new_detrend, GMST_new_detrend = remove_max_mod(offset, GMST)
    # data to fit GEV
    model_GEV = ann_max_new_detrend
    model_extremes = np.max(offset, axis=1)
    ret_per = []
    for ens in np.arange(50):
        ret_per.append(return_period(model_GEV[ens], model_extremes[ens]))
    impossible_count.append(sum(np.isnan(x) for x in ret_per))
    # full ensemble
    model_GEV = np.sort(np.reshape(ann_max.data, [65*50,]))[:-1]
    model_extreme = np.max(np.reshape(ann_max.data, [65*50,]))
    ret_per_full.append(return_period(model_GEV, model_extreme))
    print(reg, impossible_count[-1], ret_per_full[-1])

# adjust to percentage 
impossible_count = [x*2 for x in impossible_count]

# Transfer vals to regional data
region_fp = 'region_fx-WRAF05-v4-1_WRAF_All-Hist_est1_v4-1_4-1-0_000000-000000.nc'
region_ds = xr.open_mfdataset(region_fp, parallel=True)
lats = region_ds.lat
lons = region_ds.lon
region_count = region_ds.region
region_full = region_ds.region
for region in range(237):
    print('region',region)
    region_count = region_count.where(region_ds.region.values != region, impossible_count[region])
    region_full = region_full.where(region_ds.region.values != region, ret_per_full[region])

fig, axs = plt.subplots(4, 1, figsize=(10., 16.), dpi=80, num=None, subplot_kw={'projection': ccrs.PlateCarree()})
# subplot 1: CanESM impossible count
levels=[0, 10, 20, 30, 40, 50, 60]
c = axs[0].contourf(lons,lats,region_count,9,transform=ccrs.PlateCarree(),
                    cmap=mpl_cm.get_cmap('brewer_PuRd_09'), linestyle='solid',
                    levels=levels)#vmin=0, vmax=60)
cbar = plt.colorbar(c, shrink=0.7, ax=axs[0])
cbar.set_label('% of ensemble members')
cbar.outline.set_linewidth(0.5)
#cbar.axs[0].tick_params(labelsize=6,width=0.5)
axs[0].set_title('CanESM5: Exceptional ensembles')
axs[0].text(-190, 80, 'a')
axs[0].coastlines()
axs[0].set_ylim([-60, 90])
axs[0].outline_patch.set_linewidth(1)

# subplot 2: CanESM Full ensemble return periods
from matplotlib import ticker, cm
e = axs[1].contourf(lons,lats,region_count,11,transform=ccrs.PlateCarree(),
                    colors='r',
                    vmin=-10, vmax=1000) # might need adjusting
levels=np.logspace(np.log10(10),np.log10(1000000000),5)
d = axs[1].contourf(lons,lats,region_full,6,transform=ccrs.PlateCarree(),
                    levels=levels, locator=ticker.LogLocator(base=10),
                    cmap=mpl_cm.get_cmap('Blues'), linestyle='solid',
                    extend='max')
cbar = plt.colorbar(d, shrink=0.7, ax = axs[1], ticks=levels)
cbar.set_label('Years')
cbar.outline.set_linewidth(0.5)
#cbar.axs[1].tick_params(labelsize=6,width=0.5)
axs[1].set_title('CanESM5: Return periods')
axs[1].text(-190, 80, 'b')
axs[1].coastlines()
axs[1].set_ylim([-60, 90])
axs[1].outline_patch.set_linewidth(1)
axs[1].text(-170, -50, 'Exceptional', fontsize = 10, 
         bbox = dict(facecolor = 'red', alpha = 0.5))

# Delete the Antarctic regions
regs_in_ant = np.arange(170, 191)
for each in regs_in_ant:
    ret_per_full.pop(each)
    impossible_count.pop(each)

# Percentages
imp_canesm = np.mean(impossible_count) 
full_canesm=sum(np.isnan(x) for x in ret_per_full)/217*100

## MIROC
impossible_count = [] # each ensemble member - how many impossible
ret_per_full = []   # full ensemble return period of record
for reg in range(237):
    ann_max = model_ann_max(reg, 'miroc')
    a, b = calc_fit(ann_max.data, np.tile(GMST, 50)) # Trend
    offset = detrend(ann_max.data, np.tile(GMST, 50), a, b)
    ann_max_new_detrend, GMST_new_detrend = remove_max_mod(offset, GMST)
    # data to fit GEV
    model_GEV = ann_max_new_detrend
    model_extremes = np.max(offset, axis=1)
    ret_per = []
    for ens in np.arange(50):
        ret_per.append(return_period(model_GEV[ens], model_extremes[ens]))
    impossible_count.append(sum(np.isnan(x) for x in ret_per))
    # full ensemble
    model_GEV = np.sort(np.reshape(ann_max.data, [65*50,]))[:-1]
    model_extreme = np.max(np.reshape(ann_max.data, [65*50,]))
    ret_per_full.append(return_period(model_GEV, model_extreme))
    print(reg, impossible_count[-1], ret_per_full[-1])

# adjust to percentage 
impossible_count = [x*2 for x in impossible_count]

# Transfer vals to regional data
region_fp = 'region_fx-WRAF05-v4-1_WRAF_All-Hist_est1_v4-1_4-1-0_000000-000000.nc'
region_ds = xr.open_mfdataset(region_fp, parallel=True)
lats = region_ds.lat
lons = region_ds.lon
region_count = region_ds.region
region_full = region_ds.region
for region in range(237):
    print('region',region)
    region_count = region_count.where(region_ds.region.values != region, impossible_count[region])
    region_full = region_full.where(region_ds.region.values != region, ret_per_full[region])

# subplot 3: MIROC impossible count
levels=[0, 10, 20, 30, 40, 50, 60]
c = axs[2].contourf(lons,lats,region_count,9,transform=ccrs.PlateCarree(),
                    cmap=mpl_cm.get_cmap('brewer_PuRd_09'), linestyle='solid',
                    levels=levels)#vmin=0, vmax=60)
cbar = plt.colorbar(c, shrink=0.7, ax=axs[2])
cbar.set_label('% of ensemble members')
cbar.outline.set_linewidth(0.5)
#cbar.axs[2].tick_params(labelsize=6,width=0.5)
axs[2].set_title('MIROC6: Exceptional ensembles')
axs[2].text(-190, 80, 'c')
axs[2].coastlines()
axs[2].set_ylim([-60, 90])
axs[2].outline_patch.set_linewidth(1)

# subplot 4: MIROC Full ensemble return periods
from matplotlib import ticker, cm
e = axs[3].contourf(lons,lats,region_count,11,transform=ccrs.PlateCarree(),
                    colors='r',
                    vmin=-10, vmax=1000) # might need adjusting
levels=np.logspace(np.log10(10),np.log10(1000000000),5)
d = axs[3].contourf(lons,lats,region_full,6,transform=ccrs.PlateCarree(),
                    levels=levels, locator=ticker.LogLocator(base=10),
                    cmap=mpl_cm.get_cmap('Blues'), linestyle='solid',
                    extend='max')
cbar = plt.colorbar(d, shrink=0.7, ax = axs[3], ticks=levels)
cbar.set_label('Years')
cbar.outline.set_linewidth(0.5)
#cbar.axs[3].tick_params(labelsize=6,width=0.5)
axs[3].set_title('MIROC6: Return periods')
axs[3].text(-190, 80, 'd')
axs[3].coastlines()
axs[3].set_ylim([-60, 90])
axs[3].outline_patch.set_linewidth(1)
axs[3].text(-170, -50, 'Exceptional', fontsize = 10, 
         bbox = dict(facecolor = 'red', alpha = 0.5))

# Delete the Antarctic regions
regs_in_ant = np.arange(170, 191)
for each in regs_in_ant:
    ret_per_full.pop(each)
    impossible_count.pop(each)

# Percentages
imp_miroc = np.mean(impossible_count) 
full_miroc=sum(np.isnan(x) for x in ret_per_full)/217*100

print('CANESM How many individual member fits are implausible, (%): ', str(imp_canesm))
print('CANESM How many regions are implausible in full ensemble, %: ', str(full_canesm))
print('MIROC How many individual member fits are implausible, (%): ', str(imp_miroc))
print('MIROC How many regions are implausible in full ensemble, %: ', str(full_miroc))

plt.tight_layout()
plt.savefig('fig4_final.png', dpi=300)
plt.savefig('fig4_final.pdf')
#plt.close()
