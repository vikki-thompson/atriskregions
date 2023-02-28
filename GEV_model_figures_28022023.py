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
#GMST = load_CanESM5_GMST(1950, 2014).data
GMST = [-0.5527399999999716, -0.5740699999999492, -0.5913400000000024, -0.619419999999991, -0.6150199999999586, -0.5882199999999784, -0.6106899999999769, -0.5823999999999501, -0.5880499999999529, -0.5554199999999696, -0.5717499999999518, -0.5708699999999567, -0.5944199999999569, -0.6709599999999796, -0.7716699999999719, -0.8044799999999555, -0.8035600000000045, -0.8194599999999923, -0.7499699999999621, -0.7579700000000003, -0.7410600000000045, -0.6886899999999514, -0.6788699999999608, -0.6660200000000032, -0.6436199999999985, -0.6377899999999954, -0.6576000000000022, -0.5948799999999892, -0.5488199999999779, -0.5216199999999844, -0.47461999999995896, -0.4485199999999736, -0.4615199999999504, -0.5201199999999631, -0.4466199999999958, -0.40031999999996515, -0.3543399999999792, -0.31322000000000116, -0.2515599999999836, -0.1978199999999788, -0.13265999999998712, -0.10748999999998432, -0.2931199999999876, -0.2837599999999725, -0.2199399999999514, -0.10755000000000337, -0.04861999999997124, 0.005460000000027776, 0.12138000000004467, 0.20488000000000284, 0.2786500000000274, 0.3601300000000265, 0.4440800000000422, 0.5386799999999994, 0.5526399999999967, 0.5380800000000363, 0.5588300000000004, 0.5694500000000176, 0.6671100000000365, 0.7209400000000414, 0.7830500000000029, 0.8453000000000088, 0.8778599999999983, 0.9559000000000424, 1.0]
GMST_yr = np.arange(1950, 2014, 1)

'''
# 1950-2021 global mean surface temp (nasa giss)
GMST = [-0.17, -0.07, 0.01, 0.08, -0.13, -0.14, -0.19, 0.05, 0.06, 0.03, -0.03, 0.06, 0.03, 0.05, -0.20, -0.11, -0.06, -0.02, -0.08, 0.05, 0.03, -0.08, 0.01, 0.16, -0.07, -0.01, -0.10, 0.18, 0.07, 0.17, 0.26, 0.32, 0.14, 0.31, 0.16, 0.12, 0.18, 0.32, 0.39, 0.27, 0.45, 0.41, 0.22, 0.23, 0.32, 0.45, 0.33, 0.46, 0.61, 0.38, 0.39, 0.54, 0.63, 0.62, 0.54, 0.68, 0.64, 0.67, 0.54, 0.66, 0.72, 0.61, 0.65, 0.68, 0.75, 0.90, 1.02, 0.92, 0.85, 0.98, 1.02, 0.85]
GMST_yr = np.arange(1950, 2022, 1)
'''

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

'''
#
#
#
## Fig.1: Focus on region 9 (Alberta, Canada). GEV-fit
reg = 9
fig, axs = plt.subplots(2, 4, figsize=(15., 8.), dpi=80, num=None)

ann_max = model_ann_max(reg, 'canesm')
# subplot 1: T v GMST
a, b = calc_fit(ann_max.data, np.tile(GMST, 50)) # Trend
offset = detrend(ann_max.data, np.tile(GMST, 50), a, b)
ann_max_new_detrend, GMST_new_detrend = remove_max_mod(offset, GMST)
for ens in np.arange(50):
    axs[0,0].plot(GMST, offset[ens,:], 'r+')
    axs[0,0].plot(GMST_new_detrend[ens], ann_max_new_detrend[ens], '+', color='dodgerblue')

axs[0,0].axhline(np.max(offset), color='r')
axs[0,0].set_ylabel('TXx, $^\circ$C')
#axs[0,0].set_title('CanESM6', loc='right')
axs[0,0].text(-.9, 36.5, 'a')

# subplot 2: individual GEVs
model_GEV = ann_max_new_detrend
ret_per = []
for ens in np.arange(50):
    plot_EVT(axs[0,1], model_GEV[ens])
    axs[0,1].axhline(np.max(offset[ens]), alpha=.6, color='r')
    ret_per.append(return_period(model_GEV[ens], np.max(offset[ens])))

axs[0,1].set_ylabel('')
axs[0,1].set_xlabel('')
#axs[0,1].set_title('Add title?', loc='right')
axs[0,1].text(1, 36.5, 'b')

# subplot 3: full ensemble GEV as one
model_GEV = np.sort(np.reshape(offset, [65*50,]))[:-1]
model_extreme = np.max(model_GEV)
plot_EVT(axs[0,2], model_GEV)
ret_per_full = return_period(model_GEV, model_extreme)
axs[0,2].axhline(np.max(offset), color='r')
axs[0,2].set_ylabel('')
axs[0,2].set_xlabel('')
#axs[0,2].set_title('Add title?', loc='right')
axs[0,2].text(1, 36.5, 'c')
for each in [axs[0,0], axs[0,1], axs[0,2]]:
    each.set_ylim([21, 36])

# subplot 4: year of nan
ret_per_new = np.where(np.isnan(ret_per), 1, ret_per)
axs[0,3].plot(ret_per_new, np.arange(50), 'o', color='dodgerblue', label='Return period')
for i, each in enumerate(ret_per_new):
    if each == 1.0:
        axs[0,3].plot(each, i, 'ro')

axs[0,3].plot(each, 60, 'ro', label='Beyond fit')
axs[0,3].legend()
axs[0,3].set_xscale('log') 
axs[0,3].set_ylim([-1,50])
axs[0,3].text(0.8, 51, 'd')
axs[0,3].set_title('CanESM5', loc='right')

##
##
ann_max = model_ann_max(reg, 'miroc')
# subplot 5: T v GMST
a, b = calc_fit(ann_max.data, np.tile(GMST, 50)) # Trend
offset = detrend(ann_max.data, np.tile(GMST, 50), a, b)
ann_max_new_detrend, GMST_new_detrend = remove_max_mod(offset, GMST)
for ens in np.arange(50):
    axs[1,0].plot(GMST, offset[ens,:], 'r+')
    axs[1,0].plot(GMST_new_detrend[ens], ann_max_new_detrend[ens], '+', color='dodgerblue')

axs[1,0].axhline(np.max(offset), color='r')
axs[1,0].set_ylabel('TXx, $^\circ$C')
#axs[1,0].set_title('CanESM6', loc='right')
axs[1,0].text(-.9, 39, 'e')

# subplot 6: individual GEVs
model_GEV = ann_max_new_detrend
ret_per = []
for ens in np.arange(50):
    plot_EVT(axs[1,1], model_GEV[ens])
    axs[1,1].axhline(np.max(offset[ens]), alpha=.6, color='r')
    ret_per.append(return_period(model_GEV[ens], np.max(offset[ens])))

axs[1,1].set_ylabel('')
#axs[0,1].set_title('Add title?', loc='right')
axs[1,1].text(1, 39, 'f')

# subplot 7: full ensemble GEV as one
model_GEV = np.sort(np.reshape(offset, [65*50,]))[:-1]
model_extreme = np.max(model_GEV)
plot_EVT(axs[1,2], model_GEV)
ret_per_full = return_period(model_GEV, model_extreme)
axs[1,2].axhline(np.max(offset), color='r')
axs[1,2].set_ylabel('')
axs[1,2].text(1, 39, 'g')
for each in [axs[1,0], axs[1,1], axs[1,2]]:
    each.set_ylim([26, 38.5])

# subplot 8: year of nan
ret_per_new = np.where(np.isnan(ret_per), 1, ret_per)
axs[1,3].plot(ret_per_new, np.arange(50), 'o', color='dodgerblue', label='Return period')
for i, each in enumerate(ret_per_new):
    if each == 1.0:
        axs[1,3].plot(each, i, 'ro')

axs[1,3].plot(each, 60, 'ro', label='Beyond fit')
axs[1,3].legend()
axs[1,3].set_xscale('log') 
axs[1,3].set_ylim([-1,50])
axs[1,3].text(.8, 51, 'h')
axs[1,3].set_title('MIROC6', loc='right')


axs[1,0].set_xlabel('GMST, $^\circ$C')
axs[1,1].set_xlabel('Return period, yr')
axs[1,2].set_xlabel('Return period, yr')
axs[1,3].set_xlabel('Return period, yr')

plt.tight_layout()
plt.savefig('gev_Sfig2.png', dpi=300)
#plt.close()
'''





## Fig.3 global maps
#
#

'''
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
'''

impossible_count = [19,14,7,12,11,9,13,14,8,7,6,5,11,13,10,13,14,11,12,7,12,20,13,18,7,12,16,10,17,13,8,9,33,2,16,31,22,12,13,18,12,16,17,2,10,7,8,10,13,9,13,9,20,12,15,25,20,16,15,9,9,20,15,13,9,19,9,10,8,5,12,11,13,19,15,16,20,10,18,14,18,22,12,20,10,15,4,10,6,10,15,8,25,13,15,14,2,11,15,22,12,5,11,11,8,13,12,9,11,9,9,9,21,10,9,13,13,8,17,18,10,18,9,8,14,11,7,12,15,14,12,14,14,16,16,20,14,15,21,16,11,13,18,16,16,11,13,10,12,11,14,10,2,2,14,5,5,3,23,23,16,11,23,24,21,21,25,23,23,20,12,5,12,10,7,7,10,5,10,9,8,5,7,9,6,11,8,13,11,12,9,19,9,8,4,6,11,13,16,12,9,12,14,9,13,15,5,12,14,11,29,16,11,17,13,7,11,5,17,18,10,14,16,18,6,19,21,12,24,10,8,12,7,7,11,13,3]
ret_per_full = [9089427.044, 83602.61331, 12963.90029,51027.70875,136239.7645,16853548.84,3554.398274,569672.6972,459420.8394,1734.538107,24766.71087,8241.519037,np.nan,np.nan,97685.64141,35994.0824,1773479197,17567.49489,np.nan,3206.997853,30790.29752,676627.6165,6432.701421,68894.06674,7540.638728,17085.50113,np.nan,22661.17656,126059.2073,1636.099848,10937.71943,6039.855547,np.nan,390.2594265,1125.286119,122208.9038,211391.0011,1944.135448,np.nan,1723.397136,3705.76983,19033230.21,225830.1836,613.1714913,56992268.33,83348.17073,1115794.294,2852.745038,np.nan,6350.572607,19383295.84,8159.514357,2776.819431,10013.29711,10607.62836,4950.6658,4405.490732,12369.74787,111752.7768,12951.80093,95339.59915,532348604.1,194758.6716,24199.69302,4464.976986,np.nan,np.nan,3060.936802,430.7189464,984.7692853,3815.081173,1535.428769,1056925.617,11436.86055,12459.69481,16484.4686,np.nan,45290.6837,np.nan,16781206.38,np.nan,1762452174,22184.84273,518657.8011,207613.4473,130075.3667,1357.54412,8740.862876,37945.37369,2387.42454,126910.4656,4342.85014,297495.0691,np.nan,435844.4114,np.nan,5718.175504,141543.0719,33415.68997,np.nan,29844922,np.nan,6532.199212,54220.45205,5406.116573,10275.56141,30512.63654,np.nan,2935718.286,430774.1857,237239.1642,648522930.8,952530524.7,8268.291894,np.nan,127189.4457,12719.47524,10369.52957,np.nan,33641682.64,11725.34276,91766979.28,4020.820002,np.nan,213511.0022,19703.73171,5867.153584,2253472.536,55608.28382,3659.451517,114533.867,1770159.218,534671.3318,np.nan,504626.5608,311376.1251,545964.751,np.nan,501701.6029,np.nan,16918.69685,27433.70164,162383.405,np.nan,86684.51063,9998.895325,19676.89944,4832.04437,np.nan,np.nan,105.8927787,152.6493464,71278.0707,1143.10535,126542.1847,17619.07461,20503.32471,27688.70168,26202.45594,np.nan,277843320.2,np.nan,np.nan,np.nan,np.nan,322167.6956,7004884024,np.nan,3234409.392,np.nan,66267.51476,53656.79393,46836.80315,116632468.5,9604.997201,9520.611691,9337.788386,5941.354584,13520.40835,4030.639007,80494.26363,8713.340915,62817.23575,11554.17009,12609.3209,5367.750795,93506.58982,214061.0053,273744.3408,1569877.972,8710.634951,np.nan,1225.536122,781403.724,302.8763665,1154.231275,6831.375875,1382410.994,54447803.15,1287940.48,1940349.522,2092825.427,83149.17986,57100001.27,np.nan,51362269.09,3805.953592,259441.6075,44345.10397,5781677.851,8294179.079,1226987.455,97248.08406,np.nan,2703.113007,1190.464204,5542.476367,6620.001369,80849.23444,np.nan,106124.2205,np.nan,np.nan,np.nan,1493636.764,1652300.199,135879739.8,21951.06751,np.nan,7998.166062,403766.3992,259075.8098,45516.66854,114143.9629,2912.079336,5514.796655,1208478.683]	

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
'''
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
'''
impossible_count =[16, 10, 11, 11, 12, 8, 10, 12, 15, 10, 10, 11, 7, 11, 12, 8, 13, 17, 17, 9, 16, 19, 12, 16, 15, 10, 17, 21, 7, 10, 14, 13, 1, 2, 3, 8, 15, 17, 15, 8, 11, 13, 10, 8, 17, 17, 15, 14, 15, 21, 4, 3, 9, 13, 11, 18, 16, 10, 13, 7, 7, 14, 7, 12, 7, 6, 9, 4, 5, 8, 6, 25, 10, 4, 12, 14, 11, 8, 7, 13, 5, 5, 9, 6, 4, 4, 10, 3, 10, 9, 7, 11, 16, 19, 13, 15, 13, 12, 14, 16, 7, 9, 12, 9, 8, 10, 15, 12, 8, 14, 16, 22, 7, 21, 18, 12, 13, 14, 20, 11, 10, 16, 10, 6, 15, 10, 15, 14, 16, 5, 12, 14, 20, 12, 11, 15, 13, 17, 12, 27, 11, 13, 19, 11, 11, 7, 18, 11, 19, 9, 7, 12, 9, 12, 11, 20, 17, 14, 20, 22, 16, 16, 15, 22, 17, 17, 15, 19, 19, 18, 11, 9, 4, 12, 12, 9, 4, 3, 9, 10, 10, 6, 8, 9, 9, 5, 5, 12, 2, 7, 6, 3, 8, 9, 6, 9, 10, 14, 21, 19, 10, 15, 10, 15, 13, 17, 8, 15, 14, 9, 15, 11, 11, 9, 3, 10, 6, 0, 13, 24, 11, 20, 12, 17, 14, 10, 9, 14, 16, 11, 10, 5, 12, 8, 15, 14, 2] 
ret_per_full =[54911.12680288187, 694380.239184456, 2322798802.181781, np.nan, 23364.19332154876, 158385.33340444608, 191715957.45002073, 104178940558.09578, np.nan, 1602.7077762888366, 15709.164692083223, np.nan, 1717.5224581985028, 20644.878180803626, 55347.064681818614, 2908.258413313725, 220031066.05797842, 13632.123965974384, 121917.34889262386, np.nan, np.nan, 85199249.55089152, 7386970.513634085, 61492.09895243169, np.nan, 463724314.44601387, 226840.1343029534, 94946985.35415001, np.nan, 214726.89849001367, 69025.55107956739, np.nan, 515.918377130611, 2382.440857507047, 2993.93409633046, 1585.2508594452595, 95527455.6985496, 10026.550630919006, 32284.899254895685, 11464.675426746198, 3079.2747780819345, np.nan, 9122086.17461956, 350769.4925184015, 23554.06066185284, 1630052026.9393814, np.nan, np.nan, np.nan, 515206.0099359009, 1973.212630427956, 1720.0119968443114, 16232.48013876507, np.nan, 2082484.0919556436, 1729267.8089856368, 18474716.305307873, 3637.091096065267, 235767.89182364626, 64152.91673318444, 2853.6496051046925, np.nan, 11425.397332042336, 10905.273063408435, 16626.281698027055, 40342.402252491505, 1015674.6103662224, 2022.853503123874, 5171.04337252782, 18028.276727869175, 17278.225116234808, np.nan, 50601.258496403156, 6852.262007059147, 20481.003129658344, 132732.40779425434, 12644.82338325092, 6799.581671681913, 39170.79173414341, 50681536.96167703, 5447.85269071764, 21438.70496922653, 186162123097.3069, 66031.770049505, 3409.682048533616, 3095.6428129719525, 2756.3247329036494, 1347.3713086432174, 17696.95189238952, 72085.92422573268, 302628398724.0056, 53812.75959987526, 481399.8848844326, 153398406.471606, np.nan, np.nan, 7817.729259249949, 155191.70550328324, 16447.092688667806, np.nan, 25896.625317606547, 397585.1313924697, 437740.5474879096, 19729.30943864657, 20929.095674210064, 21449.354099803782, np.nan, np.nan, 12257.84316349255, 150038089.73474276, np.nan, 1538775.998067526, 8190.312962880719, np.nan, np.nan, 18292.575091641716, np.nan, 16488.17275590089, np.nan, 96570.12699201547, np.nan, 87947655.1386635, np.nan, 847566.3023205991, 4363684.674891321, np.nan, 1630743.654984014, 31851.934029342563, 1109356.034119246, np.nan, 1522307169.7333007, np.nan, 66475266.976574525, 222158.76799217545, 9014.62672801396, np.nan, 13914777.813969878, 37349395.80669164, np.nan, np.nan, 13433061011.4697, 81621226.45506248, 363958063898.47424, 153094.77361074733, 1106361.5554022056, 11921.435260569117, 8523376.20430681, 11666.810795625513, 1875658245.9459503, 194642226.21077523, 67857.91516679377, 14145.49322976765, 34967.5439957661, 1263982060.7399669, np.nan, np.nan, 53782.66263031957, 6566.961367623211, 550573.1712036921, 2892592.0793770733, np.nan, 541617.1712608326, np.nan, np.nan, 174990.16983157353, 47769895.45848727, np.nan, np.nan, np.nan, 291362.1011551686, 39474.31593690805, 9589.503202390224, 124739.80733274488, 11982708.28970308, 4168.100267369578, 5183.254186298416, 1058.6171707303968, 3874.1562453904444, 2147.342528042627, 44708.695501397335, 276603.41024425183, 16793.466420436056, 35135.04243041352, 666249.0544927322, 51096.07143284269, 88145.16694195889, 40778.85505829457, 1574.4642835334737, 7829.319929553588, 1284.5759436380242, 4280.742690057814, 7154.47875413484, 1656.7438406461622, 23466.297850735496, 3680.441611602024, 4660621.590522298, 427769.6842499587, 25003.62980571652, np.nan, np.nan, np.nan, np.nan, 185487.1689265136, 13516.88869938205, 5333310609.797244, np.nan, 44512.41986860104, 245895.79112718694, 256836.4960798537, 19564.999439810676, 1531598.0517832474, np.nan, np.nan, 109361.49206227668, 239.2234801755708, 146316.00153524277, 5512.702975831789, 110.52930869007442, np.nan, np.nan, 208695.93012372294, 32555236.836856537, 111501.2327079978, 123750.20456140868, 43173.52072081756, 9918.708354425038, 14307.234305331689, 1672592.2400242146, 14878.642126911756, 41752006.46533171, 8730.512076608074, 16796.61728146437, 3000.1960182668167, 12565.615725484644, 115396.50963664772, 11511834881.005133, 2301.9411343842507] 

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
