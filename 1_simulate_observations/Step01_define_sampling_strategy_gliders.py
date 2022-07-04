#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import netCDF4               as netcdf
from matplotlib              import dates as mdates
import EuroSea_toolbox       as to
from mpl_toolkits.basemap    import Basemap
import pickle

"""

Step 1: 
    
>> Define glider sampling strategy: get (time, lon, lat, dep) of each profile

Configuration #5

* start in the MedSea, extend to Atlantic.



>> Save (time, lon, lat, dep) of the glider profiles 



written by Bàrbara Barceló-Llull on 18-02-2021 at IMEDEA (Mallorca, Spain)

"""


def plot_bm_decor_Med(bm, fsize, label_y=1):
    
    # decor map
    bm.drawcoastlines()
    bm.fillcontinents(color='0.7', lake_color='0.7', zorder=5)
    parallels = np.arange(34.,43.,1.)
    bm.drawparallels(parallels,labels=[label_y, 0, 0, 0],fontsize=fsize, 
                     linewidth=0.2, zorder=8)
    meridians = np.arange(-6,7.,1.)
    bm.drawmeridians(meridians,labels=[0, 0, 0, 1],fontsize=fsize, 
                     linewidth=0.2, zorder=9)

def plot_bm_decor_Atl(bm, fsize, lonmin, lonmax, latmin, latmax, label_y=1,):
    
    # decor map
    bm.drawcoastlines()
    bm.fillcontinents(color='0.7', lake_color='0.7', zorder=5)
    parallels = np.arange(latmin,latmax,1.)
    bm.drawparallels(parallels,labels=[label_y, 0, 0, 0],fontsize=fsize, 
                     linewidth=0.2, zorder=8)
    meridians = np.arange(lonmin,lonmax,1.)
    bm.drawmeridians(meridians,labels=[0, 0, 0, 1],fontsize=fsize, 
                     linewidth=0.2, zorder=9)

def plot_bm_decor(bm, tz, lonmin, lonmax, latmin, latmax, label_y=1):
    
    if region == 'Med':  
        
        plot_bm_decor_Med(bm, tz+1, label_y=1)
        
    elif region == 'Atl':
        
        plot_bm_decor_Atl(bm, tz+1, lonmin, lonmax, latmin, latmax, label_y=1)
    
   
def plot_glider_sampling_strategy(region, lon_uctd_pfs, lat_uctd_pfs, time_uctd_pfs, 
                               name_scenario, name_platform, fig_dir):  
    
    # Coordinates of the swaths of SWOT in the Med Sea or Atlantic 
    
    lonsw1, latsw1, lonsw2, latsw2, lonnd1, latnd1, lonnd2, latnd2 = \
                                                 to.coord_SWOT_swath(region)


    # Create basemap for  figures 
    
    if region == 'Med':  
                                           
        lonmin, lonmax, latmin, latmax = 1., 4, 39, 42  

        bm = Basemap(projection = 'merc',llcrnrlon = lonmin,
                                   urcrnrlon = lonmax,
                                   llcrnrlat = latmin,
                                   urcrnrlat = latmax,
                                   lat_ts = 36,
                                   resolution = 'h')  

        # file topography
        
        dir_topo    = '/Users/bbarcelo/HOME_SCIENCE/Data/PRE-SWOT_Dades/topography/'
        file_topo   = 'usgsCeSrtm30v6_8303_496d_dd25.nc'
        
        # open topography
        
        nc        = netcdf.Dataset(dir_topo + file_topo, 'r')
        lat_topo  = nc.variables['latitude'][:] 
        lon_topo  = nc.variables['longitude'][:] 
        topo      = nc.variables['topo'][:] # [m]  
        nc.close()
        
        
        # limit region 

        ilont_dom = np.where(np.logical_and(lon_topo>=lonmin-1, lon_topo<=lonmax+1))
        jlatt_dom = np.where(np.logical_and(lat_topo>=latmin-1, lat_topo<=latmax+1))
    
        lon_topo_dom = lon_topo[ilont_dom]
        lat_topo_dom = lat_topo[jlatt_dom][::-1] # increasing!!
        topo_dom     = topo[jlatt_dom,:].squeeze()[:, ilont_dom].squeeze()[::-1, :] #lat axis in increasing order
   
        lon_topo2d, lat_topo2d = np.meshgrid(lon_topo_dom, lat_topo_dom)

        # open MDT
        dir_mdt  = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/Med_MDT/'
        file_mdt = 'SMDT-MED-2014.nc'

        # open MDT
        
        nc       = netcdf.Dataset(dir_mdt + file_mdt, 'r')
        lat_mdt  = nc.variables['latitude'][:].data
        lon_mdt  = nc.variables['longitude'][:].data
        mdt_mk  = nc.variables['MDT'][:] # [m]   
        nc.close()
        
        # remove mask to mdt
        mdt = np.copy(mdt_mk.data)
        
        mdt[mdt_mk.mask==True] = np.nan
        
        # compute geostrophic velocity from MDT
        
        lon_mdt2d, lat_mdt2d, ug, vg = to.compute_vgeo_from_mdt(lon_mdt, lat_mdt, mdt)

            
        
    elif region == 'Atl':
                                         
        #lonmin, lonmax, latmin, latmax = -55,-40, 30, 40
        lonmin, lonmax, latmin, latmax = -50,-46, 34, 36


        bm = Basemap(projection = 'merc',llcrnrlon = lonmin,
                                   urcrnrlon = lonmax,
                                   llcrnrlat = latmin,
                                   urcrnrlat = latmax,
                                   lat_ts = 36,
                                   resolution = 'h')   


    xsw1, ysw1  = bm(lonsw1, latsw1)
    xsw2, ysw2  = bm(lonsw2, latsw2)
    xnd1, ynd1  = bm(lonnd1, latnd1)
    xnd2, ynd2  = bm(lonnd2, latnd2)

    tz = 12    

    x_casts, y_casts = bm(lon_uctd_pfs, lat_uctd_pfs)
    
    fig = plt.figure(figsize=(13,6))
    ax1 = plt.subplot(121)
    
    plot_bm_decor(bm, tz, lonmin, lonmax, latmin, latmax, label_y=1)
    
    if region == 'Atl':
        plt.scatter(xsw1.flatten(), ysw1.flatten(), s=10, c='lightskyblue', alpha=0.2)
        plt.scatter(xsw2.flatten(), ysw2.flatten(), s=10, c='lightskyblue', alpha=0.2)
        plt.scatter(xnd1.flatten(), ynd1.flatten(), s=10, c='lightskyblue', alpha=0.2)
        plt.scatter(xnd2.flatten(), ynd2.flatten(), s=10, c='lightskyblue', alpha=0.2)


    elif region == 'Med':

        x_top, y_top  = bm(lon_topo2d, lat_topo2d)
        x_mdt, y_mdt  = bm(lon_mdt2d, lat_mdt2d)

        ctopf = plt.contourf(x_top, y_top, topo_dom, cmap = plt.cm.YlGnBu_r,#Blues_r, #cmo.haline, 
                       levels = np.arange(-3300,1,10), zorder=1, extend='min')
        
        cs1000 = plt.contour(x_top, y_top, topo_dom, levels=[-1000],#Blues_r, #cmo.haline, 
                       colors='w', linewidths = 1, zorder=1000)
        
        cs500 = plt.contour(x_top, y_top, topo_dom, levels=[-500],#Blues_r, #cmo.haline, 
                       colors='b', linewidths = 1, zorder=1000)
        
        # label the contours
        plt.clabel(cs1000, fmt='%d')
        plt.clabel(cs500, fmt='%d')

        qv = plt.quiver(x_mdt, y_mdt, ug, vg, color='k',scale = 7, zorder=900)
        plt.quiverkey(qv, 0.12, 0.9, 0.25, '0.25 m/s', 
                      coordinates='figure', color='k', alpha=1)  
        
        plt.scatter(xsw1.flatten(), ysw1.flatten(), c='lightskyblue',)
        plt.scatter(xsw2.flatten(), ysw2.flatten(), c='lightskyblue')
        plt.scatter(xnd1.flatten(), ynd1.flatten(), c='lightskyblue')
        plt.scatter(xnd2.flatten(), ynd2.flatten(), c='lightskyblue')

    plt.scatter(x_casts, y_casts, c='r', s=10, zorder=1100)
    
    # plt.xlim(xlimmin, xlimmax)
    # plt.ylim(ylimmin, ylimmax)
    
    plt.title(name_scenario  + ' >> ' + name_platform)
    
    ax2=plt.subplot(122)
    
    
    sc = plt.scatter(lon_uctd_pfs, lat_uctd_pfs, c=time_uctd_pfs, cmap=plt.cm.Spectral_r)
    
    plt.text(lon_uctd_pfs[0,0]-0.03, lat_uctd_pfs[0,0]+0.035, 
                     mdates.num2date(time_uctd_pfs[0,0]).strftime("%H:%M\n%d-%m "),
                     fontsize=9)          
    plt.text(lon_uctd_pfs[0,-1]-0.03, lat_uctd_pfs[0,-1]+0.035, 
                     mdates.num2date(time_uctd_pfs[0,-1]).strftime("%H:%M\n%d-%m "),
                     fontsize=9)     
            
    plt.colorbar(sc, ticks=mdates.HourLocator(interval=9), #DayLocator(interval=1), 
                  format=mdates.DateFormatter('%H:%M \n %b %d'), 
                  orientation='horizontal')
        
    plt.axis('image')   
    plt.ylim(lat_uctd_pfs.min()-0.1, lat_uctd_pfs.max()+0.1)      
    
    plt.title(name_platform)    
    
    plt.tight_layout()
    fig.savefig(fig_dir + name_scenario+'_' + name_platform +'.png')

  
    
if __name__ == '__main__':

    plt.close('all')
    
    fig_dir = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/configurations/'
    dir_dic = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/configurations/'


    ''' Which region? '''
    
    region = 'Med' # options 'Atl' or 'Med'
    
    
    
    '''
    >>>>>> 1) Simulate glider sampling strategy <<<<<<
    
    7 gliders, each one sampling a zonal transect from east to west simultaneusly
    
    Note 1: constant glider velocity
             
    Note 2: one vertical profile every dx km 
    
    '''   
    
    # define start time of the sampling 
    t_samp_ini    = mdates.date2num(datetime(2009, 9, 1, 0, 0, 0)) #[python format in num. of days]
    
    # Parameters to define the glider sampling strategy (change as you wish)
    
    vel_gl_ms       = 0.25  # [m/s] glider horizontal velocity
    
    dx_gl_km        = 6    # [km] resolution of the glider profiles 
    len_trans       = 80   # km, length of each zonal transect
    sep_trans       = 10   # km, separation between zonal transects
    num_trans       = 7    # total number of transects > one glider in each transect
    
    dep_gl_min    = 30   # m upper depth layer
    dep_gl_max    = 500  # m maximum depth of each cast
    dep_gl_res    = 1    # m vertical resolution of the data
    
    # number and name of this configuration
    
    conf = '5' 
    name_scenario = region + '_conf_5_dep_'+'%04i'% dep_gl_max+'m_res_'+'%02i'% dx_gl_km + 'km_Sep'
      
    
    # top glider zonal transect --> latitude, and longitude west (then calculate lon_east)
    # to have the same coordinates than the other configurations (lon_ini, lat_ini)

    
    if region == 'Med':  
        
        lon_west = 1.45 # = lon_ini of the other configurations
        lat_top  = 40.415 # = lat_ini of the other configurations     

    elif region == 'Atl':  
        
        lon_west = -48.7 # = lon_ini of the other configurations
        lat_top = 35.25  # = lat_ini of the other configurations  
        
        
    # >>>> Computations needed <<<<<

    # length of a degree of longitude and latitude
    length_deg_lon, length_deg_lat = to.length_lon_lat_degs(lat_top)   

    # beginning of 1st transect: lon_ini
    # end of 1st transect: lon_fin
    len_trans_lon_deg = len_trans/(length_deg_lon/1000)
    lon_east          = lon_west + len_trans_lon_deg  # = lon_fin of the other conf.
    
    # separation between transects in latitude degrees  
    sep_trans_lat_deg = sep_trans /(length_deg_lat/1000)



    # >>>> Depth axis (save) <<<<<
    
    # define depth axis for uCTD profiles
    dep_gl = np.arange(dep_gl_min, dep_gl_max+dep_gl_res, dep_gl_res)



    # >>>> Longitude axis (common for all zonal transects) <<<<<

    # glider profile resolution in longitude degrees
    dlon_gl        = dx_gl_km /(length_deg_lon/1000)    
    
    # longitude axis for one glider transect    
    lon_gl_pfs_1D  = np.arange(lon_east, lon_west, -dlon_gl)
    
    # repeat for all glider transects, shape(num_trans, num prof in tran)
    lon_gl_pfs     = np.tile(lon_gl_pfs_1D, (num_trans,1))
    
    
    
    # >>>> Latitude axis <<<<<
    
    lat_gl_pfs = np.ones(lon_gl_pfs.shape) * np.nan
    
    
    for n in np.arange(num_trans): 
        
        lat_gl_itran      = lat_top - n * sep_trans_lat_deg
        lat_gl_pfs[n, :]  = np.repeat(lat_gl_itran, lat_gl_pfs.shape[1])
        
        
        
    # >>>> Time axis <<<<<
    
    # time resolution of glider profiles 
    dt_gl_s      = dx_gl_km*1000/vel_gl_ms  #[s]
    dt_gl_min    = dt_gl_s/60  # [min]
    
    # time resolution of glider profiles in days
    dt_gl_fracdays = dt_gl_min/(60*24)    #as fraction of days
    
    # time needed for a glider to sample a zonal transect
    time_sampling_s    = len_trans*1000/vel_gl_ms  #[s]
    time_sampling_days = time_sampling_s/(60*60*24)    
    
    print('')
    print('Time needed to do the glider sampling (days)...', time_sampling_days)
    print('')
    

    # date of each glider profile (save)
    time_gl_1D  = np.arange(t_samp_ini, t_samp_ini+time_sampling_days, dt_gl_fracdays)
    
    time_gl_pfs = np.tile(time_gl_1D, (num_trans,1))
    
    print('Sampling starting...', mdates.num2date(time_gl_pfs.min()).strftime("%H:%M %d-%m"))
    print('Sampling ending...', mdates.num2date(time_gl_pfs.max()).strftime("%H:%M %d-%m"))
    print('')
    
    
    ''' Plot CTD sampling strategy + swaths of SWOT '''    
    
    plot_glider_sampling_strategy(region, lon_gl_pfs, lat_gl_pfs, time_gl_pfs, 
                                  name_scenario, 'gliders', fig_dir)
 
    
    
    ''' Save (time, lon, lat, dep) of the glider profiles '''
    
    dic_gl  = {}
    
    # (time_uctd_pfs, lon_uctd_pfs, lat_uctd_pfs, dep_uctd) 
    dic_gl.update({'time_ctd': time_gl_pfs, 
                    'lon_ctd' : lon_gl_pfs,
                    'lat_ctd' : lat_gl_pfs,
                    'dep_ctd' : dep_gl}) 
    

    f_gl = open(dir_dic + name_scenario + '_gliders.pkl','wb')
    pickle.dump(dic_gl,f_gl)
    f_gl.close()         
    
    
    
    
  
    
    
     
        