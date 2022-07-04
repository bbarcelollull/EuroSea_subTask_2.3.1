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
    
>> Define CTD sampling strategy: get (time, lon, lat, dep) of each cast

Valid for reference configuration and configurations 1, 2 and 4.

* MedSea and Atlantic.


>> With the defined CTD sampling strategy, find (time, lon, lat, dep)
of the ADCP profiles. 


>> Save (time, lon, lat, dep) of the CTD and ADCP profiles in each configuration


In next codes: simulate uCTD and glider, extract model data at these locations.


written by Bàrbara Barceló-Llull on 01-02-2021 at IMEDEA (Mallorca, Spain)

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
        
    
def make_figure_CTD(bm, region, lon_topo2d=None, lat_topo2d=None, topo_dom=None,
                    lon_mdt2d=None, lat_mdt2d=None, ug=None, vg=None):    
        
    xsw1, ysw1  = bm(lonsw1, latsw1)
    xsw2, ysw2  = bm(lonsw2, latsw2)
    xnd1, ynd1  = bm(lonnd1, latnd1)
    xnd2, ynd2  = bm(lonnd2, latnd2)
    
    # add box with the reference configuration domain to center the 
    # configuration 2 samplings
    lons_ref = [1.45, 2.39]
    lats_ref = [39.87, 40.415]
    
    x_ref, y_ref = bm(lons_ref, lats_ref)
    
    tz = 12    

    x_casts, y_casts = bm(lon_casts, lat_casts)
    
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
        
        plt.scatter(xsw1.flatten(), ysw1.flatten(), c='lightskyblue', alpha=0.2)
        plt.scatter(xsw2.flatten(), ysw2.flatten(), c='lightskyblue', alpha=0.2)
        plt.scatter(xnd1.flatten(), ynd1.flatten(), c='lightskyblue', alpha=0.2)
        plt.scatter(xnd2.flatten(), ynd2.flatten(), c='lightskyblue', alpha=0.2)
        
        # add box with the reference configuration domain to center the 
        # configuration 2 samplings

        # plt.plot([x_ref[0], x_ref[0], x_ref[1], x_ref[1], x_ref[0]],
        #          [y_ref[0], y_ref[1], y_ref[1], y_ref[0], y_ref[0]],
        #          '-b', zorder=2000)

    plt.scatter(x_casts, y_casts, c='r', s=15, zorder=1100)    
    
    
    plt.title(name_scenario + ' >> CTD')
    
    ax2=plt.subplot(122)
    plt.scatter(lon_casts, lat_casts, c=np.arange(num_casts), cmap=plt.cm.Spectral_r)
    
    for i, txt in enumerate(np.arange(num_casts)+1):
        plt.annotate(txt, (lon_casts[i]+0.005, lat_casts[i]+0.01))

    # for i, txt_time in enumerate(time_casts):
    #     plt.annotate(mdates.num2date(txt_time).strftime("%H:%M %d-%m "), 
    #                  (lon_casts[i]-0.5, lat_casts[i]+0.5))  
    for j, txt_time in enumerate(time_casts):
        # plt.annotate(mdates.num2date(txt_time).strftime("%H:%M %d-%m "), 
        #              (lon_casts[j]-0.03, lat_casts[j]+0.03))  
        plt.text(lon_casts[j]-0.03, lat_casts[j]+0.035, 
                     mdates.num2date(txt_time).strftime("%H:%M %d-%m "),
                     fontsize=7)
        
    plt.axis('image')   
    plt.ylim(lat_casts.min()-0.1, lat_ini+0.1)  
    plt.xlim(lon_casts.min()-0.04, lon_casts.max()+0.08)    
    
    plt.title('CTD')    
    
    plt.tight_layout()
    fig.savefig(fig_dir + name_scenario+'_CTD.png')
    
def make_figure_ADCP(bm):  
    
    xsw1, ysw1  = bm(lonsw1, latsw1)
    xsw2, ysw2  = bm(lonsw2, latsw2)
    xnd1, ynd1  = bm(lonnd1, latnd1)
    xnd2, ynd2  = bm(lonnd2, latnd2)

    tz = 12    

    x_adcp, y_adcp = bm(lon_adcp_pf, lat_adcp_pf)

    
    fig = plt.figure(figsize=(13,6))
    ax1 = plt.subplot(121)
    
    plot_bm_decor(bm, tz, lonmin, lonmax, latmin, latmax, label_y=1)
    
    if region == 'Atl':
        plt.scatter(xsw1.flatten(), ysw1.flatten(), s=10, c='lightskyblue', alpha=0.2)
        plt.scatter(xsw2.flatten(), ysw2.flatten(), s=10, c='lightskyblue', alpha=0.2)
        plt.scatter(xnd1.flatten(), ynd1.flatten(), s=10, c='lightskyblue', alpha=0.2)
        plt.scatter(xnd2.flatten(), ynd2.flatten(), s=10, c='lightskyblue', alpha=0.2)


    elif region == 'Med':
        
        plt.scatter(xsw1.flatten(), ysw1.flatten(), c='lightskyblue',)
        plt.scatter(xsw2.flatten(), ysw2.flatten(), c='lightskyblue')
        plt.scatter(xnd1.flatten(), ynd1.flatten(), c='lightskyblue')
        plt.scatter(xnd2.flatten(), ynd2.flatten(), c='lightskyblue')

    plt.scatter(x_adcp, y_adcp, c=time_adcp_pf, s=3,  cmap=plt.cm.Spectral_r)
    
    # plt.xlim(xlimmin, xlimmax)
    # plt.ylim(ylimmin, ylimmax)
    
    plt.title(name_scenario  + ' >> ADCP')
    
    ax2=plt.subplot(122)
    
    # ADCP
    plt.scatter(lon_adcp_pf, lat_adcp_pf, c=time_adcp_pf, s=8, cmap=plt.cm.Spectral_r)
    
    plt.colorbar(ticks=mdates.DayLocator(interval=1), 
                  format=mdates.DateFormatter('%b %d'),
                  orientation='horizontal')
    # CTD casts
    plt.scatter(lon_casts, lat_casts, c='k', s=8, marker='s')
    
    for i, txt in enumerate(np.arange(num_casts)+1):
        plt.annotate(txt, (lon_casts[i]+0.005, lat_casts[i]+0.01))
    plt.axis('image')   
    plt.ylim(lat_casts.min()-0.1, lat_ini+0.1)    
      
    plt.title('CTD + ADCP')    
    
    plt.tight_layout()
    fig.savefig(fig_dir + name_scenario+'_ADCP.png')


    
if __name__ == '__main__':


    
    plt.close('all')
    
    
    ''' Directories to save figures and data '''
    
    fig_dir = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/configurations/'
    dir_dic = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/configurations/'


    ''' Which configuration do you want to simulate? '''
    
    region = 'Med' # 'Med' or 'Atl'
    
    conf = 'r' # 'r' is reference, '1', '2', and '4'
    
    # if conf='2', which resolution do you want to simulate? (5, 8, 12 or 15 km)
    subconf = 12  # available options: (5, 8, 12 or 15 km)



    ''' Definition of the different settings for each configuration '''

    if conf == 'r':
        

        # define start time of the sampling 
        t_samp_ini    = mdates.date2num(datetime(2009, 9, 1, 0, 0, 0)) #[python format in num. of days]

        # To test different dates for the reference configuration
        # t_samp_ini    = mdates.date2num(datetime(2009, 8, 1, 0, 0, 0))
        # t_samp_ini    = mdates.date2num(datetime(2009, 10, 1, 0, 0, 0))
        
        str_month = mdates.num2date(t_samp_ini).strftime("%b")
        
        # reference configuration
        name_scenario = region + '_conf_r_dep_1000m_res_10km_' + str_month
        
        
        # Parameters to define the CTD sampling strategy (change as you wish)
    
        cast_res_km = 10 # resolution of the CTD casts [km]
        dep_ctd_max = 1000 #m maximum depth of each cast
        
        
    elif conf == '1':
 
        # define start time of the sampling 
        t_samp_ini    = mdates.date2num(datetime(2009, 9, 1, 0, 0, 0)) #[python format in num. of days]

        str_month = mdates.num2date(t_samp_ini).strftime("%b")
        
        # configuration 1: CTD casts at 500 m depth
        name_scenario = region + '_conf_1_dep_0500m_res_10km_' + str_month
    

        # Parameters to define the CTD sampling strategy (change as you wish)
    
        cast_res_km = 10 # resolution of the CTD casts [km]
        dep_ctd_max = 500 #m maximum depth of each cast        
        
    elif conf == '2':
        
        #configuration 2: test different resolutions of CTD casts (5, 8, 12, 15 km))
    
        # define start time of the sampling 
        t_samp_ini    = mdates.date2num(datetime(2009, 9, 1, 0, 0, 0)) #[python format in num. of days]

        str_month = mdates.num2date(t_samp_ini).strftime("%b")
        
        # Parameters to define the CTD sampling strategy (change as you wish)
    
        cast_res_km = subconf # (5, 8, 12, 15 km) resolution of the CTD casts [km]
        dep_ctd_max = 1000 #m maximum depth of each cast          
        
        name_scenario = region + '_conf_2_dep_1000m_res_' + '%02i'% cast_res_km  + 'km_' + str_month
  
    elif conf == '4':
        
    
        # define start time of the sampling 
        t_samp_ini    = mdates.date2num(datetime(2010, 1, 1, 0, 0, 0)) #[python format in num. of days]
    
        str_month = mdates.num2date(t_samp_ini).strftime("%b")

        # reference configuration
        name_scenario = region + '_conf_4_dep_1000m_res_10km_' + str_month
        
        # Parameters to define the CTD sampling strategy (change as you wish)
    
        cast_res_km = 10 # resolution of the CTD casts [km]
        dep_ctd_max = 1000 #m maximum depth of each cast        
        
    else:
        
        'Error! Define which configuration to simulate! '
        
    # configuration 3 --> in another code            
    # configuration 5 --> in another code    
        
        
    ''' 
    Common parameters for all configurations that 
    define the CTD and ADCP sampling strategies 
    '''
    
    # CTD parameters
    
    ship_speed_kt   = 8  # knots
    
    # if np.logical_and(region == 'Med', 
    #     np.logical_and(conf == '2', np.logical_or(subconf == 12, subconf == 15))): 
    if np.logical_and(region == 'Med', 
        np.logical_and(conf == '2', subconf == 12)): 
        
        ''' In the Med for conf=2 and 12 km resolution, we remove the 
        western and northern transect '''
        
        num_casts_row   = 9 - 2 # to stay in deep waters (updated on 26-Apr-2021) #1
        num_casts_col   = 7 - 1
        
    elif np.logical_and(region == 'Med', 
        np.logical_and(conf == '2', subconf == 15)): 
        
        ''' In the Med for conf=2 and 15 km resolution, we remove the 
        western and northern 2 transects '''
        
        num_casts_row   = 9 - 3 # to stay in deep waters (updated on 26-Apr-2021)#2
        num_casts_col   = 7 - 2  
        
    else:                  
        num_casts_row   = 9 #7  # number of casts in a row
        num_casts_col   = 7 #9  # number of casts in a column 
      
    
    dep_ctd_min     = 5    #m upper depth layer
    dep_ctd_res     = 0.5  #m vertical resolution of the data
    

    # check the time needed to do the CTD cast depending on the maximum depth
    if dep_ctd_max == 1000:
      t_cast_min = 60  # min 
    elif dep_ctd_max == 500:
      t_cast_min = 30  # min 
      
      
    # ADCP parameters
      
    adcp_res_min    = 5 #one profile every 5 minutes, in PRE-SWOT 468 seconds (7.8 min)
    dep_adcp_min    = 20 #m, in PRE-SWOT 19.96 m
    dep_adcp_max    = 600 #m, in PRE-SWOT 587.96 M
    dep_adcp_res    = 8 #m # bin size of the ADCP data
    
    # Start point for each region > coordinates of cast 1 (top, left cast)
    
    if region == 'Med':
        
        ''' In the Mediterranean we want all configurations to cover
        the south-eastern part of the domain. Because of this we give
        the coordinates of the last CTD cast and from here we calculate 
        the coordinates of the first CTD cast '''

        # # coordinates of the last CTD cast
        # # to capture the Balearic current
        # lon_fin_ref = 2.39166
        # lat_fin_ref = 39.80966
        

        # # length of a degree of longitude and latitude
        # length_deg_lonp, length_deg_latp = to.length_lon_lat_degs(lat_fin_ref)    
    
        # # cast resolution from km to longitude and latitude degrees
        # dlon_ctdp = cast_res_km /(length_deg_lonp/1000)
        # dlat_ctdp = cast_res_km /(length_deg_latp/1000)
        
        # # coordinates of the first CTD cast
        
        # lon_ini = lon_fin_ref - (num_casts_row-1) *dlon_ctdp
        # lat_ini = lat_fin_ref + (num_casts_col-1) *dlat_ctdp

        # print( 'lon ini = ', lon_ini) 
        # print( 'lat ini = ', lat_ini) 
        
        # Domain moved to the north (last update, on 22 April 2021)
        
        lon_ini_all = 1.45
        lat_ini_all = 40.415   # 40.35  
        
        lon_ini = lon_ini_all
        lat_ini = lat_ini_all

        if conf == '2':
            if cast_res_km == 5:
                lon_ini = lon_ini_all  + 0.23
                lat_ini = lat_ini_all  - 0.13   
            
            elif cast_res_km == 8:    
                lon_ini = lon_ini_all + 0.093
                lat_ini = lat_ini_all - 0.055

            elif cast_res_km == 12:    
                lon_ini = lon_ini_all 
                lat_ini = lat_ini_all           

            elif cast_res_km == 15:    
                lon_ini = lon_ini_all 
                lat_ini = lat_ini_all 
            
            else:
                lon_ini = lon_ini_all
                lat_ini = lat_ini_all            
            
        else: 
            lon_ini = lon_ini_all
            lat_ini = lat_ini_all
            
            
        # if conf == '2':
        #     if cast_res_km == 5:
        #         # lon_ini = lon_ini_all + 0.57
        #         # lat_ini = lat_ini_all
                
        #         lon_ini = lon_fin_ref - (num_casts_row-1) *dlon_ctdp
        #         lat_ini = lat_fin_ref + (num_casts_col-1) *dlat_ctdp
                
                
        #     elif cast_res_km == 8:    
        #         # lon_ini = lon_ini_all
        #         # lat_ini = lat_ini_all  

        #         lon_ini = lon_fin_ref - (num_casts_row-1) *dlon_ctdp
        #         lat_ini = lat_fin_ref + (num_casts_col-1) *dlat_ctdp
                
        #     elif cast_res_km == 12:    
        #         # lon_ini = lon_ini_all - 0.12
        #         # lat_ini = lat_ini_all 
                
        #         lon_ini = lon_fin_ref - (num_casts_row-1) *dlon_ctdp
        #         lat_ini = lat_fin_ref + (num_casts_col-1) *dlat_ctdp                

        #     elif cast_res_km == 15:    
        #         # lon_ini = lon_ini_all - 0.27
        #         # lat_ini = lat_ini_all + 0.1 

        #         lon_ini = lon_fin_ref - (num_casts_row-1) *dlon_ctdp
        #         lat_ini = lat_fin_ref + (num_casts_col-1) *dlat_ctdp
                
        #     else:
        #         lon_ini = lon_ini_all
        #         lat_ini = lat_ini_all            
            
        # else: 
        #     lon_ini = lon_ini_all
        #     lat_ini = lat_ini_all
    

    elif region == 'Atl':
        
        lon_ini_all = -48.7
        lat_ini_all = 35.25   
        
        
        if conf == '2':
            if cast_res_km == 5:
                lon_ini = lon_ini_all 
                lat_ini = lat_ini_all
            
            elif cast_res_km == 8:    
                lon_ini = lon_ini_all
                lat_ini = lat_ini_all  

            elif cast_res_km == 12:    
                lon_ini = lon_ini_all - 0.08
                lat_ini = lat_ini_all           

            elif cast_res_km == 15:    
                lon_ini = lon_ini_all - 0.21
                lat_ini = lat_ini_all + 0.1 
            
            else:
                lon_ini = lon_ini_all
                lat_ini = lat_ini_all            
            
        else: 
            lon_ini = lon_ini_all
            lat_ini = lat_ini_all
            
            


    '''
    >>>>>> 1) Simulate CTD sampling strategy <<<<<<
    
    To obtain lon, lat and time of each CTD cast.
    
    Note 1: We assume a regular grid. Define before the number of casts 
            in each row and column.
    
    Note 2: Order of casts: Row 1 (top) from left to right, row 2 from right to left, etc.
        E.g. c01, c02, c03, c04, c05, c06, c07
             c14, c13, c12, c11, c10, c09, c08
             c15, c16, etc. 
             
    Note 3: We assume that during the cast the water column properties do
            not change and we will extract the data corresponding to the 
            time of the cast launch.
    '''     


    # >>>> Computations needed <<<<<
    
    kt2ms          = 0.514444 # 1 knot is 0.514444 m/s
    ship_speed_ms  = ship_speed_kt * kt2ms
    t_transit_s    = cast_res_km*1000/ship_speed_ms # transit time in [s]
    t_transit_min  = t_transit_s/60
    # dist_lat_deg   = cast_res_km
    # dist_lon_deg   = cast_res_km     

    # length of a degree of longitude and latitude
    length_deg_lon, length_deg_lat = to.length_lon_lat_degs(lat_ini)    
    
    # cast resolution from km to longitude and latitude degrees
    dlon_ctd = cast_res_km /(length_deg_lon/1000)
    dlat_ctd = cast_res_km /(length_deg_lat/1000)
    
    # >>>> compute start date of each CTD cast <<<<<
    # considering t_samp_ini, t_transit_min, t_cast_min
    
    t_transit_fracdays = t_transit_min/(60*24) #as fraction of days
    t_cast_fracdays    = t_cast_min/(60*24)    #as fraction of days
    
    num_casts  = num_casts_row*num_casts_col
    time_casts = np.ones(num_casts) * np.nan
    
    for ic in np.arange(num_casts):
        #ic = 0 is cast 1, etc. 
        time_casts[ic] = t_samp_ini + ic*t_cast_fracdays + ic*t_transit_fracdays

        
    # >>>> compute coordinates of each CTD cast <<<<<   
            
    
    # lon and lat coordinates of each column and row
    
    lon_casts_col = np.ones(num_casts_row) * np.nan
    
    lon_casts = np.ones(num_casts) * np.nan
    lat_casts = np.ones(num_casts) * np.nan
    
    for icol in np.arange(num_casts_row):
        
        lon_casts_col[icol] =  lon_ini + icol*dlon_ctd
        

    for irow in np.arange(num_casts_col):
        
        lat_casts_irow =  lat_ini - irow*dlat_ctd
        lat_casts[irow*num_casts_row:(irow+1)*num_casts_row] = lat_casts_irow
        

        if irow in np.arange(0, num_casts_col, 2):
            # longitude increasing from cast 0 to cast 7 (num_casts_row)  
            lon_casts[irow*num_casts_row:(irow+1)*num_casts_row] = lon_casts_col
    
        elif irow in np.arange(1, num_casts_col, 2):
            # longitude decreasing from cast 8 to cast 14 
            lon_casts[irow*num_casts_row:(irow+1)*num_casts_row] = lon_casts_col[::-1]
    
        
    # define depth axis for each cast
    
    dep_casts = np.arange(dep_ctd_min, dep_ctd_max+dep_ctd_res, dep_ctd_res)
        
    
    # Now we have (time_casts, lon_casts, lat_casts, dep_casts) for this CTD sampling strategy.
         
    
    ''' Plot CTD sampling strategy + swaths of SWOT ''' 
    
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

        
        # Make figure
        make_figure_CTD(bm, region, lon_topo2d, lat_topo2d, topo_dom,
                        lon_mdt2d, lat_mdt2d, ug, vg)
            
        
    elif region == 'Atl':
                                         
        #lonmin, lonmax, latmin, latmax = -55,-40, 30, 40
        lonmin, lonmax, latmin, latmax = -50,-46, 34, 36


        bm = Basemap(projection = 'merc',llcrnrlon = lonmin,
                                   urcrnrlon = lonmax,
                                   llcrnrlat = latmin,
                                   urcrnrlat = latmax,
                                   lat_ts = 36,
                                   resolution = 'h')   


    
        make_figure_CTD(bm, region)
    
    
    
    '''
    >>>>>> 2) Simulate ADCP profiles <<<<<<
    
    To obtain lon, lat and time of each ADCP profile.
    
    Note 1: 1 profile every 5 minuts during transit, discard time and place 
    where CTD casts.
    
    Note 2: We remove data at the CTD cast position/time and in a distance 
    smaller than the resolution of the ADCP profiles (adcp_res_km).
    During this time the ship is decelerating/accelerating and we disregard data. 

    '''      
    
    
    adcp_res_s   = adcp_res_min * 60 # in seconds 
    adcp_res_km  = (ship_speed_ms*adcp_res_s)/1000 #in km
    
    adcp_res_fracdays = adcp_res_min/(60*24) 
    
    
    
    ''' 
    
    >>> Time of each ADCP profile <<<
    
    
    Trans 1 between CTD 1 and CTD 2, i.e., between 
    (time0 + time_ctd) and (time0 + time_ctd + time_transit)
    
    Trans 2 between CTD 2 and CTD 3, i.e., between
    (time0 + 2*time_ctd + time_transit) and(time0 + 2*time_ctd + 2*time_transit)
    
    Trans n between CTD n and CTD n+1, i.e., between
    (time0 + n*time_ctd + (n-1)*time_transit) and(time0 + n*time_ctd + n*time_transit)
    
    time0        = t_samp_ini [days]
    time_ctd     = t_cast_fracdays [days]
    time_transit = + t_transit_fracdays [days]
    
    '''
    
    num_adcp_trans = num_casts - 1
    time_adcp_pf = []
    
    for itran in np.arange(num_adcp_trans):
        
    
        tadcp_itran_i = t_samp_ini + itran*t_cast_fracdays + (itran-1)*t_transit_fracdays
        tadcp_itran_f = t_samp_ini + itran*t_cast_fracdays + (itran)*t_transit_fracdays
        
        #tadcp_itran = np.arange(tadcp_itran_i, tadcp_itran_f, adcp_res_fracdays)
        
        # maybe remove first and final data points because the ship should be already
        # stopped at the CTD cast position... In this way we would remove data
        # at adcp_res_km apart from the CTD cast position. 
        
        
        tadcp_itran = np.arange(tadcp_itran_i+adcp_res_fracdays, 
                                tadcp_itran_f-adcp_res_fracdays, 
                                adcp_res_fracdays)

        # plt.figure()
        # plt.plot(tadcp_itran, 'o-')
        # plt.plot(tadcp_itran_i, 'xb')
        # plt.plot(tadcp_itran_f, 'xr')

        time_adcp_pf = np.append(time_adcp_pf, tadcp_itran)
        
        #number of adcp profiles in each transect between CTD casts
        num_adcp_pf_in_tran = tadcp_itran.shape[0]

     
    ''' 
    >>> longitude and latitude of each ADCP profile <<<
    '''    
    
    # Calculate dlon and dlat for the ADCP resolution 
    
    dlon_adcp = adcp_res_km /(length_deg_lon/1000)
    dlat_adcp = adcp_res_km /(length_deg_lat/1000)    
    
    
    # longitude of each ADCP profile along each row
    
    lon_adcp_row_asc_list  = []
    lon_adcp_row_des_list  = []
    

    for irow in np.arange(num_casts_col):
      # those rows where CTD are from west to east (ascending longitude)
      # will have this longitude for the ADCP profiles       
      if irow == 0:
        lons_ctd = lon_casts[irow*num_casts_row:(irow+1)*num_casts_row]
            
        for il, ll in enumerate(lons_ctd[:-1]): 
                     
            # longitude of the ADCP profiles between the CTD casts
            # at lons_ctd[il] and lons_ctd[il+1]
                     
            lon_adcp_trans = np.arange(ll + dlon_adcp, 
                                                lons_ctd[il+1] - dlon_adcp, 
                                                dlon_adcp)
                     
            # Save the longitude of all the ADCP profiles in the 
            # ascending row
            lon_adcp_row_asc_list = np.append(lon_adcp_row_asc_list, lon_adcp_trans)
                     
      # those rows where CTD are from east to west (descending longitude)  
      # will have this longitude for the ADCP profiles                   
      if irow == 1:
        lons_ctd = lon_casts[irow*num_casts_row:(irow+1)*num_casts_row]
            
        for il, ll in enumerate(lons_ctd[:-1]): 
                     
            # longitude of the ADCP profiles between the CTD casts
            # at lons_ctd[il] and lons_ctd[il+1]
                     
            lon_adcp_trans = np.arange(ll - dlon_adcp, 
                                        lons_ctd[il+1] + dlon_adcp, 
                                        - dlon_adcp)
                 
            # Save the longitude of all the ADCP profiles in the 
            # ascending row
            lon_adcp_row_des_list = np.append(lon_adcp_row_des_list, lon_adcp_trans)            
    
    lon_adcp_row_asc = np.array(lon_adcp_row_asc_list)
    lon_adcp_row_des = np.array(lon_adcp_row_des_list)


    # save these longitudes for the corresponding ADCP profile
    # latitude is the same in each row, changes between rows
            
    lon_adcp_pf = np.ones(time_adcp_pf.shape)*np.nan
    lat_adcp_pf = np.ones(time_adcp_pf.shape)*np.nan
    
    for irow in np.arange(num_casts_col):
        
        #indices of the first and last ADCP profiles in each row
        ind_ini = irow * num_casts_row * num_adcp_pf_in_tran
        ind_fin = (((irow+1) * num_casts_row)-1) * num_adcp_pf_in_tran
        
        # same latitude for all profiles in the same row
        lat_adcp_pf[ind_ini:ind_fin] = lat_casts[irow*num_casts_row]
        
        
        #in these rows save ascending longitude
        if irow in np.arange(0, num_casts_col, 2):
             
             lon_adcp_pf[ind_ini:ind_fin] = lon_adcp_row_asc
             
        # in these rows save descending longitude     
        elif irow in np.arange(1, num_casts_col, 2):
             
             lon_adcp_pf[ind_ini:ind_fin] = lon_adcp_row_des

        # save the longitude that corresponds to the profiles 
        # between rows.         
        if irow < num_casts_col-1:

            ind_bi =  ind_fin 
            ind_bf =  ind_fin + num_adcp_pf_in_tran 
             
            lon_adcp_pf[ind_bi:ind_bf] = lon_casts[(irow+1)*num_casts_row-1]
            
            # between transects latitude decreases
            lat_adcp_pf[ind_bi:ind_bf] = \
                np.arange(lat_casts[(irow+1)*num_casts_row-1] - dlat_adcp, 
                          lat_casts[(irow+1)*num_casts_row] + dlat_adcp, 
                          - dlat_adcp)
                                  
    ''' 
    >>> depth of ADCP profiles <<<
    '''    
    
    dep_adcp_pf = np.arange(dep_adcp_min, dep_adcp_max, dep_adcp_res)
    
    
    # we have time_adcp_pf, lon_adcp_pf, lat_adcp_pf, dep_adcp_pf   


    ''' Plot ADCP sampling strategy + swaths of SWOT '''    
    
    make_figure_ADCP(bm)
    

    
    ''' Save (time, lon, lat, dep) of the CTD and ADCP profiles '''
    
    dic_ctd  = {}
    dic_adcp = {}
    
    # (time_casts, lon_casts, lat_casts, dep_casts)
    dic_ctd.update({'time_ctd': time_casts, 
                    'lon_ctd' : lon_casts,
                    'lat_ctd' : lat_casts,
                    'dep_ctd' : dep_casts}) 
    
    # time_adcp_pf, lon_adcp_pf, lat_adcp_pf, dep_adcp_pf

    dic_adcp.update({'time_adcp': time_adcp_pf, 
                    'lon_adcp'  : lon_adcp_pf,
                    'lat_adcp'  : lat_adcp_pf,
                    'dep_adcp'  : dep_adcp_pf}) 
    
    f_ctd = open(dir_dic + name_scenario + '_ctd.pkl','wb')
    pickle.dump(dic_ctd,f_ctd)
    f_ctd.close()         
    
    f_adcp = open(dir_dic + name_scenario + '_adcp.pkl','wb')
    pickle.dump(dic_adcp,f_adcp)
    f_adcp.close()       
    
    
     
        