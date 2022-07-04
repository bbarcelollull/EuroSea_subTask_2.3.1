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
    
>> Define uCTD sampling strategy: get (time, lon, lat, dep) of each profile

Configuration #3

* MedSea and Atlantic.


>> With the defined uCTD sampling strategy, find (time, lon, lat, dep)
of the ADCP profiles. 


>> Save (time, lon, lat, dep) of the uCTD and ADCP profiles 



written by Bàrbara Barceló-Llull on 11-02-2021 at IMEDEA (Mallorca, Spain)

"""


def extract_lon_lat_continuous_profiles(dist_pfs_all, dlon_uctd, len_trans, sep_trans,
                                        sep_trans_lat_deg, lat_ini, lon_ini, 
                                        lon_fin, length_deg_lon, length_deg_lat,
                                        num_trans):
    
    ''' 
    Extract longitude and latitude of profiles that are done continuously
    from top left to top right, turn downward, then continue the 2nd row from 
    right to left, etc.
    
    Valid for uCTD/Seasoar profiles and the associated ADCP profiles.
    
    This function does this:
        
    > Which of the profiles in dist_pfs_all is in each transect or separation 
    between transects?
    
    > Then, which are the coordinates of these profiles?
    
    
    Input: 
        dist_pfs_all: array with distance of each profile from the beginnning
                      of the sampling (top left)
        dlon_uctd:          horizontal resolution along zonal transect in longitude degrees  
        len_trans:          length of zonal transect [km]
        sep_trans:          separation between zonal transects [km]
        sep_trans_lat_deg:  separation between zonal transects [latitude degrees]
        lat_ini:            latitude of first profile (beginning of sampling)
        lon_ini:            longitude of first profile (beginning of sampling)
        lon_fin:            longitude at len_trans from the beginning of sampling
        length_deg_lon:     length of a degree of longitude [m]
        length_deg_lat:     length of a degree of latitude [m]
        num_trans:          number of zonal transects
        
    Output: lon_uctd_pfs, lat_uctd_pfs
    
    '''
    
    lon_uctd_pfs = np.ones(len(dist_pfs_all)) * np.nan
    lat_uctd_pfs = np.ones(len(dist_pfs_all)) * np.nan    
    
    
    for n in np.arange(num_trans): 
        
        # distance of the profiles in each transect
        
        dist_ini_tran_n = n * (len_trans + sep_trans)
        dist_fin_tran_n = dist_ini_tran_n + len_trans
        
        cond_tran_n =  np.logical_and(dist_pfs_all >= dist_ini_tran_n, 
                                      dist_pfs_all <= dist_fin_tran_n)         

        dist_pfs_tran_n = dist_pfs_all[cond_tran_n] # save distance
        
        # Which are the coordinates of these profiles?
        
        # latitude of profiles in this transect
        
        lat_uctd_itran             = lat_ini - n * sep_trans_lat_deg
        lat_uctd_pfs[cond_tran_n]  = lat_uctd_itran        
        
        # longitude of profiles in this transect

        if n in np.arange(0, num_trans, 2):
            
            # distance starting from 0 to 80 of this transect
            
            rel_dist_tran = dist_pfs_tran_n - dist_ini_tran_n
            
            
            # min dist of this transect
            
            min_dist_km      = rel_dist_tran.min()
            min_dist_lon_deg = min_dist_km/(length_deg_lon/1000)
            
            # longitude of the western profile
            
            lon_ini_tran     = lon_ini + min_dist_lon_deg
            
            # max dist of this transect
            
            max_dist_km      = rel_dist_tran.max() 
            max_dist_lon_deg = max_dist_km/(length_deg_lon/1000)
            
            # longitude of the eastern profile
            
            lon_fin_tran     = lon_ini + max_dist_lon_deg            
            
            
            # longitude of all profiles between first and last profile in transet
            lon_uctd_itran   = np.arange(lon_ini_tran, lon_fin_tran+0.5*dlon_uctd, dlon_uctd)
            
            
            # save longitude of the profiles in this transect
            lon_uctd_pfs[cond_tran_n] = lon_uctd_itran
  

        elif n in np.arange(1, num_trans, 2):
            
            # distance starting from 0 to 80 of this transect
            # 0 km is east and 80 km is west (uCTD goes from east to west)
            
            rel_dist_tran = dist_pfs_tran_n - dist_ini_tran_n
            
            
            # min dist of this transect
            
            min_dist_km      = rel_dist_tran.min()
            min_dist_lon_deg = min_dist_km/(length_deg_lon/1000)
            
            # longitude of the EASTERN profile
            
            lon_ini_tran     = lon_fin - min_dist_lon_deg
            
            # max dist of this transect
            
            max_dist_km      = rel_dist_tran.max() 
            max_dist_lon_deg = max_dist_km/(length_deg_lon/1000)
            
            # longitude of the WESTERN profile
            
            lon_fin_tran     = lon_fin - max_dist_lon_deg            
            
            
            # longitude of all profiles between first and last profile in transet
            lon_uctd_itran   = np.arange(lon_ini_tran, lon_fin_tran-0.5*dlon_uctd, -dlon_uctd)
            
            
            # save longitude of the profiles in this transect
            lon_uctd_pfs[cond_tran_n] = lon_uctd_itran
            
        # distance of the profiles in each separation between transects
            
        if n < num_trans-1:
            dist_ini_sep_n  = dist_fin_tran_n
            dist_fin_sep_n  = dist_ini_sep_n + sep_trans # or (n+1) * (len_trans + sep_trans)) 
        
            cond_sep_n  =  np.logical_and(dist_pfs_all > dist_ini_sep_n, 
                                      dist_pfs_all < dist_fin_sep_n) 
            
            dist_pfs_sep_n  = dist_pfs_all[cond_sep_n] # save distance
            
            # Which are the coordinates of these profiles?
        
            # longitude of profiles in this transect
            
            if n in np.arange(0, num_trans, 2):

                lon_uctd_pfs[cond_sep_n]  = lon_fin

                 
            elif n in np.arange(1, num_trans, 2):    
                lon_uctd_pfs[cond_sep_n]  = lon_ini                
                
 
            # latitude of profiles in this transect
                
            # lat_uctd_itran + distance    
            
            dist_sep_pf_km      = dist_pfs_sep_n - dist_ini_sep_n
            dist_sep_pf_lat_deg = dist_sep_pf_km / (length_deg_lat/1000)   

            lat_uctd_pfs[cond_sep_n] = lat_uctd_itran - dist_sep_pf_lat_deg
            
            
    return lon_uctd_pfs, lat_uctd_pfs
  


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
        
        
        
def make_figure_UCTD(bm, region, lon_topo2d=None, lat_topo2d=None, topo_dom=None,
                    lon_mdt2d=None, lat_mdt2d=None, ug=None, vg=None):    

    xsw1, ysw1  = bm(lonsw1, latsw1)
    xsw2, ysw2  = bm(lonsw2, latsw2)
    xnd1, ynd1  = bm(lonnd1, latnd1)
    xnd2, ynd2  = bm(lonnd2, latnd2)

    tz = 12    

    x_casts, y_casts = bm(lon_uctd_pfs, lat_uctd_pfs)
    # xlimmin, ylimmin = bm(1, 39)
    # xlimmax, ylimmax = bm(4, 42)
    
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
    
    plt.title(name_scenario  + ' >> uCTD')
    
    ax2=plt.subplot(122)
    plt.scatter(lon_uctd_pfs, lat_uctd_pfs, c=np.arange(len(lon_uctd_pfs)), cmap=plt.cm.Spectral_r)
    
    if dx_uctd_km > 5: 
        for i, txt in enumerate(np.arange(len(lon_uctd_pfs))+1):
            plt.annotate(txt, (lon_uctd_pfs[i]+0.005, lat_uctd_pfs[i]+0.01))

    # for i, txt_time in enumerate(time_casts):
    #     plt.annotate(mdates.num2date(txt_time).strftime("%H:%M %d-%m "), 
    #                  (lon_casts[i]-0.5, lat_casts[i]+0.5))  
    for j, txt_time in enumerate(time_uctd_pfs):
        # plt.annotate(mdates.num2date(txt_time).strftime("%H:%M %d-%m "), 
        #              (lon_casts[j]-0.03, lat_casts[j]+0.03))  
        #if np.any(j==np.arange(0, len(time_uctd_pfs), 5)):
        if j==0 or j==len(time_uctd_pfs)-1:    
            plt.text(lon_uctd_pfs[j]-0.03, lat_uctd_pfs[j]+0.035, 
                     mdates.num2date(txt_time).strftime("%H:%M %d-%m "),
                     fontsize=7)
        
    plt.axis('image')   
    plt.ylim(lat_uctd_pfs.min()-0.1, lat_uctd_pfs.max()+0.1)      
    
    plt.title('uCTD')    
    
    plt.tight_layout()
    fig.savefig(fig_dir + name_scenario+'_uCTD.png')

def make_figure_ADCP_uCTD(bm):

    xsw1, ysw1  = bm(lonsw1, latsw1)
    xsw2, ysw2  = bm(lonsw2, latsw2)
    xnd1, ynd1  = bm(lonnd1, latnd1)
    xnd2, ynd2  = bm(lonnd2, latnd2)

    tz = 12    

    x_adcp, y_adcp = bm(lon_adcp_pfs, lat_adcp_pfs)

    
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

    plt.scatter(x_adcp, y_adcp, c=time_adcp_pfs, s=3,  cmap=plt.cm.Spectral_r)
    
    # plt.xlim(xlimmin, xlimmax)
    # plt.ylim(ylimmin, ylimmax)
    
    plt.title(name_scenario  + ' >> ADCP')
    
    ax2=plt.subplot(122)
    
    # ADCP
    plt.scatter(lon_adcp_pfs, lat_adcp_pfs, c=time_adcp_pfs, s=8, cmap=plt.cm.Spectral_r)
    
    plt.colorbar(ticks=mdates.HourLocator(interval=6), #DayLocator(interval=1), 
                  format=mdates.DateFormatter('%H:%M \n %b %d'), 
                  orientation='horizontal')
    # uCTD casts
    plt.scatter(lon_uctd_pfs, lat_uctd_pfs, c='k', s=8, marker='s')

    if dx_uctd_km > 5: 
        for i, txt in enumerate(np.arange(len(lon_uctd_pfs))+1):
            plt.annotate(txt, (lon_uctd_pfs[i]+0.005, lat_uctd_pfs[i]+0.01))
        
    # for i, txt in enumerate(np.arange(len(lon_uctd_pfs))+1):
    #     plt.annotate(txt, (lon_uctd_pfs[i]+0.005, lat_uctd_pfs[i]+0.01))
        
    plt.axis('image')   
    plt.ylim(lat_uctd_pfs.min()-0.1, lat_uctd_pfs.max()+0.1)     
      
    plt.title('uCTD + ADCP')    
    
    plt.tight_layout()
    fig.savefig(fig_dir + name_scenario+'_ADCP.png')


    
if __name__ == '__main__':

    plt.close('all')
    
    fig_dir = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/configurations/'
    dir_dic = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/configurations/'
    
    ''' Which region? '''
    
    region = 'Med' # options 'Atl' or 'Med'
    
    
    ''' 
    Which subconfiguration?  
    
    Configuration 3a:
        - Horizontal resolution of uCTD profiles: 6 km
        - Vertical axis: from 5 to 500 m depth, with a vertical spacing of 0.5 m
        
    Configuration 3b (similar to Calypso):
        - Horizontal resolution of uCTD profiles: 2.5 km
        - Vertical axis: from 5 to 200 m depth, with a vertical spacing of 0.5 m
    
    '''
    
    subconf = '3b' # Options: '3a' and '3b'
    
    

    '''
    >>>>>> 1) Simulate uCTD/Seasoar sampling strategy <<<<<<
    
    To obtain lon, lat and time of each uCTD profile.
    
    Note 1: We assume a regular grid. 
            num_trans: number of transects
            sep_trans: separation between transects
    
    Note 2: Order: Row 1 (top) from left to right, row 2 from right to left, etc.
             
    Note 3: We assume one vertical profile every dx_uctd_km 
    
    Note 4: We assume constant ship velocity, including in turns. And turns of 90 deg.
    '''   
    

    
    # define start time of the sampling 
    t_samp_ini    = mdates.date2num(datetime(2009, 9, 1, 0, 0, 0)) #[python format in num. of days]
    
    # Parameters to define the uCTD sampling strategy (change as you wish)
    
    ship_speed_kt   = 8  # knots
    
    len_trans       = 80   # km, length of each zonal transect
    sep_trans       = 10   # km, separation between zonal transects
    num_trans       = 7    # total number of transects
    
    dep_uctd_min = 5   # m upper depth layer
    dep_uctd_res = 0.5    # m vertical resolution of the data
    
    if subconf == '3a':
        
      dx_uctd_km      = 6    #  [km] resolution of the uCTD profiles   
      dep_uctd_max    = 500  #  [m]  maximum depth of each cast  

    elif subconf == '3b':

      dx_uctd_km      = 2.5  #  [km] resolution of the uCTD profiles  
      dep_uctd_max    = 200  #  [m]  maximum depth of each cast    


      
    # number and name of this configuration
    
    name_scenario = region + '_conf_3_dep_'+'%04i'% dep_uctd_max+'m_res_'+'%04.1f'% dx_uctd_km + 'km_Sep'
      
    # ADCP parameters (change as you wish)
      
    dt_adcp_min     = 5 #one profile every 5 minutes, in PRE-SWOT 468 seconds (7.8 min)
    dep_adcp_min    = 20 #m, in PRE-SWOT 19.96 m
    dep_adcp_max    = 600 #m, in PRE-SWOT 587.96 M
    dep_adcp_res    = 8 #m # bin size of the ADCP data
    
    #define coordinates of profile 1 (top, left cast)

    if region == 'Med':  
        
        lon_ini = 1.45
        lat_ini = 40.415     

    elif region == 'Atl':  
        
        lon_ini = -48.7
        lat_ini = 35.25    

        
    # >>>> Computations needed <<<<<

    # length of a degree of longitude and latitude
    length_deg_lon, length_deg_lat = to.length_lon_lat_degs(lat_ini)   

    # beginning of 1st transect: lon_ini
    # end of 1st transect: lon_fin
    len_trans_lon_deg = len_trans/(length_deg_lon/1000)
    lon_fin           = lon_ini + len_trans_lon_deg
    
    # separation between transects in latitude degrees  
    sep_trans_lat_deg = sep_trans /(length_deg_lat/1000)


    # ship velocity from knots to m/s
    kt2ms          = 0.514444 # 1 knot is 0.514444 m/s
    ship_speed_ms  = ship_speed_kt * kt2ms

    # ADCP temporal resolution    
    dt_adcp_s        = dt_adcp_min * 60 # in seconds 
    dt_adcp_fracdays = dt_adcp_min/(60*24)
    
    # ADCP profile resolution
    dx_adcp_km       = (ship_speed_ms*dt_adcp_s)/1000 #in km
    
    
    # >>>> Depth axis: uCTD and ADCP (save) <<<<<
    
    # define depth axis for uCTD profiles
    dep_uctd = np.arange(dep_uctd_min, dep_uctd_max+dep_uctd_res, dep_uctd_res)

    # define depth axis for ADCP profiles
    dep_adcp = np.arange(dep_adcp_min, dep_adcp_max+1, dep_adcp_res)
    


    # >>>> Distance axis: uCTD <<<<<
    
    # total distance of the uCTD trajectory 
    tot_dist_uCTD = num_trans * len_trans + (num_trans - 1) * sep_trans
    
    # distance of each profile from the beginning of the uCTD trajectory
    dist_pfs_uctd_all  = np.arange(0, tot_dist_uCTD, dx_uctd_km)

    # uCTD profile resolution in longitude degrees
    dlon_uctd = dx_uctd_km /(length_deg_lon/1000)    
        
    

    # >>>> Time axis: uCTD <<<<<
    
    # time resolution of uCTD profiles 
    dt_uctd_s      = dx_uctd_km*1000/ship_speed_ms  #[s]
    dt_uctd_min    = dt_uctd_s/60  # [min]
    
    # time resolution of uCTD profiles in days
    dt_uctd_fracdays = dt_uctd_min/(60*24)    #as fraction of days
    
    # time needed for the uCTD sampling  
    time_sampling_s = tot_dist_uCTD*1000/ship_speed_ms  #[s]
    time_sampling_days = time_sampling_s/(60*60*24)    
    
    print('')
    print('Time needed to do the uCTD sampling (hours)...', time_sampling_s/3600)
    print('')
    
    # date of each uCTD profile (save)
    time_uctd_pfs = np.arange(t_samp_ini, t_samp_ini+time_sampling_days, dt_uctd_fracdays)
    
    

    # >>>> Distance axis: ADCP <<<<<
 
    # distance of each ADCP profile from the beginning of the uCTD trajectory
    dist_pfs_adcp_all  = np.arange(0, tot_dist_uCTD, dx_adcp_km)

    # ADCP profile resolution in longitude degrees
    dlon_adcp = dx_adcp_km /(length_deg_lon/1000)  
    
    
    # >>>> Time axis: ADCP <<<<<
        
    # date of each ADCP profile (save)
    time_adcp_pfs = np.arange(t_samp_ini, t_samp_ini+time_sampling_days, dt_adcp_fracdays)
    

    
    # -------------------------------------------------------------------
    # Extract (lon, lat) of each uCTD profile
    # Input: dist_pfs_uctd_all, dlon_uctd
    
    lon_uctd_pfs, lat_uctd_pfs = extract_lon_lat_continuous_profiles(
                                        dist_pfs_uctd_all, dlon_uctd, len_trans, sep_trans,
                                        sep_trans_lat_deg, lat_ini, lon_ini, 
                                        lon_fin, length_deg_lon, length_deg_lat,
                                        num_trans)
    
    # Now we have (time_uctd_pfs, lon_uctd_pfs, lat_uctd_pfs, dep_uctd) 
    # for this uCTD/seasoar sampling strategy.
    # Save these data to extract the model data at this location in another code.
    
    # -------------------------------------------------------------------
    # Extract (lon, lat) of each ADCP profile
    # Input: dist_pfs_adcp_all, dlon_adcp
    
    lon_adcp_pfs, lat_adcp_pfs = extract_lon_lat_continuous_profiles(
                                        dist_pfs_adcp_all, dlon_adcp, len_trans, sep_trans,
                                        sep_trans_lat_deg, lat_ini, lon_ini, 
                                        lon_fin, length_deg_lon, length_deg_lat,
                                        num_trans)
    
    # Now we have (time_adcp_pfs, lon_adcp_pfs, lat_adcp_pfs, dep_adcp) 
    # for this uCTD/seasoar sampling strategy.
    # Save these data to extract the model data at this location in another code.
    
    # -------------------------------------------------------------------

    
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
        make_figure_UCTD(bm, region, lon_topo2d, lat_topo2d, topo_dom,
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


    
        make_figure_UCTD(bm, region)
    
    
    

    ''' Plot ADCP sampling strategy + swaths of SWOT '''    
    
    make_figure_ADCP_uCTD(bm)
    

    
    ''' Save (time, lon, lat, dep) of the uCTD and ADCP profiles '''
    
    dic_uctd  = {}
    dic_adcp = {}
    
    # (time_uctd_pfs, lon_uctd_pfs, lat_uctd_pfs, dep_uctd) 
    dic_uctd.update({'time_ctd': time_uctd_pfs, 
                    'lon_ctd' : lon_uctd_pfs,
                    'lat_ctd' : lat_uctd_pfs,
                    'dep_ctd' : dep_uctd}) 
    
    # time_adcp_pf, lon_adcp_pf, lat_adcp_pf, dep_adcp_pf

    dic_adcp.update({'time_adcp': time_adcp_pfs, 
                    'lon_adcp'  : lon_adcp_pfs,
                    'lat_adcp'  : lat_adcp_pfs,
                    'dep_adcp'  : dep_adcp}) 
    
    f_uctd = open(dir_dic + name_scenario + '_uctd.pkl','wb')
    pickle.dump(dic_uctd,f_uctd)
    f_uctd.close()         
    
    f_adcp = open(dir_dic + name_scenario + '_adcp.pkl','wb')
    pickle.dump(dic_adcp,f_adcp)
    f_adcp.close()       
    
    
     
        