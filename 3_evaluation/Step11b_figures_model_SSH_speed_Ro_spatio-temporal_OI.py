#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import matplotlib.gridspec   as gridspec
import pickle
import cartopy.crs           as ccrs   # import projections
import cartopy.feature       as cf     # import features
from cartopy.mpl.gridliner   import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean               as cmo

"""
Step 11b
    
Plot for each region and model, and for each configuration: 
    SSH, speed, Ro
    for comparison with reconstructed fields.
 

written by Bàrbara Barceló-Llull on 11-04-2022 at IMEDEA (Mallorca, Spain)

"""

def plot_decor_cartopy(region, ax, fsize, lon, lat, num_conf):
        
        if region == 'Med':
            
            plot_decor_Med_cartopy(ax, fsize, lon, lat, num_conf)
            
        elif region == 'Atl':
            
            plot_decor_Atl_cartopy(ax, fsize, lon, lat, num_conf)
            
def plot_decor_Atl_cartopy(ax, fsize, lon, lat, num_conf):
    
    # plt.figure()
    # ax = plt.subplot(projection=ccrs.Mercator(central_longitude=0))
    # decor map
    ax.coastlines(resolution='10m')
    ax.add_feature(cf.LAND, facecolor='0.7')
    parallels = np.arange(30.,40.,1)
    meridians = np.arange(-50,-45.,1)

    gl = ax.gridlines(draw_labels=True, xlocs = meridians, ylocs=parallels,
                  crs=ccrs.PlateCarree(), linestyle='-', linewidth=0.15, color='grey')#, alpha=0.5, linestyle='--'
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': fsize-2}
    gl.ylabel_style = {'size': fsize-2}
    
    # ax.set_extent([lon.min()-0.05, lon.max()+0.05, 
    #                lat.min()-0.05, lat.max()+0.05], crs=ccrs.PlateCarree())
    
    ''' limits of configuration... 2 (15km 1000m)
    lon min =  -48.91
    lon max =  -47.59843
    lat min =  34.538803
    lat max =  35.34936 
    '''
    # all subplots the same limits
    
    plon_min =  -48.91
    plon_max =  -47.59843
    plat_min =  34.538803
    plat_max =  35.34936 
    sep_plot = 0.025
    ax.set_extent([plon_min-sep_plot, plon_max+sep_plot, 
                   plat_min-sep_plot, plat_max+sep_plot], crs=ccrs.PlateCarree())
            
def plot_decor_Med_cartopy(ax, fsize, lon, lat, num_conf):
    
    # decor map
    ax.coastlines(resolution='10m')
    ax.add_feature(cf.LAND, facecolor='0.7')
    parallels = np.arange(34.,43.,1)
    meridians = np.arange(-6,7.,1)
    

    gl = ax.gridlines(draw_labels=True, xlocs = meridians, ylocs=parallels,
                  crs=ccrs.PlateCarree(), linestyle='-', linewidth=0.15, color='grey')#, alpha=0.5, linestyle='--'
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': fsize-2}
    gl.ylabel_style = {'size': fsize-2}

    ''' limits of configuration... r (10km 1000m)
    lon min =  1.45
    lon max =  2.3868356
    lat min =  39.87467
    lat max =  40.397026 
    '''
    # all subplots the same limits
    
    plon_min =  1.45
    plon_max =  2.3868356
    plat_min =  39.87467
    plat_max =  40.397026 
    sep_plot = 0.025
    ax.set_extent([plon_min-sep_plot, plon_max+sep_plot, 
                   plat_min-sep_plot, plat_max+sep_plot], crs=ccrs.PlateCarree())

def define_set_of_axis(fig, gs):    
    set_of_axis = [
        
        fig.add_subplot(gs[0,1], projection=ccrs.Mercator(central_longitude=0)), #1
        fig.add_subplot(gs[0,2], projection=ccrs.Mercator(central_longitude=0)), #2a
        fig.add_subplot(gs[1,0], projection=ccrs.Mercator(central_longitude=0)), #2b
        fig.add_subplot(gs[1,1], projection=ccrs.Mercator(central_longitude=0)), #2c
        fig.add_subplot(gs[1,2], projection=ccrs.Mercator(central_longitude=0)), #2d
        fig.add_subplot(gs[2,1], projection=ccrs.Mercator(central_longitude=0)), #3b
        fig.add_subplot(gs[2,0], projection=ccrs.Mercator(central_longitude=0)), #3a
        fig.add_subplot(gs[3,0], projection=ccrs.Mercator(central_longitude=0)), #4
        fig.add_subplot(gs[2,2], projection=ccrs.Mercator(central_longitude=0)), #5
        fig.add_subplot(gs[0,0], projection=ccrs.Mercator(central_longitude=0))  #r
       ] 
    
    return set_of_axis

    
def subplot_SSH(time_map, SSHavg):

        if icf == 0:
                figDH.suptitle(region + ' ' + 
                             model + ' - ocean truth SSH anomaly [m]',
                      fontsize=fsize+2)
        
        axDH2 = set_of_axisDH[icf]

        plot_decor_cartopy(region, axDH2, fsize, lon, lat, num_conf)
      
        
        if region == 'Atl':

          if model == 'eNATL60':
              
            if num_conf != '4':
                extdh   = 0.24
                levdh   = np.linspace(-extdh, extdh, 13)
            else: 
                extdh   = 0.16
                levdh   = np.linspace(-extdh, extdh,17)

          else:
            if num_conf != '4':
                extdh   = 0.07 #max(np.nanmax(dhanom), np.abs(np.nanmin(dhanom)))
                levdh   = np.linspace(-extdh, extdh, 15)
            else: 
                extdh   = 0.045
                levdh   = np.linspace(-extdh, extdh,19)

            
        elif region == 'Med':
            
          if model == 'CMEMS':
            if num_conf != '4':
                extdh   = 0.06
                levdh   = np.linspace(-extdh, extdh, 13)
            else: 
                extdh   = 0.11
                levdh   = np.linspace(-extdh, extdh,12)

          elif model == 'WMOP':
            if num_conf != '4':
                extdh   = 0.08
                levdh   = np.linspace(-extdh, extdh, 17)
            else: 
                extdh   = 0.06
                levdh   = np.linspace(-extdh, extdh,13)

          elif model == 'eNATL60':
            if num_conf != '4':
                extdh   = 0.11
                levdh   = np.linspace(-extdh, extdh, 12)
            else: 
                extdh   = 0.1
                levdh   = np.linspace(-extdh, extdh,11)


        cf2 = axDH2.contourf(lon, lat, SSHavg - SSHavg.mean(), 
                           levels=levdh, #(-extdh, extdh, 10),
                           cmap = cmo.cm.balance, 
                         transform=ccrs.PlateCarree()) 
                        
        axDH2.set_title(compl_name_conf, fontsize=fsize)
          
        # Colorbars
        if num_conf == 'r':
            axcb2 = figDH.add_subplot(gsDH[0:3, 3])
            cb2 = plt.colorbar(cf2, cax=axcb2, orientation='vertical',
                            ticks=cf2.levels[::2])
            cb2.ax.tick_params(labelsize=fsize)
        
        elif num_conf == '4':
            axcb2 = figDH.add_subplot(gsDH[3, 3])
            cb2 = plt.colorbar(cf2, cax=axcb2, orientation='vertical',
                            ticks=cf2.levels[::4])
            cb2.ax.tick_params(labelsize=fsize)     


            
def subplot_SP(time_map, speed_avg):

        if icf == 0:
                figSP.suptitle(region + ' ' + 
                             model + ' - ocean truth horizontal velocity magnitude [m/s]',
                      fontsize=fsize+2)
        
        axSP2 = set_of_axisSP[icf]

        plot_decor_cartopy(region, axSP2, fsize, lon, lat, num_conf)
      
        
        if region == 'Atl':

          if model == 'eNATL60':
              
            if num_conf != '4':
                levSP = np.arange(0,0.6,0.05) 
            else: 
                levSP = np.arange(0,0.8,0.05) 

          elif model == 'CMEMS':
              
            if num_conf != '4':
                levSP = np.arange(0,0.32,0.02) 
            else: 
                levSP = np.arange(0,0.22,0.02) 

            
        elif region == 'Med':
            
          if model == 'CMEMS':
            if num_conf != '4':
                levSP = np.arange(0,0.44,0.02) #np.arange(0,0.34,0.02) 
            else: 
                levSP = np.arange(0,0.5,0.04) #np.arange(0,0.38,0.02)  

          elif model == 'WMOP':
            if num_conf != '4':
                levSP = np.arange(0,0.44,0.02) #(0,0.36,0.02)  
            else: 
                levSP = np.arange(0,0.42,0.02) #(0,0.34,0.02)  

          elif model == 'eNATL60':
            if num_conf != '4':
                levSP = np.arange(0,0.46,0.02) #(0,0.48,0.04)
            else: 
                levSP = np.arange(0,0.54,0.04) 
        
                
        cf2 = axSP2.contourf(lon, lat, speed_avg, 
                           levels=levSP, #(-extdh, extdh, 10),
                           cmap = plt.cm.Spectral_r, 
                         transform=ccrs.PlateCarree()) 

        axSP2.set_title(compl_name_conf, fontsize=fsize)
          
        # Colorbars
        if num_conf == 'r':
            axcb2 = figSP.add_subplot(gsSP[0:3, 3])
            cb2 = plt.colorbar(cf2, cax=axcb2, orientation='vertical',
                            ticks=cf2.levels[::2])
            cb2.ax.tick_params(labelsize=fsize)
        
        elif num_conf == '4':
            axcb2 = figSP.add_subplot(gsSP[3, 3])
            cb2 = plt.colorbar(cf2, cax=axcb2, orientation='vertical',
                            ticks=cf2.levels[::4])
            cb2.ax.tick_params(labelsize=fsize)  



def subplot_RO(time_map, Rog):

        if icf == 0:
                figRO.suptitle('('+ region + '  ' + 
                             model + '  averaged) ' + 'Ro',
                      fontsize=fsize+2)
        
        axRO2 = set_of_axisRO[icf]

        plot_decor_cartopy(region, axRO2, fsize, lon, lat, num_conf)
        

        levRog = np.linspace(-0.3,0.3, 16)    

        if region == 'Atl':

          if model == 'eNATL60':
              
            if num_conf != '4':
                extRog   = 0.3
                levRog   = np.linspace(-extRog, extRog, 16)
            else: 
                extRog   = 0.7
                levRog   = np.linspace(-extRog, extRog, 15)

          else:
            if num_conf != '4':
                extRog   = 0.3
                levRog   = np.linspace(-extRog, extRog, 16)
            else: 
                extRog   = 0.3
                levRog   = np.linspace(-extRog, extRog, 16)

            
        elif region == 'Med':
            
          if model == 'CMEMS':
            if num_conf != '4':
                extRog   = 0.24
                levRog   = np.linspace(-extRog, extRog, 17)
            else: 
                extRog   = 0.34
                levRog   = np.linspace(-extRog, extRog, 18)

          elif model == 'WMOP':
            if num_conf != '4':
                extRog   = 0.48
                levRog   = np.linspace(-extRog, extRog, 13)
            else: 
                extRog   = 0.4
                levRog   = np.linspace(-extRog, extRog, 17)

          elif model == 'eNATL60':
            if num_conf != '4':
                extRog   = 0.26
                levRog   = np.linspace(-extRog, extRog, 14)
            else: 
                extRog   = 0.6
                levRog   = np.linspace(-extRog, extRog, 13)
        
        
        cf2 = axRO2.contourf(lon, lat, Rog, 
                           levels=levRog, #(-extdh, extdh, 10),
                           cmap = cmo.cm.balance, 
                         transform=ccrs.PlateCarree()) 
        
        axRO2.set_title(compl_name_conf, fontsize=fsize)
          
        # Colorbars
        if num_conf == 'r':
            axcb2 = figRO.add_subplot(gsRO[0:3, 3])
            cb2 = plt.colorbar(cf2, cax=axcb2, orientation='vertical',
                            ticks=cf2.levels[::2])
            cb2.ax.tick_params(labelsize=fsize)
        
        elif num_conf == '4':
            axcb2 = figRO.add_subplot(gsRO[3, 3])
            cb2 = plt.colorbar(cf2, cax=axcb2, orientation='vertical',
                            ticks=cf2.levels[::4])
            cb2.ax.tick_params(labelsize=fsize)     


            
if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directories '''
    
    dir_OIdata    = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/reconstructed_fields/spatio-temporal_OI_all_conf/'
    dir_pseudoobs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'
    dir_figures   = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/reconstructed_fields/spatio-temporal_OI/'
    dir_dic       = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/comparison/spatio-temporal_OI/'
    
    ''' Which model and region? '''
    
    model         = 'CMEMS' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Atl' #'Atl' or 'Med'
    
    
    ''' Open model data ("ocean truth") '''
    
    f_model    = open(dir_dic + region + '_' + model + '_ocean_truth_SSH_speed_Ro.pkl','rb')
    model_data = pickle.load(f_model)
    f_model.close()         
    
    keys = model_data.keys()
    
             
    ''' Start figures - One figure for each region, model and variable '''
        
    fsize = 12
        
    if region == 'Med':
        figDH = plt.figure(figsize=(8,12))
        figSP = plt.figure(figsize=(8,12))
        figRO = plt.figure(figsize=(8,12))
        
        
    elif region == 'Atl':
        figDH = plt.figure(figsize=(8,12))
        figSP = plt.figure(figsize=(8,12))
        figRO = plt.figure(figsize=(8,12))
       

    gsDH = gridspec.GridSpec(4, 4, width_ratios=[20,20,20, 1], figure=figDH)
    gsSP = gridspec.GridSpec(4, 4, width_ratios=[20,20,20, 1], figure=figSP)
    gsRO = gridspec.GridSpec(4, 4, width_ratios=[20,20,20, 1], figure=figRO)
    
    
    set_of_axisDH = define_set_of_axis(figDH, gsDH)    
    set_of_axisSP = define_set_of_axis(figSP, gsSP)
    set_of_axisRO = define_set_of_axis(figRO, gsRO)
    
 
    '''' Make figures ''' 
    titles_subplots = ['', 'a', 'b', 'c', 'd', 'b', 'a', '', '', '',]
    
    for icf, key in enumerate(keys):
        name_conf = key
        num_conf  = key[9]
      
        compl_name_conf = num_conf + titles_subplots[icf] + \
                        ' (' + name_conf[25:29] + \
                        ' ' + name_conf[15:20] + ')'
 
        if num_conf == '3':                

            compl_name_conf = num_conf + titles_subplots[icf] +\
                        ' (' + name_conf[25:31] + \
                        ' ' + name_conf[15:20] + ')'   
                        
        print('')
        print('--------------------------------------')
        print('')
        print('Configuration file...', name_conf)
        print('')
        
        print('configuration...', num_conf)  
        print(compl_name_conf)                  

        # Open averaged model data for this configuration
        
        lon        = model_data[key]['lon']
        lat        = model_data[key]['lat']
        time_map   = model_data[key]['time_map']
        ssh_int    = model_data[key]['ssh_int']
        # u_int       = model_data[key]['u_int']
        # v_int       = model_data[key]['v_int']
        speed_int  = model_data[key]['speed_int']
        Ro_int     = model_data[key]['Ro_int']
                 
        ''' Make a figure for each configuration '''

        # Make subplot DH
        subplot_SSH(time_map, ssh_int)
        
        # Make subplot SPEED
        subplot_SP(time_map, speed_int)
        
        # Make subplot Ro
        subplot_RO(time_map, Ro_int)

      
     
    # Save figure DH    
    figDH.canvas.draw()
    figDH.tight_layout(rect=[0, 0.03, 1, 0.95])
    figDH.savefig(dir_figures + 'DH_comp_' + region + '_' + model + '_model_stOI_cd.png', dpi=500)    
     
    # Save figure SPEED    
    figSP.canvas.draw()
    figSP.tight_layout(rect=[0, 0.03, 1, 0.95])
    figSP.savefig(dir_figures + 'Sp_comp_' + region + '_' + model + '_model_stOI_cd.png', dpi=500)           

    # Save figure Ro    
    figRO.canvas.draw()
    figRO.tight_layout(rect=[0, 0.03, 1, 0.95])
    figRO.savefig(dir_figures + 'Ro_comp_' + region + '_' + model + '_model_stOI_cd.png', dpi=500)    
    


