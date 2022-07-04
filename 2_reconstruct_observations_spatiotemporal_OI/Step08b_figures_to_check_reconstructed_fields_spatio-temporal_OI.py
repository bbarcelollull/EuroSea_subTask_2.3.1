#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import matplotlib.gridspec   as gridspec
import netCDF4               as netcdf
from matplotlib              import dates as mdates
import cartopy.crs           as ccrs   # import projections
import cartopy.feature       as cf     # import features
from cartopy.mpl.gridliner   import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean               as cmo
from matplotlib.colors       import BoundaryNorm, ListedColormap

"""
Step 8b

Plot for each configuration: 
    T (with observations) + S (with observations) + dh+vgeo
     
Reconstructed fields with the spatio-temporal OI

written by Bàrbara Barceló-Llull on 05-04-2022 at IMEDEA (Mallorca, Spain)

"""

def plot_decor_cartopy(region, ax, fsize, lon, lat):
        
        if region == 'Med':
            
            plot_decor_Med_cartopy(ax, fsize, lon, lat)
            
        elif region == 'Atl':
            
            plot_decor_Atl_cartopy(ax, fsize, lon, lat)
            
def plot_decor_Atl_cartopy(ax, fsize, lon, lat):
    
    # decor map
    ax.coastlines(resolution='10m')
    ax.add_feature(cf.LAND, facecolor='0.7')
    parallels = np.arange(34.,36.,0.5)
    meridians = np.arange(-50,-45.,0.5)
    gl = ax.gridlines(draw_labels=True, xlocs = meridians, ylocs=parallels,
                  crs=ccrs.PlateCarree(),)# linestyle='--') #linewidth=2, color='gray', alpha=0.5, linestyle='--'
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': fsize}
    gl.ylabel_style = {'size': fsize}
    
    ax.set_extent([lon.min()-0.05, lon.max()+0.05, 
                   lat.min()-0.05, lat.max()+0.05], crs=ccrs.PlateCarree())
            
def plot_decor_Med_cartopy(ax, fsize, lon, lat):
    
    # decor map
    ax.coastlines(resolution='10m')
    ax.add_feature(cf.LAND, facecolor='0.7')
    parallels = np.arange(34.,43.,0.5)
    meridians = np.arange(-6,7.,0.5)
    gl = ax.gridlines(draw_labels=True, xlocs = meridians, ylocs=parallels,
                  crs=ccrs.PlateCarree(),)# linestyle='--') #linewidth=2, color='gray', alpha=0.5, linestyle='--'
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': fsize}
    gl.ylabel_style = {'size': fsize}
    
    ax.set_extent([lon.min()-0.05, lon.max()+0.05, 
                   lat.min()-0.05, lat.max()+0.05], crs=ccrs.PlateCarree())


if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directories '''
    
    dir_OIdata    = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/reconstructed_fields/spatio-temporal_OI_all_conf/'
    dir_pseudoobs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'
    dir_figures   = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/reconstructed_fields/spatio-temporal_OI/'


    ''' Which model and region? '''
    
    model         = 'eNATL60' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Med' #'Atl' or 'Med'
    lev           = 5 #5 #100 #960 #m
            
    
    '''
    >>>>>> Interpolated fields <<<<<<
    '''
    
    oi_files  = sorted(glob.glob(dir_OIdata + region + '*_'+model + '*T.nc'))
        
    
    for file in [oi_files[-1]]: #oi_files: 
        

        name_conf  = file[96:-5]
        num_conf   = name_conf[:33] 
        psobs_file =  name_conf[:33] + '_' + model + '.nc'
        
        print('')
        print('--------------------------------------')
        print('')
        print('Configuration file...', name_conf)
        print('')
        
        print('configuration...', num_conf)
        
        
    
        ''' Read pseudo-observations '''
        
        ncpo        = netcdf.Dataset(dir_pseudoobs + psobs_file, 'r')
    
        print('opening CTD pseudo-observations')
        time_ctd  = ncpo.variables['time_ctd'][:].data   
        dep_ctd   = ncpo.variables['dep_ctd'][:].data 
        lat_ctd   = ncpo.variables['lat_ctd'][:].data   
        lon_ctd   = ncpo.variables['lon_ctd'][:].data 
    
        tem_ctd   = ncpo.variables['tem_ctd'][:].data      
        sal_ctd   = ncpo.variables['sal_ctd'][:].data     

        ncpo.close()
        
        ''' Read .nc file with interpolated fields '''
           
        ncT      = netcdf.Dataset(dir_OIdata + name_conf + '_T.nc', 'r')
        ptem     = ncT.variables['ptem'][:].data #'ptem_mk'][:].data 
        eptem    = ncT.variables['error'][:].data    
        lon      = ncT.variables['longitude'][:].data
        lat      = ncT.variables['latitude'][:].data   
        dep      = ncT.variables['depth'][:].data 
        time_map = ncT.variables['time'][:].data 
        ncT.close() 
        
        ncS      = netcdf.Dataset(dir_OIdata + name_conf + '_S.nc', 'r')
        psal     = ncS.variables['psal'][:].data  #'psal_mk'][:].data 
        epsal    = ncS.variables['error'][:].data          
        ncS.close() 
        

        
        ''' dh and derived variables file '''
        
        filedh  = name_conf + '_derived_variables.nc'
        
        ncdh   = netcdf.Dataset(dir_OIdata + filedh, 'r')
        sig    = ncdh.variables['sig'][:].data          
        dh     = ncdh.variables['dh'][:].data   
        ug     = ncdh.variables['ug'][:].data  
        vg     = ncdh.variables['vg'][:].data  
        Rog    = ncdh.variables['Rog'][:].data  
        N      = ncdh.variables['N'][:].data 
        ncdh.close() 


        
        ''' Make figure to check dh computation '''
        
        
        iz     = np.argmin(np.abs(dep-lev))
        izctd  = np.argmin(np.abs(dep_ctd-lev))

        
        print('Depth', dep[iz])        
        print('Depth CTD obs', dep_ctd[izctd])
        
        fsize = 12
        
        fig = plt.figure(figsize=(15,5))

        #fig.suptitle(region + ' ' + model + ' conf=' + num_conf, fontsize=fsize+2)
        #fig.suptitle(name_conf +  '   dep = ' + np.str(dep[iz]) + ' m', fontsize=fsize+1)
        fig.suptitle(region + '  ' + model + '  ' +  num_conf + '  ' +\
                     mdates.num2date(time_map[0]).strftime("%Y-%m-%d %H:%M")+\
                     '   dep = ' + np.str(dep[iz]) + ' m',
                     fontsize=fsize+1)
        
        gs = gridspec.GridSpec(2, 3, height_ratios=[1,20])
  
        ax0 = plt.subplot(gs[1,0], projection=ccrs.Mercator(central_longitude=0))
        ax1 = plt.subplot(gs[1,1], projection=ccrs.Mercator(central_longitude=0))
        ax2 = plt.subplot(gs[1,2], projection=ccrs.Mercator(central_longitude=0))

        plot_decor_cartopy(region, ax0, fsize, lon, lat)
        plot_decor_cartopy(region, ax1, fsize, lon, lat)
        plot_decor_cartopy(region, ax2, fsize, lon, lat)
        
        # temp contourf
        cf0 = ax0.contourf(lon, lat, ptem[iz], cmap = cmo.cm.thermal,
                         transform=ccrs.PlateCarree())
        
        #same color levels for the pseudo-obs --> IMPORTANT!!!
        cmap_sc = ListedColormap(cf0.get_cmap()(cf0.norm(cf0.cvalues)))
        norm_sc = BoundaryNorm(cf0.levels, len(cf0.levels) - 1)

        # pseudo-obs of T
        sc0= ax0.scatter(lon_ctd, lat_ctd, c=tem_ctd[:, izctd], 
                    vmin=cf0.levels.min(), vmax=cf0.levels.max(),
                    cmap = cmap_sc, norm=norm_sc,
                    s=50, linewidths=0.2, edgecolors='w',
                    transform=ccrs.PlateCarree())
        # plt.colorbar(cf0)
        # plt.colorbar(sc0)
   
        # nan in pseudo-obs
        if len(lon_ctd[np.isnan(tem_ctd[:, izctd])])>0:
            sc00= ax0.scatter(lon_ctd[np.isnan(tem_ctd[:, izctd])], 
                         lat_ctd[np.isnan(tem_ctd[:, izctd])], 
                         c='0.5',
                         marker='x',
                         transform=ccrs.PlateCarree())        
        
        # plt.colorbar(sc0)
        # plt.colorbar(cf0)
        
        # salinity contourf
        cf1 = ax1.contourf(lon, lat, psal[iz], cmap = cmo.cm.haline,
                         transform=ccrs.PlateCarree())
        
        #same color levels for the pseudo-obs --> IMPORTANT!!!
        cmap_scS = ListedColormap(cf1.get_cmap()(cf1.norm(cf1.cvalues)))
        norm_scS = BoundaryNorm(cf1.levels, len(cf1.levels) - 1)

        # pseudo-obs of S     
        
        sc1= ax1.scatter(lon_ctd, lat_ctd, c=sal_ctd[:, izctd], 
                    vmin=cf1.levels.min(), vmax=cf1.levels.max(),
                    cmap = cmap_scS, norm=norm_scS, 
                    s=50, linewidths=0.2, edgecolors='w',
                    transform=ccrs.PlateCarree())     

        # nan in pseudo-obs
        if len(lon_ctd[np.isnan(sal_ctd[:, izctd])])>0:
            sc11= ax1.scatter(lon_ctd[np.isnan(sal_ctd[:, izctd])], 
                         lat_ctd[np.isnan(sal_ctd[:, izctd])], 
                         c='0.5',
                         marker='x',
                         transform=ccrs.PlateCarree())      
                
        
        # plt.colorbar(sc1)
        # plt.colorbar(cf1)        
        
        dhanom = dh[iz]-np.nanmean(dh[iz])
        extdh   = max(np.nanmax(dhanom), np.abs(np.nanmin(dhanom)))
        cf2 = ax2.contourf(lon, lat, dhanom, 
                           levels=np.linspace(-extdh, extdh, 10), #(-extdh, extdh, 10),
                           cmap = cmo.cm.balance, #curl, #delta, #balance, #plt.cm.coolwarm, #Spectral, #
                         transform=ccrs.PlateCarree()) 
        
        qv2 = ax2.quiver(lon[::2, ::2], lat[::2, ::2], 
                         ug[iz, ::2, ::2], vg[iz, ::2, ::2],
                         scale=5, color='k', 
                         transform=ccrs.PlateCarree(), zorder=20)
        ax2.quiverkey(qv2, 0.85, 1.25, 0.25, ' vgeo ' + "%.2f" % 0.25 +r' m s$^{-1}$', 
                  color='k', labelsep = 0.03, fontproperties={'size': fsize-1})

 
        # Colorbars
        axcb0 = plt.subplot(gs[0, 0])
        axcb1 = plt.subplot(gs[0, 1])
        axcb2 = plt.subplot(gs[0, 2])

        
        cb0 = plt.colorbar(cf0, cax=axcb0, orientation='horizontal',
                           ticks=cf0.levels[::2])
        cb1 = plt.colorbar(cf1, cax=axcb1, orientation='horizontal',
                           ticks=cf1.levels[::2])
        cb2 = plt.colorbar(cf2, cax=axcb2, orientation='horizontal',
                           ticks=cf2.levels[::2])
        
        cb0.ax.tick_params(labelsize=fsize)
        cb1.ax.tick_params(labelsize=fsize)
        cb2.ax.tick_params(labelsize=fsize)
        
        cb0.ax.set_title('Temperature [$^{\circ}$C]' , fontsize = fsize+1) 
        cb1.ax.set_title('Salinity' , fontsize = fsize+1) 
        cb2.ax.set_title('DH anomaly [dyn m]' , fontsize = fsize+1) 
        
        fig.canvas.draw()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.savefig(dir_figures + 'TSDHU_' + \
                                 name_conf +  '_dep_' + np.str(dep[iz]) + 'm.png', dpi=500)    
        
        
        