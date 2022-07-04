#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import netCDF4               as netcdf
from matplotlib              import dates as mdates
import EuroSea_toolbox       as to



"""
Step 3: check pseudo-observations, make figures

written by Bàrbara Barceló-Llull on 23-02-2021 at IMEDEA (Mallorca, Spain)

""" 
            
def plot_pseudoobs_CTD_4D(var_model_2d, var_obs_1d, string, save_name, \
                          lonm_reg, latm_reg, dep_mod,
                          time_model, dtime_model,
                          lon_ctd, lat_ctd, dep_obs, time_ctd,                          
                          dir_fig, name_conf):
         
        fig = plt.figure(figsize=(9,5))
        pc = plt.pcolor(lonm_reg, latm_reg, var_model_2d, cmap=plt.cm.jet,
                        vmin=np.nanmin(var_obs_1d), 
                        vmax=np.nanmax(var_obs_1d))
        
        plt.scatter(lon_ctd, lat_ctd, c=var_obs_1d, cmap=plt.cm.jet,
                    vmin=np.nanmin(var_obs_1d), 
                    vmax=np.nanmax(var_obs_1d), 
                    linewidths=0.5, edgecolors='w', s=60, 
                    label = 'CTD data at ' + np.str(dep_obs) + ' m')
        
        cond_near_ctd = np.where(np.logical_and(time_ctd>time_model-0.5*dtime_model,
                                                time_ctd<=time_model+0.5*dtime_model))
        
        plt.scatter(lon_ctd[cond_near_ctd], lat_ctd[cond_near_ctd], 
                    c='None', marker = 's', s=120,
                    linewidths=0.5, edgecolors='m',
                    label = 'CTD data on t_model ' + r'$\pm$ ' + 
                    np.str(0.5*dtime_model) + ' days' )        
        
        # nan in pseudo-obs
        if len(lon_ctd[np.isnan(var_obs_1d)])>0:
            sc11= plt.scatter(lon_ctd[np.isnan(var_obs_1d)], 
                         lat_ctd[np.isnan(var_obs_1d)], 
                         c='0.5',
                         marker='x',)   
            
        plt.colorbar(pc)
        plt.title(string + '    time = ' + 
                  mdates.num2date(time_model).strftime("%Y-%m-%d %H:%M") +
                  '    dep model (CTD) = ' + "{:.1f}".format(dep_mod) + ' (' +
                  "{:.1f}".format(dep_obs) +') m')
        plt.axis('image')
        plt.legend()
        plt.tight_layout()


        fig.savefig(dir_fig + name_conf + '_' + save_name +  '_' + \
                    'lev_' + '{0:04d}'.format(int(round(dep_obs))) +  'm'+  '_' + \
                    mdates.num2date(time_model).strftime("%Y%m%d_%H%M")  + \
                    '.png', dpi=300) 
            


def plot_pseudoobs_ADCP_4D(string, save_name, sc, 
                           dir_fig, name_conf,
                           lonm_reg, latm_reg, depm_reg,
                           um_reg_lev, vm_reg_lev, 
                           time_model, dtime_model,
                           lon_adcp, lat_adcp, 
                           u_adcp_lev, v_adcp_lev, time_adcp):
        
        fig = plt.figure(figsize=(9,5))
        qm = plt.quiver(lonm_reg, latm_reg, um_reg_lev, vm_reg_lev,
                        color='0.5', scale = sc, scale_units='inches')
        
        qd = plt.quiver(lon_adcp, lat_adcp, u_adcp_lev, v_adcp_lev,
                        color = 'b', scale = qm.scale, scale_units='inches', alpha=0.7)
        
        plt.quiverkey(qm, 0.9, 0.9, 0.2, r'0.2 m/s',
                   coordinates='figure') 

        plt.quiverkey(qd, 0.9, 0.8, 0.2, r'0.2 m/s',
                   coordinates='figure') 
        
        cond_near_adcp = np.where(np.logical_and(time_adcp>time_model-0.5*dtime_model,
                                                time_adcp<=time_model+0.5*dtime_model))
        
        plt.scatter(lon_adcp[cond_near_adcp], lat_adcp[cond_near_adcp], 
                    c='m', s=4,
                    label = 'ADCP data on t_model ' + r'$\pm$ ' + 
                    np.str(0.5*dtime_model) + ' days' )        
        
        plt.title(string + '    time = ' + 
                  mdates.num2date(time_model).strftime("%Y-%m-%d %H:%M") +
                  '    dep = ' +  np.str(int(round(depm_reg[izma])))  + ' m')
        plt.axis('image')
        plt.tight_layout()



        fig.savefig(dir_fig + name_conf + '_' + save_name +  '_' + \
                    'lev_' + '{0:04d}'.format(int(round(depm_reg[izma]))) +  'm'+  '_' + \
                    mdates.num2date(time_model).strftime("%Y%m%d_%H%M")  + \
                    '.png', dpi=300)  


            
if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directory to save figures '''
    
    dir_fig       = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/pseudo_observations/'
    
    
    ''' Which model and region? '''
    
    model         = 'eNATL60' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Med' #'Atl' or 'Med'
    dep_to_plot   = 100

    if model == 'eNATL60':
            
        dir_fig = dir_fig + region + '_eNATL60/'
            
            
    '''
    >>>>>> OPEN PSEUDO-OBSERVATIONS <<<<<<
    '''
    
    dir_save = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'

    pobs_files  = sorted(glob.glob(dir_save + region + '*_'+model+'.nc'))
    
    for file in [pobs_files[-1]]: #pobs_files: 
        
        name_conf = file[67:-3]
        num_conf  = file[76:77]
        
        print('')
        print('--------------------------------------')
        print('')
        print('Configuration file...', name_conf)
        print('')
        
        print('configuration...', num_conf)
        
        ''' Read .nc file '''
             
        nc        = netcdf.Dataset(file, 'r')
        
        print('opening CTD pseudo-observations')
        time_ctd  = nc.variables['time_ctd'][:]  
        dep_ctd   = nc.variables['dep_ctd'][:]  
        lat_ctd   = nc.variables['lat_ctd'][:]   
        lon_ctd   = nc.variables['lon_ctd'][:]  
    
        tem_ctd   = nc.variables['tem_ctd'][:]      
        sal_ctd   = nc.variables['sal_ctd'][:]  


        if num_conf != '5':
            
            print('opening ADCP pseudo-observations')
            time_adcp = nc.variables['time_adcp'][:]  
            dep_adcp  = nc.variables['dep_adcp'][:]  
            lat_adcp  = nc.variables['lat_adcp'][:]   
            lon_adcp  = nc.variables['lon_adcp'][:]      
    
            u_adcp    = nc.variables['u_adcp'][:]  
            v_adcp    = nc.variables['v_adcp'][:]
            
            period_for_model = [min(time_ctd.min(), time_adcp.min()),
                        max(time_ctd.max(), time_adcp.max())]  
            
        else:    
            period_for_model = [time_ctd.min(), time_ctd.max()]  
            
        nc.close() 
    
        

    
        '''
        >>>>>> OPEN MODEL OUTPUTS <<<<<<
        '''
    
        dir_outputs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/model_outputs/'

        time, lon, lat, dep, tem, sal, u, v, ssh = \
           to.open_4D_model_outputs(dir_outputs, model, region, period_for_model)
    


        '''
        Only use model data near the sampling domain
        '''

        if num_conf == '5':
            lonm_reg, latm_reg, timem_reg, depm_reg,  \
                temm_reg, salm_reg, um_reg, vm_reg = to.limit_model_data(num_conf, \
                                                       tem, sal, u, v, \
                                                       lon, lat, time, dep, \
                                                       lon_ctd, lat_ctd, time_ctd)

        else: 
            lonm_reg, latm_reg, timem_reg, depm_reg,  \
                temm_reg, salm_reg, um_reg, vm_reg = to.limit_model_data(num_conf, \
                                                       tem, sal, u, v, \
                                                       lon, lat, time, dep, \
                                                       lon_ctd, lat_ctd, time_ctd,\
                                                       lon_adcp, lat_adcp, time_adcp)               

    
        ''' Figures to check pseudo-obs '''
 
        print('')
        
        # dtime_model
        dtime_model = np.min(np.diff(timem_reg)) # days
        
        if num_conf == '5':
            d_mctd  = dep_to_plot #100 #30 
            
        else:
            d_mctd  = dep_to_plot #100 #5 
            d_madcp = dep_to_plot #100 #20
            
            iza  = np.argmin(np.abs(dep_adcp - d_madcp))
        
            izma = np.argmin(np.abs(depm_reg - dep_adcp[iza]))
            

            print('ADCP plotting model depth = ', depm_reg[izma])
            print('ADCP plotting observ. depth = ', dep_adcp[iza])
        
        izm = np.argmin(np.abs(depm_reg - d_mctd))
        izo = np.argmin(np.abs(dep_ctd - d_mctd))
    
        print('CTD plotting model depth = ', depm_reg[izm])
        print('CTD plotting observ. depth = ', dep_ctd[izo])
        
            
        for itm, time_model in enumerate(timem_reg):
        
            # T and S figures
        
            plot_pseudoobs_CTD_4D(temm_reg[itm,izm,:,:], 
                                  tem_ctd[:,izo], r'T [$^{\circ}$C]', 'T',
                                  lonm_reg, latm_reg, depm_reg[izm],
                                  time_model, dtime_model,
                                  lon_ctd, lat_ctd, dep_ctd[izo], time_ctd,
                                  dir_fig, name_conf)
            
            plot_pseudoobs_CTD_4D(salm_reg[itm,izm,:,:],
                                  sal_ctd[:,izo],  'S', 'S',
                                  lonm_reg, latm_reg, depm_reg[izm], 
                                  time_model, dtime_model,
                                  lon_ctd, lat_ctd, dep_ctd[izo], time_ctd,
                                  dir_fig, name_conf)
            
            if num_conf != '5':
                
                # u and v figures
                
                plot_pseudoobs_ADCP_4D('Horizontal velocity', 'ADCP', 1, #0.2
                                       dir_fig, name_conf, 
                                       lonm_reg, latm_reg, depm_reg,
                                       um_reg[itm,izma,:,:], vm_reg[itm,izma,:,:], 
                                       time_model, dtime_model,
                                       lon_adcp, lat_adcp, 
                                       u_adcp[:,iza], v_adcp[:,iza], time_adcp)


                
            
            plt.close('all')
            
