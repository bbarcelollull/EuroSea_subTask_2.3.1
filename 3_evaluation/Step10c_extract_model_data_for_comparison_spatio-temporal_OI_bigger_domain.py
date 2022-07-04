#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import netCDF4               as netcdf
from matplotlib              import dates as mdates
import EuroSea_toolbox       as to
import pickle
from scipy                   import interpolate
import gsw
import deriv_tools           as dt 


"""
Step 10c

Extract 2D model data within a domain bigger than the corresponding configuration
and linearly interpolate model fields to the time of the OI map. 

(for CMEMS 3D data, we use data at the upper layer). 

Save fields, to be used in Step12f.

Fields: SSH, U, V, speed and Ro.

[To make figures, use fields saved in *Step10b*. Figures done in *Step11b*.]

** eNATL60 2D model outputs have been previously interpolated onto 
   a regular grid. Remember, SSH is only saved in the 2D files. **

written by Bàrbara Barceló-Llull on 31-05-2022 at IMEDEA (Mallorca, Spain)

"""
    

def compute_Ro(ut, vt, deltax, deltay, coriol_mean):    
        
    # Horizontal gradient of the geostrophic velocity
    gradu   = dt.grad2d_2d(ut,deltax,deltay)
    gradv   = dt.grad2d_2d(vt,deltax,deltay)
    
    # Geostrophic vertical relative vorticity
    vvort = gradv.real - gradu.imag
    
    # Gesotrophic Rossby number
    Ro   = vvort /coriol_mean

    
    return Ro   


def interp1dvar(timem_reg, ssh_reg, time_map):
       
        fi_ssh  = interpolate.interp1d(timem_reg, ssh_reg, axis=0, \
                                      fill_value=np.nan,bounds_error=False)
        ssh_int = fi_ssh(time_map[0])
        
        return ssh_int
    
            
if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directories '''
    
    dir_OIdata    = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/reconstructed_fields/spatio-temporal_OI_all_conf/'
    dir_pseudoobs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'
    dir_figures   = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/reconstructed_fields/spatio-temporal_OI/'
    dir_outputs   = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/model_outputs/'
    dir_dic       = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/comparison/spatio-temporal_OI/'
    
    ''' Which model and region? '''
    
    model         = 'CMEMS' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Atl' #'Atl' or 'Med'

    
    ''' Start dictionary where to save the data '''
    
    dic_all  = {}
    
    
    '''
        Open pseudo-observations to limit region for each configuration.
    '''
    
    oi_files  = sorted(glob.glob(dir_OIdata + region + '*_'+model + '*T.nc'))

    
    if  np.logical_and(region == 'Med', model == 'eNATL60'):
        
        oi_files_selec = oi_files[:-3] + [oi_files[-1]]
        
        oi_files = np.copy(oi_files_selec)
    
    
    for file in oi_files: 
        
        name_conf   = file[96:-42]
  

        print('')
        print('--------------------------------------')
        print('')
        print('Configuration name +  model ...', name_conf)
        print('')
        
        
        ''' 1) Read time and grid of the interpolated (reconstructed) fields '''
           
        ncT      = netcdf.Dataset(file, 'r')   
        lon_map  = ncT.variables['longitude'][:].data
        lat_map  = ncT.variables['latitude'][:].data   
        time_map = ncT.variables['time'][:].data 
        ncT.close() 

        print('')
        print('Time of the OI map ... ' )
        print(mdates.num2date(time_map[0]).strftime("%Y-%m-%d %H:%M"))
        print('')
        
        
        ''' 2) Open time period and coordinates of the pseudo-obs '''

        psobs_file =  name_conf + '.nc'
        
        print('')
        print('opening CTD pseudo-observations...', psobs_file)
        print('')
        
        ncpo        = netcdf.Dataset(dir_pseudoobs + psobs_file, 'r')
        time_ctd  = ncpo.variables['time_ctd'][:]  
        lat_ctd   = ncpo.variables['lat_ctd'][:]   
        lon_ctd   = ncpo.variables['lon_ctd'][:]  
        ncpo.close() 
        
        period_for_model = [time_ctd.min(),time_ctd.max()]
        
        print('')
        print('This configuration goes from... ' )
        print(mdates.num2date(period_for_model[0]).strftime("%Y-%m-%d %H:%M"))
        print('to...', )
        print(mdates.num2date(period_for_model[1]).strftime("%Y-%m-%d %H:%M"))
        print('')
        
        
        '''  3) Open model outputs for each time step within period_for_model '''
        # CMEMS 4D model outputs have the variable SSH, but the other models don't
        
        if model == 'CMEMS':
            
            time, lon, lat, dep, tem, sal, u, v, ssh = \
                   to.open_4D_model_outputs(dir_outputs, model, region, period_for_model)
            
            # Only use variables at the upper layer
                   
            iz = np.argmin(dep)
            
            print('')
            print('Considering CMEMS data at dep [m] = ...', dep[iz])
            print('')
            
            tem2d_all = tem[:, iz, :, :]
            sal2d_all = sal[:, iz, :, :]
            u2d_all   = u[:, iz, :, :]
            v2d_all   = v[:, iz, :, :]
            ssh_all   = np.copy(ssh) # already at the surface
            
        # WMOP and eNATL60 models only have SSH in the 2D files.
            
        elif model == 'WMOP':
                
            time, lon, lat, tem2d_all, sal2d_all, u2d_all, v2d_all, ssh_all  = \
                   to.open_WMOP_2D_outputs(dir_outputs, region, period_for_model)
                   
        elif model == 'eNATL60':
            
            time, lon, lat, u2d_all, v2d_all, ssh_all = \
                   to.open_eNATL60_2D_interp_outputs(dir_outputs, region, period_for_model)

        ''' 
           Extract model data:
               * in a domain 0.1º bigger than the sampling domain 
               * for the sampling period +- 1 day 
        '''
        
        dbigger = 0.1 #degrees 
        
        lonm_reg, latm_reg, timem_reg, \
               um_reg, vm_reg, ssh_reg = to.limit_bigger_2D_model_data(\
                                         u2d_all, v2d_all, 
                                         ssh_all, lon, lat, time, 
                                         lon_ctd, lat_ctd, time_ctd, dbigger) 
                   
                   
        plt.figure()
        plt.pcolor(lonm_reg, latm_reg, ssh_reg[0]) 
        plt.colorbar()
        plt.scatter(lon_ctd, lat_ctd, marker='o', color='y')
        plt.scatter(lon_map, lat_map, marker='x', color='r') 
        aa, bb = np.meshgrid(lonm_reg, latm_reg)
        plt.scatter(aa, bb, marker='*', color='b') 
        plt.title(name_conf)

        
        print('')
        print('Time period extracted data ... ' )
        print(mdates.num2date(timem_reg.min()).strftime("%Y-%m-%d %H:%M"))
        print('to...', )
        print(mdates.num2date(timem_reg.max()).strftime("%Y-%m-%d %H:%M"))
        print('')           
        
        
        lon, lat = np.meshgrid(lonm_reg, latm_reg)


        ''' Calculate Ro and speed from um_reg and vm_reg 
        (total horizontal velocities)'''

        lat_ref = 40
        
        length_deg_lonp, length_deg_latp = to.length_lon_lat_degs(lat_ref)    

        deltax      = (lonm_reg[1]-lonm_reg[0])*length_deg_lonp # meters
        deltay      = (latm_reg[1]-latm_reg[0])*length_deg_latp # meters
          
        coriol_mean = gsw.f(lat_ref) 
          
        Ro_reg    = np.ones(um_reg.shape) * np.nan
        speed_reg = np.ones(um_reg.shape) * np.nan
          
        for it in np.arange(timem_reg.shape[0]):
              
              ut = um_reg[it]
              vt = vm_reg[it]
              
              Ro_reg[it]    = compute_Ro(ut, vt, deltax, deltay, coriol_mean)
              speed_reg[it] = np.sqrt(ut**2 + vt**2)

        ''' Interpolate model data to the date of the OI map '''
        # timem_reg, ssh_reg, um_reg, vm_reg, speed_reg, Ro_reg, 
        
        ssh_int    = interp1dvar(timem_reg, ssh_reg,   time_map)
        u_int      = interp1dvar(timem_reg, um_reg,    time_map)
        v_int      = interp1dvar(timem_reg, vm_reg,    time_map)
        speed_int  = interp1dvar(timem_reg, speed_reg, time_map)
        Ro_int     = interp1dvar(timem_reg, Ro_reg,    time_map)
        
        # plt.figure()
        # for ii in [3]:#np.arange(lonm_reg.shape[0]):
        #     for jj in np.arange(latm_reg.shape[0]):
        #         plt.plot(timem_reg, speed_reg[:,jj, ii], 'o-b')
        #         plt.plot(time_map[0], speed_int[jj, ii], 'xr')

        ''' Save data in a dictionary '''
            
        dic_ref  = {}

        dic_ref.update({
                        'lon'      : lonm_reg,
                        'lat'      : latm_reg,
                        
                        'time_all' : timem_reg,
                        'ssh_all'  : ssh_reg,
                        'u_all'    : um_reg,
                        'v_all'    : vm_reg,
                        'speed_all': speed_reg,
                        'Ro_all'   : Ro_reg,
                        
                        # Variables of the ocean truth interpolated 
                        # to the time of the OI map
                        
                        'time_map'  : time_map,
                        'ssh_int'   : ssh_int,
                        'u_int'     : u_int, 
                        'v_int'     : v_int,
                        'speed_int' : speed_int,    
                        'Ro_int'    : Ro_int       
                        
                        }) 
            
        dic_all.update({name_conf: dic_ref})
             

    
    ''' Save extracted model data '''
    
    f_model = open(dir_dic + region + '_' + model + '_ocean_truth_SSH_speed_Ro_bigger.pkl','wb')
    pickle.dump(dic_all, f_model)
    f_model.close()         
    
