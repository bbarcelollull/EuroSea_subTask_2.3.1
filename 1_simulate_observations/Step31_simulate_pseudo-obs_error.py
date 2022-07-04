#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import netCDF4               as netcdf


"""
Step 31: Simulate random error for CTD and ADCP pseudo-observations,
following a Gaussian distribution with a standard deviation 
defined by the instrumental error. 
Different (uncorrelated) for each observations.

Following Gasparin et al., 2019.

Instrumental errors:
    ADCP: 0.01 m/s (PRE-SWOT, Gomis et al., 2001)
    CTD: 0.01ºC for T, 0.01 for S (Gasparin et al., 2019)
        
    
Note that more sofisticated errors may need to be added in future
such as representative error, depth error, etc.
(Halliwell et al., 2017, 2019)


For subtask 2.3.2 (reconstruction methods comparison). 
Only Med, eNATL60, reference configuration.

written by Bàrbara Barceló-Llull on 22-06-2021 at IMEDEA (Mallorca, Spain)

"""

def create_nc_pseudoobs_error(dir_save, name_file,
                            error_tem_ctd, error_u_adcp):
    
    '''
    Create a netcdf file to save the pseudo-observation errors
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('time_ctd', error_tem_ctd.shape[0]) 
    nc.createDimension('dep_ctd',  error_tem_ctd.shape[1])     

    if num_conf != '5':
    
        nc.createDimension('time_adcp', error_u_adcp.shape[0]) 
        nc.createDimension('dep_adcp',  error_u_adcp.shape[1])   
    
    # Create the variables...

    nc.createVariable('error_tem_ctd',  'f8', ('time_ctd', 'dep_ctd'))
    nc.createVariable('error_sal_ctd',  'f8', ('time_ctd', 'dep_ctd'))

    
    if num_conf != '5':
          
        nc.createVariable('error_u_adcp',  'f8', ('time_adcp', 'dep_adcp'))
        nc.createVariable('error_v_adcp',  'f8', ('time_adcp', 'dep_adcp'))
    
    # Write in variable attributes...

    
    nc.variables['error_tem_ctd'].long_name  = 'error of CTD sea_water_potential_temperature pseudo-observations'
    nc.variables['error_tem_ctd'].units      = 'degC'    

    nc.variables['error_sal_ctd'].long_name  = 'error of CTD sea_water_salinity pseudo-observations'
    nc.variables['error_sal_ctd'].units      = '1e-3' # check the units of sal!! I thinks it's PSU


    if num_conf != '5':
        
    
        nc.variables['error_u_adcp'].long_name  = 'error of ADCP eastward_sea_water_velocity pseudo-observations'
        nc.variables['error_u_adcp'].units      = 'm/s'      

        nc.variables['error_v_adcp'].long_name  = 'error of ADCP northward_sea_water_velocity pseudo-observations'
        nc.variables['error_v_adcp'].units      = 'm/s'    
    
    nc.close()   
    
    print('File created! >>>', name_file)    
    
    
def save2nc_pseudoobs_error(dir_save, name_file,
                            error_tem_ctd, error_sal_ctd, 
                            error_u_adcp, error_v_adcp):  
        
        nc = netcdf.Dataset(dir_save + name_file, 'a', format='NETCDF3_CLASSIC')

    
        nc.variables['error_tem_ctd'][:]   = error_tem_ctd
        nc.variables['error_sal_ctd'][:]   = error_sal_ctd
    
        if num_conf != '5':

            nc.variables['error_u_adcp'][:]    = error_u_adcp
            nc.variables['error_v_adcp'][:]    = error_v_adcp

        nc.close() 
    
        print('Data saved! >>>', name_file)    
  
if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directory to save figures '''
    
    #dir_fig       = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/pseudo_observations/'
    
    ''' Which model and region? '''
    
    model         = 'eNATL60' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Med' #'Atl' or 'Med'
    #config        = 'r'
      
    ''' Define instrumental errors '''
    ierror_adcp  = 0.01 #m/s (PRE-SWOT, Gomis et al., 2001)
    ierror_ctd_t = 0.01 #ºC  (Gasparin et al., 2019)
    ierror_ctd_s = 0.01 #PSU (Gasparin et al., 2019)   
            
    '''
    >>>>>> OPEN PSEUDO-OBSERVATIONS to know the data size <<<<<<
    '''
    
    dir_save = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'

    pobs_files  = sorted(glob.glob(dir_save + region + '*_'+model+'.nc'))
    
    for file in [pobs_files[-1]]: #only reference configurations for subtask 2.3.2
        
        name_conf = file[67:-3]
        num_conf  = file[76:77]
        
        print('')
        print('--------------------------------------')
        print('')
        print('Configuration file...', name_conf)
        print('')
        
        print('configuration...', num_conf)
        
        ''' Read .nc file to extract pseudo-obs size '''
             
        nc        = netcdf.Dataset(file, 'r')
        
        print('opening CTD pseudo-observations')
        # time_ctd  = nc.variables['time_ctd'][:]  
        # dep_ctd   = nc.variables['dep_ctd'][:]  
        # lat_ctd   = nc.variables['lat_ctd'][:]   
        # lon_ctd   = nc.variables['lon_ctd'][:]  
    
        tem_ctd   = nc.variables['tem_ctd'][:]      
        sal_ctd   = nc.variables['sal_ctd'][:]  


        if num_conf != '5':
            
            # print('opening ADCP pseudo-observations')
            # time_adcp = nc.variables['time_adcp'][:]  
            # dep_adcp  = nc.variables['dep_adcp'][:]  
            # lat_adcp  = nc.variables['lat_adcp'][:]   
            # lon_adcp  = nc.variables['lon_adcp'][:]      
    
            u_adcp    = nc.variables['u_adcp'][:]  
            v_adcp    = nc.variables['v_adcp'][:]
            
        #     period_for_model = [min(time_ctd.min(), time_adcp.min()),
        #                 max(time_ctd.max(), time_adcp.max())]  
            
        # else:    
        #     period_for_model = [time_ctd.min(), time_ctd.max()]  
            
        nc.close() 
    
        ''' 
        Create random error for each CTD and ADCP pseudo-obs following
        a Gaussian distribution with the standard deviation defined by the
        instrumental error (Gasparin et al., 2019)
        '''
        
        error_tem_ctd = np.random.normal(0, ierror_ctd_t, tem_ctd.shape)
        error_sal_ctd = np.random.normal(0, ierror_ctd_s, sal_ctd.shape)
        error_u_adcp  = np.random.normal(0, ierror_adcp, u_adcp.shape)
        error_v_adcp  = np.random.normal(0, ierror_adcp, v_adcp.shape)
        
        # check distribution
        plt.figure()
        plt.hist(error_sal_ctd.flatten(), bins = 200) 
        plt.show()
        
        
        ''' Save error data into an .nc file '''
        
        name_file = name_conf + '_error.nc'
        
        create_nc_pseudoobs_error(dir_save, name_file,
                            error_tem_ctd, error_u_adcp)
        
        save2nc_pseudoobs_error(dir_save, name_file,
                            error_tem_ctd, error_sal_ctd, 
                            error_u_adcp, error_v_adcp)
        
    
