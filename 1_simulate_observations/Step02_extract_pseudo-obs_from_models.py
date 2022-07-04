#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import glob
import netCDF4               as netcdf
#from matplotlib              import dates as mdates
import EuroSea_toolbox       as to
from scipy.interpolate       import RegularGridInterpolator
import pickle
import time                  as counter_time

"""

Step 2: extract pseudo-observations from model outputs 
        for each configuration with CTD and ADCP
        profiles (time, lon, lat, depth)
        
        Model options: CMEMS, WMOP, eNATL60
        Region options: Med, Atl

written by Bàrbara Barceló-Llull on 18-02-2021 at IMEDEA (Mallorca, Spain)

"""

def open_conf_with_CTD(file_ctd, file_adcp):
      

    f_ctd    = open(dir_conf + file_ctd,'rb')    
    ctd_data = pickle.load(f_ctd)
    f_ctd.close()
    
    time_ctd = ctd_data['time_ctd']
    lon_ctd  = ctd_data['lon_ctd']
    lat_ctd  = ctd_data['lat_ctd']
    dep_ctd  = ctd_data['dep_ctd']
    
    f_adcp    = open(dir_conf + file_adcp,'rb')    
    adcp_data = pickle.load(f_adcp)
    f_adcp.close()    
    
    time_adcp = adcp_data['time_adcp']
    lon_adcp  = adcp_data['lon_adcp']
    lat_adcp  = adcp_data['lat_adcp']
    dep_adcp  = adcp_data['dep_adcp']      
    
    period_for_model = [min(time_ctd.min(), time_adcp.min()),
                        max(time_ctd.max(), time_adcp.max())]
     
    return time_ctd, lon_ctd, lat_ctd, dep_ctd, time_adcp, lon_adcp, \
                  lat_adcp, dep_adcp, period_for_model


def open_conf_with_gliders(file_ctd):
    
    ''' The configuration with gliders only has CTD profiles '''
    
    f_ctd    = open(dir_conf + file_ctd,'rb')    
    ctd_data = pickle.load(f_ctd)
    f_ctd.close()
    
    time_ctd = ctd_data['time_ctd']
    lon_ctd  = ctd_data['lon_ctd']
    lat_ctd  = ctd_data['lat_ctd']
    dep_ctd  = ctd_data['dep_ctd']    
    
    period_for_model = [time_ctd.min(), time_ctd.max()]    
    
    return time_ctd, lon_ctd, lat_ctd, dep_ctd, period_for_model

                
def extract_model_data_4D(var, time, dep, lat, lon, 
                  time_ctd, dep_ctd, lat_ctd, lon_ctd):
    
    '''
    
    Extract pseudo-observations from 
    the model variable "var" that has coordinates 
    (time, dep, lat, lon) to the position of the profiles
    with coordinates: time_ctd, dep_ctd, lat_ctd, lon_ctd
    
    >> valid for CTD and ADCP profiles, only change the profile coordinates. 
    
    We do this with a 4D linear interpolation (RegularGridInterpolator:
    The data must be defined on a regular grid; the grid spacing however 
    may be uneven.)
    
    
    output: var_int with coordinates (num_profiles, depth)
    '''
    
    mt = time
    mk = dep
    mj = lat
    mi = lon
    V = var

    start = counter_time.process_time()

    interp_func = RegularGridInterpolator((mt, mk, mj, mi), V) #default linear interp.

    
    print('time needed to create interp func  [s]...', 
                 counter_time.process_time() - start)

    var_int = np.ones((time_ctd.shape[0], dep_ctd.shape[0])) * np.nan

    #same index for time, lat and lon to define a profile
    for pfi in np.arange(time_ctd.shape[0]): 
        # different index for each depth layer
        for pfi_dep in np.arange(dep_ctd.shape[0]):
            pt = np.array([time_ctd[pfi], dep_ctd[pfi_dep], 
                            lat_ctd[pfi], lon_ctd[pfi]])
        
            var_int[pfi, pfi_dep] = interp_func(pt)
        
    print('time needed to apply interp func to profiles [s]...', 
                counter_time.process_time() - start)
        
    return var_int  

 

def create_nc_pseudoobs(num_conf):
    
    '''
    Create a netcdf file to save the pseudo-observations
    
    check units for each model!
    '''
    
    nc = netcdf.Dataset(dir_save + name_file , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('time_ctd', time_ctd.shape[0]) 
    nc.createDimension('dep_ctd',  dep_ctd.shape[0])     

    if num_conf != '5':
    
        nc.createDimension('time_adcp', time_adcp.shape[0]) 
        nc.createDimension('dep_adcp',  dep_adcp.shape[0])   
    
    # Create the variables...
    
    nc.createVariable('time_ctd', 'f4', ('time_ctd'))
    nc.createVariable('dep_ctd',  'f4', ('dep_ctd'))
    nc.createVariable('lat_ctd',  'f4', ('time_ctd'))
    nc.createVariable('lon_ctd',  'f4', ('time_ctd'))

    nc.createVariable('tem_ctd',  'f4', ('time_ctd', 'dep_ctd'))
    nc.createVariable('sal_ctd',  'f4', ('time_ctd', 'dep_ctd'))
    #nc.createVariable('ssh_ctd',  'f4', ('time_ctd'))
    
    if num_conf != '5':
        
        nc.createVariable('time_adcp', 'f4', ('time_adcp'))
        nc.createVariable('dep_adcp',  'f4', ('dep_adcp'))
        nc.createVariable('lat_adcp',  'f4', ('time_adcp'))
        nc.createVariable('lon_adcp',  'f4', ('time_adcp'))    
    
        nc.createVariable('u_adcp',  'f4', ('time_adcp', 'dep_adcp'))
        nc.createVariable('v_adcp',  'f4', ('time_adcp', 'dep_adcp'))
    
    # Write in variable attributes...
    nc.variables['time_ctd'].long_name  = 'Time of the CTD profiles in python format'
    nc.variables['time_ctd'].units      = 'Number of days (fraction part represents hours, minutes, seconds, ms) since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['dep_ctd'].long_name  = 'Depth of the CTD profiles'
    nc.variables['dep_ctd'].units      = 'm'
        
    nc.variables['lat_ctd'].long_name  = 'Latitude of the CTD profiles'
    nc.variables['lat_ctd'].units      = 'degrees_north'

    nc.variables['lon_ctd'].long_name  = 'Longitude of the CTD profiles'
    nc.variables['lon_ctd'].units      = 'degrees_east'
    
    nc.variables['tem_ctd'].long_name  = 'sea_water_potential_temperature of the CTD profiles'
    nc.variables['tem_ctd'].units      = 'degC'    

    nc.variables['sal_ctd'].long_name  = 'sea_water_salinity of the CTD profiles'
    nc.variables['sal_ctd'].units      = '1e-3' # check the units of sal!! I thinks it's PSU

    #nc.variables['ssh_ctd'].long_name  = 'sea_surface_height_above_sea_level of the CTD profiles'
    #nc.variables['ssh_ctd'].units      = 'm' 

    if num_conf != '5':
        
        nc.variables['time_adcp'].long_name  = 'Time of the ADCP profiles in python format'
        nc.variables['time_adcp'].units      = 'Number of days (fraction part represents hours, minutes, seconds, ms) since 0001-01-01 00:00:00 UTC, plus one.'
        
        nc.variables['dep_adcp'].long_name  = 'Depth of the ADCP profiles'
        nc.variables['dep_adcp'].units      = 'm'
        
        nc.variables['lat_adcp'].long_name  = 'Latitude of the ADCP profiles'
        nc.variables['lat_adcp'].units      = 'degrees_north'

        nc.variables['lon_adcp'].long_name  = 'Longitude of the ADCP profiles'
        nc.variables['lon_adcp'].units      = 'degrees_east'
    
        nc.variables['u_adcp'].long_name  = 'eastward_sea_water_velocity'
        nc.variables['u_adcp'].units      = 'm/s'      

        nc.variables['v_adcp'].long_name  = 'northward_sea_water_velocity'
        nc.variables['v_adcp'].units      = 'm/s'    
    
    nc.close()   
    
    print('File created! >>>', name_file)    
 
    
def save2nc_pseudoobs(num_conf, dir_save, name_file,  
                      time_ctd, dep_ctd, lat_ctd, lon_ctd, 
                      tem_int, sal_int, 
                      time_adcp = None, dep_adcp = None, lat_adcp = None, 
                      lon_adcp = None, u_int = None, v_int = None):

    
    ''' Save CTD and ADCP pseudo-observations to the nc. file '''
     
    nc = netcdf.Dataset(dir_save + name_file, 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time_ctd'][:]  = time_ctd
    nc.variables['dep_ctd'][:]   = dep_ctd
    nc.variables['lat_ctd'][:]   = lat_ctd
    nc.variables['lon_ctd'][:]   = lon_ctd
    
    nc.variables['tem_ctd'][:]   = tem_int
    nc.variables['sal_ctd'][:]   = sal_int
    #nc.variables['ssh_ctd'][:]   = ssh_int
    
    if num_conf != '5':
        
        nc.variables['time_adcp'][:] = time_adcp
        nc.variables['dep_adcp'][:]  = dep_adcp
        nc.variables['lat_adcp'][:]  = lat_adcp
        nc.variables['lon_adcp'][:]  = lon_adcp    

        nc.variables['u_adcp'][:]    = u_int
        nc.variables['v_adcp'][:]    = v_int

    nc.close() 
    
    print('Data saved! >>>', name_file) 

    
if __name__ == '__main__':
    
    
    ''' Directory to save pseudo-observations '''
    
    # comment these lines in lluna:
    dir_save    = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'
    dir_outputs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/model_outputs/'

    
    # uncomment these lines in lluna:
    #dir_save    = '/home/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'
    #dir_outputs = '/home/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/model_outputs/'
    
    ''' Which model and region? '''
    
    model         = 'eNATL60' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Med' #'Atl' or 'Med'


    ''' ----- If running in lluna uncomment this ----- '''
    
    # # Number of days since 0001-01-01 00:00:00 UTC, plus one.
    # old_epoch = '0000-12-31T00:00:00'
    # # configure mdates with the old_epoch
    # mdates.set_epoch(old_epoch)  # old epoch (pre MPL 3.3)

    # print('')
    # print('mdates works with days since... ', mdates.get_epoch())    
    
    ''' ----------------------------------------- '''
    
    
    '''
    >>>>>> 1) OPEN CTD (& ADCP) PROFILES FOR THE DESIRED CONFIGURATION <<<<<<
    '''    
    
    dir_conf = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/configurations/'
    
    # all configurations (Sep, Jan, Aug, Oct)
    conf_files_ctd_all = sorted(glob.glob(dir_conf + region + '*ctd.pkl') + 
                            glob.glob(dir_conf + region + '*gliders.pkl'))
    
    # all configurations (Sep and Jan) except Aug and Oct
    #conf_files_ctd = conf_files_ctd_all[:-3] + [conf_files_ctd_all[-1]]

    # all configurations
    conf_files_ctd = conf_files_ctd_all
    
    
    # only conf 3 and 5                        
    # conf_files_ctd = sorted(glob.glob(dir_conf + region + '*uctd.pkl') + 
    #                         glob.glob(dir_conf + region + '*gliders.pkl'))

    # only conf 5                        
    # conf_files_ctd = sorted(glob.glob(dir_conf + region + '*gliders.pkl'))
    
    for conf_file in conf_files_ctd: #conf_files_ctd[6:]: #conf_files_ctd:
        
        print('')
        print('Configuration file...', conf_file)
        print('')
        
        num_conf = conf_file[71:72]
        print('configuration...', num_conf)
        
        if num_conf == '3': # uctd + ADCP
            
            name_scenario = conf_file[62:97]
            
            print('name_scenario...', name_scenario)
        
            file_ctd  = name_scenario + '_uctd.pkl'
            file_adcp = name_scenario + '_adcp.pkl'
        
            # open configuration CTD/uCTD and ADCP data 
            time_ctd, lon_ctd, lat_ctd, dep_ctd, time_adcp, lon_adcp, \
                lat_adcp, dep_adcp, period_for_model = \
                                        open_conf_with_CTD(file_ctd, file_adcp)

                           
        elif  num_conf == '5':   # glider CTD
            name_scenario = conf_file[62:95] 
            print('name_scenario...', name_scenario)
            
            file_ctd  = name_scenario + '_gliders.pkl'

            # open configuration with gliders
            time_ctd_2d, lon_ctd_2d, lat_ctd_2d, dep_ctd, period_for_model = \
                                        open_conf_with_gliders(file_ctd)
            
            # shape of time/lon/lat variables = num_transects X number profiles in transect
            
            time_ctd = time_ctd_2d.flatten()
            lon_ctd  = lon_ctd_2d.flatten()  
            lat_ctd  = lat_ctd_2d.flatten()                
                                        
                                        
        else: # CTD rosette casts + ADCP
            
            name_scenario = conf_file[62:95] # region + '_conf_'+configuration+'_dep_1000m_res_10km_Sep'    
            print('name_scenario...', name_scenario)
        
            file_ctd  = name_scenario + '_ctd.pkl'
            file_adcp = name_scenario + '_adcp.pkl'
        
            # open configuration CTD/uCTD and ADCP data 
            time_ctd, lon_ctd, lat_ctd, dep_ctd, time_adcp, lon_adcp, \
                lat_adcp, dep_adcp, period_for_model = \
                                        open_conf_with_CTD(file_ctd, file_adcp)

    
        '''
        >>>>>> 2) OPEN MODEL OUTPUTS <<<<<<
        '''
        
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

        '''
        >>>>>> 3) EXTRACT PSEUDO-OBSERVATIONS <<<<<<
        '''
    
        print('')
        print('extracting temperature profiles...')
        print('')   
    
        tem_int = extract_model_data_4D(temm_reg, 
                                  timem_reg, depm_reg, latm_reg, lonm_reg, 
                                  time_ctd, dep_ctd, lat_ctd, lon_ctd)
    
        print('')
        print('extracting salinity profiles...')
        print('')   
    
        sal_int = extract_model_data_4D(salm_reg, 
                                  timem_reg, depm_reg, latm_reg, lonm_reg, 
                                  time_ctd, dep_ctd, lat_ctd, lon_ctd)
    
    
        if num_conf != '5':   # Configurations with ADCP pseudo-obs
            
            ''' 
            Note: before generating pseudo-obs of ADCP from 
            eNATL60 outputs, the original model velocities are
            along the original x and y axis (not meridional and zonal)
            and they need to be rotated.
            '''
            
            print('')
            print('extracting u profiles...')
            print('')   
    
            u_int = extract_model_data_4D(um_reg, 
                                  timem_reg, depm_reg, latm_reg, lonm_reg, 
                                  time_adcp, dep_adcp, lat_adcp, lon_adcp)      

            print('')
            print('extracting v profiles...')
            print('')   
    
            v_int = extract_model_data_4D(vm_reg, 
                                  timem_reg, depm_reg, latm_reg, lonm_reg, 
                                  time_adcp, dep_adcp, lat_adcp, lon_adcp)     
    
            print('')
            print('pseudo-observations generated!!!')
            print('')   
    

        ''' Save pseudo-observations '''
    
        name_file     = name_scenario + '_' + model + '.nc'
    
        # create netcdf file
        create_nc_pseudoobs(num_conf)
        
        if num_conf == '5': 
            # save data into the netcdf file
            save2nc_pseudoobs(num_conf, dir_save, name_file,  
                      time_ctd, dep_ctd, lat_ctd, lon_ctd, 
                      tem_int, sal_int)   
            
        else:     
            # save data into the netcdf file
            save2nc_pseudoobs(num_conf, dir_save, name_file,  
                      time_ctd, dep_ctd, lat_ctd, lon_ctd, 
                      tem_int, sal_int, 
                      time_adcp, dep_adcp, lat_adcp, lon_adcp,
                      u_int, v_int)     
        
        
        

        
   
        
         
    