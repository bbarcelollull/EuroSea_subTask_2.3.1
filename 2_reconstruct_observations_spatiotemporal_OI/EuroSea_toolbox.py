#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import matplotlib.gridspec   as gridspec
import netCDF4               as netcdf
from matplotlib              import dates as mdates
from datetime                import datetime, timedelta
import sys
import time                  as counter_time
from scipy.interpolate       import griddata
import gsw 
import deriv_tools           as dt

"""
EuroSea Task 2.3 Toolbox (2020-2022)

written by Bàrbara Barceló-Llull at IMEDEA (Mallorca, Spain)

"""


def convert_timesec2py(time_orig, d_ini_or):
    
        '''
        time_orig is the time in seconds since d_ini_or 
        convert this time to python format
        '''
        

        # Return seconds as days
        time_days = np.zeros(time_orig.shape)
  
        for ind, tt in enumerate(time_orig):
           #time_days[ind] = mdates.seconds(tt) # Return seconds as days.
           sec_in_day = 3600 * 24
           time_days[ind] = tt / sec_in_day 
    
        # Sum these days to d_ini_or
        time = d_ini_or + time_days
      
        
        # print('')
        # print('time period converted: ') 
        # print(mdates.num2date(time.min()).strftime("%Y-%m-%d %H:%M")) 
        # print(mdates.num2date(time.max()).strftime("%Y-%m-%d %H:%M")) 
        # print('')
        
        return time

def convert_timemin2py(time_orig, d_ini_or):
    
        '''
        time_orig is the time in minutes since d_ini_or 
        convert this time to python format
        '''
         
  
        # Return minutes as days
        time_days = np.zeros(time_orig.shape)
  
        for ind, tt in enumerate(time_orig):
           #time_days[ind] = mdates.minutes(tt) # Return minutes as days.
           min_in_day = 60 * 24
           time_days[ind] = tt / min_in_day 
           
           
        # Sum these days to d_ini_or
        time = d_ini_or + time_days
      
        
        print('')
        print('time period: ') 
        print(mdates.num2date(time.min()).strftime("%Y-%m-%d %H:%M")) 
        print(mdates.num2date(time.max()).strftime("%Y-%m-%d %H:%M")) 
        print('')
        
        return time
    
def convert_timehours2py(time_orig, d_ini_or):
    
        '''
        time_orig is the time in hours since d_ini_or 
        convert this time to python format
        '''
        
        # first date
        #d_ini_or    = mdates.date2num(datetime(1950, 1, 1, 0, 0, 0))  
  
        # Return hours as days
        time_days = np.zeros(time_orig.shape)
  
        for ind, tt in enumerate(time_orig):
           #time_days[ind] = mdates.hours(tt) # Return hours as days.
           hours_in_day = 24
           time_days[ind] = tt / hours_in_day     
           
        # Sum these days to d_ini_or
        time = d_ini_or + time_days
      
        
        print('')
        print('time period: ') 
        print(mdates.num2date(time.min()).strftime("%Y-%m-%d %H:%M")) 
        print(mdates.num2date(time.max()).strftime("%Y-%m-%d %H:%M")) 
        print('')
        
        return time

def open_CMEMS_outputs(dir_outputs, region):   
     

        
      if region == 'Atl':

        print('') 
        print('Opening CMEMS Atl 3D files...') 
        print('')          
          
        file_3D     = 'COSNWA_3D_CMEMS.nc'
        
        nc    = netcdf.Dataset(dir_outputs + file_3D, 'r')
        v     = nc.variables['vo'][:]  # northward_sea_water_velocity, m/s
        u     = nc.variables['uo'][:]  # eastward_sea_water_velocity, m/s
        tem   = nc.variables['thetao'][:] #sea_water_potential_temperature
        sal   = nc.variables['so'][:]     #sea_water_salinity, PSU
        ssh   = nc.variables['zos'][:]
        lat   = nc.variables['latitude'][:]   
        lon   = nc.variables['longitude'][:]  
        dep   = nc.variables['depth'][:]  
        time_orig  = nc.variables['time'][:]  # hours since 1950-01-01 00:00:00
        nc.close()    
        
        # first date
        d_ini_or    = mdates.date2num(datetime(1950, 1, 1, 0, 0, 0))  
        
        #convert time in hours to python format:
        time = convert_timehours2py(time_orig, d_ini_or)

        
      elif region == 'Med':    
        
        print('')  
        print('Opening CMEMS Med 3D files...') 
        print('')
        
        # old
        # file_3D_cur = 'MEDBAL_3D_CMEMS-currents.nc' 
        # file_3D_sal = 'MEDBAL_3D_CMEMS-sal.nc' 
        # file_3D_tem = 'MEDBAL_3D_CMEMS-tem.nc'
        # file_3D_ssh = 'MEDBAL_3D_CMEMS-ssh.nc'
          
        file_3D_cur = 'med-cmcc-cur-rean-d_1614869235237.nc' 
        file_3D_sal = 'med-cmcc-sal-rean-d_1614868760746.nc' 
        file_3D_tem = 'med-cmcc-tem-rean-d_1614867771545.nc'
        file_3D_ssh = 'med-cmcc-ssh-rean-d_1614868108446.nc'
        
        nc    = netcdf.Dataset(dir_outputs + '/CMEMS_MED_3D/' + file_3D_cur, 'r')
        v_mk     = nc.variables['vo'][:]  #northward_sea_water_velocity m/s
        u_mk     = nc.variables['uo'][:]  # eastward_sea_water_velocity m/s
        lat_mk   = nc.variables['lat'][:]   
        lon_mk   = nc.variables['lon'][:]  
        dep_mk   = nc.variables['depth'][:]  
        time_orig  = nc.variables['time'][:]   # minutes since 1900-01-01 00:00:00
        #time_orig  = nc.variables['time'][:]  # seconds since 1970-01-01 00:00:00
        nc.close()     
        
        nc    = netcdf.Dataset(dir_outputs + '/CMEMS_MED_3D/' + file_3D_sal, 'r')
        sal_mk   = nc.variables['so'][:] #sea_water_salinity, units = "1e-3" ;
        nc.close()            

        nc    = netcdf.Dataset(dir_outputs + '/CMEMS_MED_3D/' + file_3D_tem, 'r')
        tem_mk   = nc.variables['thetao'][:] #sea_water_potential_temperature  degrees_C
        nc.close()  

        nc    = netcdf.Dataset(dir_outputs + '/CMEMS_MED_3D/' + file_3D_ssh, 'r')
        ssh_mk   = nc.variables['zos'][:] #sea_surface_height_above_geoid [m]
        nc.close() 
        
        # first date
        #d_ini_or    = mdates.date2num(datetime(1970, 1, 1, 0, 0, 0)) 
        d_ini_or    = mdates.date2num(datetime(1900, 1, 1, 0, 0, 0))
        
        
        #convert time in seconds to python format:
        #time = convert_timesec2py(time_orig, d_ini_or)

        #convert time in minuts to python format:
        time = convert_timemin2py(time_orig, d_ini_or)
        
        # remove mask and change to nans
        lon = lon_mk.data
        lat = lat_mk.data
        dep = dep_mk.data
    
        tem = tem_mk.data
        tem[tem_mk.mask==True] = np.nan
    
        sal = sal_mk.data
        sal[sal_mk.mask==True] = np.nan
    
        u = u_mk.data
        u[u_mk.mask==True] = np.nan    
    
        v = v_mk.data
        v[v_mk.mask==True] = np.nan 
    
        ssh = ssh_mk.data
        ssh[ssh_mk.mask==True] = np.nan 

      return time, lon, lat, dep, tem, sal, u, v, ssh

    
def open_WMOP_3D_outputs(dir_models, region, period_for_model):   
     
      print('')
      print('Opening WMOP outputs...') 
      print('')   
                
      if region == 'Med':    
        
        print('')
        print('Opening Med 3D files...') 
        print('')
        
        dir_outputs = dir_models + 'WMOP_MED_3D/'
        
        # Extract only model data within the period of the configuration
        
        period_WMOP_min = int(period_for_model[0] - 1)
        period_WMOP_max = int(period_for_model[1] + 1)
        
        period_WMOP_all = np.arange(period_WMOP_min, period_WMOP_max+1, 1)
        
        create_arrays = 1
        
        for idt, pydate in enumerate(period_WMOP_all):
            
            date = mdates.num2date(pydate).strftime("%Y%m%d")
            
            file_WMOP = 'MEDBAL_3D_WMOP_HR_' + date + '.nc'

            # open file  (data without mask, where mask there are nans)        
     
            nc       = netcdf.Dataset(dir_outputs +  file_WMOP, 'r')
            temi     = nc.variables['temp'][:].data # potential temperature, Celsius
            sali     = nc.variables['salt'][:].data # salinity, no units
            ui       = nc.variables['u'][:].data  # eastward_velocity m/s            
            vi       = nc.variables['v'][:].data  # northward_velocity m/s

            
            lat     = nc.variables['lat'][:].data   
            lon     = nc.variables['lon'][:].data  
            dep     = nc.variables['depth'][:].data       
            timeio   = nc.variables['ocean_time'][:].data  # seconds since 1968-05-23 00:00:00 GMT
            nc.close() 

            # convert time to python format
        
            # first date
            d_ini_or    = mdates.date2num(datetime(1968, 5, 23, 0, 0, 0)) 
        
            #convert time in seconds since d_ini_or to python format:
            timei = convert_timesec2py(timeio, d_ini_or)
        
            # concatenate the data for each date to a single array
            
            if create_arrays == 1:

                create_arrays = 0
                
                tem = np.ones((period_WMOP_all.shape[0],
                               temi.shape[1],
                               temi.shape[2],
                               temi.shape[3])) * np.nan
                
                sal = np.copy(tem)
                u   = np.copy(tem)
                v   = np.copy(tem)
                ssh = np.ones((period_WMOP_all.shape[0],
                               temi.shape[2],
                               temi.shape[3])) * np.nan 
                
                time = np.ones((period_WMOP_all.shape[0])) * np.nan
             
            tem[idt]  = temi
            sal[idt]  = sali
            u[idt]    = ui
            v[idt]    = vi
            time[idt] = timei
           
      else:
        sys.exit('WMOP MODEL ONLY AVAILABLE IN THE MEDITERRANEAN!!! ')
        
      return time, lon, lat, dep, tem, sal, u, v, ssh  #ssh nan!

def open_WMOP_2D_outputs(dir_models, region, period_for_model):   
     
      print('')
      print('Opening WMOP outputs...') 
      print('')   
                
      if region == 'Med':    
        
        print('')
        print('Opening Med 2D files...') 
        print('')
        
        dir_outputs = dir_models + 'WMOP_MED_2D/'
        
        # Extract only model data within the period of the configuration
        
        period_WMOP_min = int(period_for_model[0] - 1)
        period_WMOP_max = int(period_for_model[1] + 1)
        
        period_WMOP_all = np.arange(period_WMOP_min, period_WMOP_max+1, 1)
        
        create_arrays = 1
        
        for idt, pydate in enumerate(period_WMOP_all):
            
            date = mdates.num2date(pydate).strftime("%Y%m%d")
            
            file_WMOP = 'MEDWEST_2D_WMOP_HR_' + date + '.nc'
            
            # open file  (data without mask, where mask there are nans)        
     
            nc       = netcdf.Dataset(dir_outputs +  file_WMOP, 'r')
            temi     = nc.variables['temp'][:].data # potential temperature, Celsius
            sali     = nc.variables['salt'][:].data # salinity, no units
            ui       = nc.variables['u'][:].data  # eastward_velocity m/s            
            vi       = nc.variables['v'][:].data  # northward_velocity m/s
            sshi     = nc.variables['zeta'][:].data # sea_surface_height[m]
            lat     = nc.variables['lat'][:].data   
            lon     = nc.variables['lon'][:].data  
            #dep     = nc.variables['depth'][:].data       
            timeio   = nc.variables['ocean_time'][:].data  # seconds since 1968-05-23 00:00:00 GMT
            nc.close() 
            

            # convert time to python format
        
            # first date
            d_ini_or    = mdates.date2num(datetime(1968, 5, 23, 0, 0, 0)) 
        
            #convert time in seconds since d_ini_or to python format:
            timei = convert_timesec2py(timeio, d_ini_or)
        
            # concatenate the data for each date to a single array
            
            if create_arrays == 1:

                create_arrays = 0
                
                tem = np.ones((period_WMOP_all.shape[0],
                               temi.shape[0],
                               temi.shape[1])) * np.nan
                
                sal = np.copy(tem)
                u   = np.copy(tem)
                v   = np.copy(tem)
                ssh = np.copy(tem)
                
                time = np.ones((period_WMOP_all.shape[0])) * np.nan
             
            tem[idt]  = temi
            sal[idt]  = sali
            u[idt]    = ui
            v[idt]    = vi
            time[idt] = timei
            ssh[idt]  = sshi
           
      else:
        sys.exit('WMOP MODEL ONLY AVAILABLE IN THE MEDITERRANEAN!!! ')
        
      return time, lon, lat, tem, sal, u, v, ssh  

def open_WMOP_2D_outputs_1file(file_WMOP):   
     

    nc      = netcdf.Dataset(file_WMOP, 'r')
    temi    = nc.variables['temp'][:].data # potential temperature, Celsius
    sali    = nc.variables['salt'][:].data # salinity, no units
    ui      = nc.variables['u'][:].data  # eastward_velocity m/s            
    vi      = nc.variables['v'][:].data  # northward_velocity m/s
    sshi    = nc.variables['zeta'][:].data # sea_surface_height[m]
    lat     = nc.variables['lat'][:].data   
    lon     = nc.variables['lon'][:].data  
    timeio  = nc.variables['ocean_time'][:].data  # seconds since 1968-05-23 00:00:00 GMT
    nc.close() 
            
    # convert time to python format
        
    # first date
    d_ini_or    = mdates.date2num(datetime(1968, 5, 23, 0, 0, 0)) 
        
    #convert time in seconds since d_ini_or to python format:
    timei = convert_timesec2py(timeio, d_ini_or)
        

    return timei, lon, lat, temi, sali, ui, vi, sshi  

def open_WMOP_2D_SSH_outputs_1file(file_NR):
    
      nc    = netcdf.Dataset(file_NR, 'r')
      
      sshi   = nc.variables['zeta'][:] # sea_surface_height
      latp   = nc.variables['lat'][:].data   
      lonp   = nc.variables['lon'][:].data 
      timeio = nc.variables['ocean_time'][:].data  # "seconds since 1968-05-23 00:00:00 GMT"
            
      nc.close() 

      # convert time to python format
        
      # first date
      d_ini_or    = mdates.date2num(datetime(1968, 5, 23, 0, 0, 0)) 
        
      #convert time in seconds since d_ini_or to python format:
      timei = convert_timesec2py(timeio, d_ini_or)
      
      return timei, lonp, latp, sshi    

def open_original_2D_eNATL60_outputs(dir_models, region, period_for_model):   
     
      print('')
      print('Opening original eNATL60 outputs...') 
      print('')   
                
      if region == 'Med':    
        
        print('')
        print('Opening Med 2D files...') 
        print('')
        
        dir_outputs = dir_models + 'eNATL60_MED_2D/'
        start_name  = 'eNATL60MEDWEST-BLB002_'

      elif region == 'Atl':
          
        print('')
        print('Opening Atl 2D files...') 
        print('')
        
        dir_outputs = dir_models + 'eNATL60_ATL_2D/'
        start_name  = 'eNATL60NANFL-BLB002_'      
        

      # Extract only model data within the period of the configuration
        
      period_WMOP_min = int(period_for_model[0])
      period_WMOP_max = int(period_for_model[1])
        
      period_WMOP_all = np.arange(period_WMOP_min, period_WMOP_max+1, 1)
        
      create_arrays = 1
        
      for idt, pydate in enumerate(period_WMOP_all):
            
            date = mdates.num2date(pydate).strftime("y%Ym%md%d")
            
            
            file_ssh = start_name + date + '.1h_sossheig.nc'
            file_v   = start_name + date + '.1h_somecrty.nc' #V: meridional current
            file_u   = start_name + date + '.1h_sozocrtx.nc' #U: zonal current
                        
            # open file  (data without mask, where mask there are nans)        
     
            nc      = netcdf.Dataset(dir_outputs +  file_ssh, 'r')
            ssh_mk = nc.variables['sossheig'][:] # sea_surface_height_above_geoid
            
            lat     = nc.variables['nav_lat'][:].data   
            lon     = nc.variables['nav_lon'][:].data 
            timeio  = nc.variables['time_counter'][:].data  # seconds since 1900-01-01 00:00:00
            
            nc.close() 
            
            
            nc      = netcdf.Dataset(dir_outputs +  file_v, 'r')
            vi_mk   = nc.variables['somecrty'][:] # ocean surface current along j-axis, m/s 
            nc.close()               

            nc      = netcdf.Dataset(dir_outputs +  file_u, 'r')
            ui_mk   = nc.variables['sozocrtx'][:] # ocean surface current along i-axis, m/s
            nc.close()   
            
            
            # convert time to python format
        
            # first date
            '''
            Note: in lluna this gives days since 1970-01-01 UTC
            '''
            d_ini_or    = mdates.date2num(datetime(1900, 1, 1, 0, 0, 0)) 
        
            #convert time in seconds since d_ini_or to python format:
            timei = convert_timesec2py(timeio, d_ini_or)
            
            
            # remove mask and change to nans
    
            sshi = ssh_mk.data
            sshi[ssh_mk.mask==True] = np.nan

            ui = ui_mk.data
            ui[ui_mk.mask==True] = np.nan    
    
            vi = vi_mk.data
            vi[vi_mk.mask==True] = np.nan 
            

            # concatenate the data for each date to a single array
            
            if create_arrays == 1:

                create_arrays = 0
                
                st, sy, sx = sshi.shape
                
                ssh = np.ones((period_WMOP_all.shape[0]*st, sy, sx)) * np.nan
                
                u   = np.copy(ssh)
                v   = np.copy(ssh)
                
                time = np.ones((period_WMOP_all.shape[0]*st)) * np.nan
             
            ssh[idt*st:(idt+1)*st]  = sshi
            u[idt*st:(idt+1)*st]    = ui
            v[idt*st:(idt+1)*st]    = vi
            time[idt*st:(idt+1)*st] = timei
           
      
      # nan when lon = 0 (and lat = 0)
            
      lonmk = np.ma.masked_where(lon==0, lon)
      #latmk = np.ma.masked_where(lon==0, lat)
    
      lonp = lonmk.data
      lonp[lonmk.mask==True] = np.nan 
    
      latp = lat
      latp[lonmk.mask==True] = np.nan 
    
      return time, lonp, latp, ssh, u, v  
 
def open_original_2D_eNATL60_outputs_1year(ssh_ofile):   
    
      ''' Open original 2D SSH eNATL60 output files (used in Step00h*.py '''
     
      print('')
      print('Opening original eNATL60 outputs...') 
      print('')   
   
      nc      = netcdf.Dataset(ssh_ofile, 'r')
      ssh_mk  = nc.variables['sossheig'][:] # sea_surface_height_above_geoid
      lat     = nc.variables['nav_lat'][:].data   
      lon     = nc.variables['nav_lon'][:].data 
      timeio  = nc.variables['time_counter'][:].data  # seconds since 1900-01-01 00:00:00
            
      nc.close() 
            
      # convert time to python format
        
      # first date

      d_ini_or    = mdates.date2num(datetime(1900, 1, 1, 0, 0, 0)) 
        
      #convert time in seconds since d_ini_or to python format:
      timei = convert_timesec2py(timeio, d_ini_or)
            
            
      # remove mask and change to nans
      sshi = ssh_mk.data
      sshi[ssh_mk.mask==True] = np.nan
           
      
      # nan when lon = 0 (and lat = 0)
            
      lonmk = np.ma.masked_where(lon==0, lon)
    
      lonp = lonmk.data
      lonp[lonmk.mask==True] = np.nan 
    
      latp = lat
      latp[lonmk.mask==True] = np.nan 
    
      return timei, lonp, latp, sshi  

def open_interpolated_2D_eNATL60_outputs_1year(ssh_ifile):   
    
      ''' 
      Open interpolated 2D SSH eNATL60 output files 
      (smoothed in Step00i*.py) 
      
      '''
     
      print('')
      print('Opening eNATL60 outputs...', ssh_ifile) 
      print('')   

      nc    = netcdf.Dataset(ssh_ifile, 'r')
      sshi  = nc.variables['ssh'][:] # sea_surface_height_above_geoid
      latp  = nc.variables['lat'][:].data   
      lonp  = nc.variables['lon'][:].data 
      timei = nc.variables['time'][:].data  # time in python datetime format: 
      # Number of days (fraction part represents hours, minutes, seconds, ms)
      # since 0001-01-01 00:00:00 UTC, plus one.
            
      nc.close() 
            
    
      return timei, lonp, latp, sshi  
    
def open_original_eNATL60_outputs(dir_models, region, period_for_model):   
     
      print('')
      print('Opening original eNATL60 outputs...') 
      print('')   
                
      if region == 'Med':    
        
        print('')
        print('Opening Med 3D files...') 
        print('')
        
        dir_outputs = dir_models + 'eNATL60_MED_3D/'
        start_name  = 'eNATL60MEDBAL-BLB002_'

      elif region == 'Atl':
          
        print('')
        print('Opening Atl 3D files...') 
        print('')
        
        dir_outputs = dir_models + 'eNATL60_ATL_3D/'
        start_name  = 'eNATL60COSNWA-BLB002_'      
        

      # Extract only model data within the period of the configuration
        
      period_WMOP_min = int(period_for_model[0])
      period_WMOP_max = int(period_for_model[1])
        
      period_WMOP_all = np.arange(period_WMOP_min, period_WMOP_max+1, 1)
        
      create_arrays = 1
        
      for idt, pydate in enumerate(period_WMOP_all):
            
            date = mdates.num2date(pydate).strftime("y%Ym%md%d")
            
            
            file_tem = start_name + date + '.1h_votemper_0-1000m.nc'
            file_sal = start_name + date + '.1h_vosaline_0-1000m.nc'
            file_v   = start_name + date + '.1h_vomecrty_0-1000m.nc' #ocean current along j-axis
            file_u   = start_name + date + '.1h_vozocrtx_0-1000m.nc' #ocean current along i-axis
            
            
            #file_mask = 'mask_eNATL60MEDBAL_3.6.nc'
            #file_grid = 'mesh_hgr_eNATL60MEDBAL_3.6.nc'
            
            # open file  (data without mask, where mask=True --> nans)        
     
            nc      = netcdf.Dataset(dir_outputs +  file_tem, 'r')
            temi_mk = nc.variables['votemper'][:] # sea_water_potential_temperature 
            
            lat     = nc.variables['nav_lat'][:].data   
            lon     = nc.variables['nav_lon'][:].data 
            depT    = nc.variables['deptht'][:].data   # Vertical T levels      
            timeio  = nc.variables['time_counter'][:].data  # seconds since 1900-01-01 00:00:00
            
            nc.close() 
            
            
            nc      = netcdf.Dataset(dir_outputs +  file_sal, 'r')
            sali_mk = nc.variables['vosaline'][:] # sea_water_practical_salinity, units=1e-3           
            nc.close()   
            
            nc      = netcdf.Dataset(dir_outputs +  file_v, 'r')
            vi_mk   = nc.variables['vomecrty'][:] # sea_water_y_velocity, m/s 
            #depV    = nc.variables['depthv'][:].data # Vertical V levels  # depT = depV
            nc.close()               

            nc      = netcdf.Dataset(dir_outputs +  file_u, 'r')
            ui_mk   = nc.variables['vozocrtx'][:] # sea_water_x_velocity, m/s
            #depU    = nc.variables['depthu'][:].data # Vertical U levels >>> depV=depU
            nc.close()   
            
            dep = depT
            
            # convert time to python format
        
            # first date
            d_ini_or    = mdates.date2num(datetime(1900, 1, 1, 0, 0, 0)) 
        
            #convert time in seconds since d_ini_or to python format:
            timei = convert_timesec2py(timeio, d_ini_or)
            
            
            # remove mask and change to nans
    
            temi = temi_mk.data
            temi[temi_mk.mask==True] = np.nan
    
            sali = sali_mk.data
            sali[sali_mk.mask==True] = np.nan
    
            ui = ui_mk.data
            ui[ui_mk.mask==True] = np.nan    
    
            vi = vi_mk.data
            vi[vi_mk.mask==True] = np.nan 
            

            # concatenate the data for each date to a single array
            
            if create_arrays == 1:

                create_arrays = 0
                
                st, sd, sy, sx = temi.shape
                
                tem = np.ones((period_WMOP_all.shape[0]*st, sd, sy, sx)) * np.nan
                
                sal = np.copy(tem)
                u   = np.copy(tem)
                v   = np.copy(tem)
                ssh = np.ones((period_WMOP_all.shape[0]*st, sy, sx)) * np.nan 
                
                time = np.ones((period_WMOP_all.shape[0]*st)) * np.nan
             
            tem[idt*st:(idt+1)*st]  = temi
            sal[idt*st:(idt+1)*st]  = sali
            u[idt*st:(idt+1)*st]    = ui
            v[idt*st:(idt+1)*st]    = vi
            time[idt*st:(idt+1)*st] = timei
           
      
      # nan when lon = 0 (and lat = 0)
            
      lonmk = np.ma.masked_where(lon==0, lon)
      #latmk = np.ma.masked_where(lon==0, lat)
    
      lonp = lonmk.data
      lonp[lonmk.mask==True] = np.nan 
    
      latp = lat
      latp[lonmk.mask==True] = np.nan 
    
      return time, lonp, latp, dep, tem, sal, u, v, ssh  


def interp_original_eNATL60_TS(tem, lon, lat, time, lon_new2d, lat_new2d):    
    
    ''' 
    Interpolate original eNATL60 temperature and salinity data 
    to a regular horizontal grid using griddata.
    
    input:
        tem: variable to interpolate dimensions: (time, dep, lat, lon)
        lon, lat: 2D longitude and latitude of the variable to interpolate
        time: 1D time of the variable to interpolate

        lon_new2d, lat_new2d: 2D new horizontal axis
    
    output: 
        tem_int : variable interpolated
    '''
    
    start = counter_time.process_time()
            
    print('time start griddata  [s]...', 
                  counter_time.process_time() - start)
    
    # points  = (lon[~np.isnan(lon)], lat[~np.isnan(lon)])
    tem_int = np.ones((tem.shape[0], tem.shape[1], 
                        lon_new2d.shape[0], lon_new2d.shape[1]))
    
    # save the mask of the original fields to apply to the original U and V
    var_ori_mask = np.ones(tem.shape) * np.nan
    
    
    for it in np.arange(tem.shape[0]):
        
        print('interpolating data on... ', mdates.num2date(time[it]).strftime("%Y-%m-%d %H:%M"))
        
        for iz in np.arange(tem.shape[1]):
            
            var    = tem[it, iz, :, :]
            
            # There are nan values in the original fields and also values = 0
            # Change 0 to nan
            var_masked = np.ma.masked_where(var<=0, var)
            var_mk_nan = np.copy(var_masked.data)
            var_mk_nan[var_masked.mask==True] = np.nan
            
            # Save the mask: 
            
            var_ori_mask[it, iz, :, :] = var_masked.mask
            
            # plt.figure()
            # plt.pcolor(lon, lat, var_mk_nan)
            
            # remove nans for the interpolation
            values = var_mk_nan[~np.isnan(var_mk_nan)] 
            points = (lon[~np.isnan(var_mk_nan)], lat[~np.isnan(var_mk_nan)])

            var_int = griddata(points, values, (lon_new2d, lat_new2d), 
                               method='linear')
            
            # The interpolated field has a bit of extrapolation, 
            # interpolate the mask and mask the interpolated field
            
            mk_orig  = np.ones(var_masked.shape)
            mk_orig[var_masked.mask==True] = 0
            
            points_mk = (lon.flatten(), lat.flatten())
            values_mk = mk_orig.flatten()
            
            mk_int = griddata(points_mk, values_mk, (lon_new2d, lat_new2d), 
                               method='nearest')
            # plt.figure()
            # plt.pcolor(lon_new2d, lat_new2d, mk_int, cmap=plt.cm.jet)
            # plt.colorbar()
            
            var_int_mked  = np.ma.masked_where(mk_int<1, var_int)
            var_int_final = np.copy(var_int_mked.data)
            var_int_final[var_int_mked.mask==True] = np.nan
            
            # plt.figure()
            # plt.pcolor(lon_new2d, lat_new2d, var_int_mked, cmap=plt.cm.jet)
            # plt.colorbar()

            # Where mk_int<1, mask with nans
            
            tem_int[it, iz, :, :] = var_int_final
            
            # plt.figure(figsize=(10,5))
            # plt.subplot(121)
            # plt.pcolor(lon, lat, tem[it,iz],
            #        cmap=plt.cm.jet)
            # plt.colorbar()
            # plt.subplot(122)
            # # plt.pcolor(lon_new2d, lat_new2d, tem_int[it, iz, :, :],
            # #        cmap=plt.cm.jet)
            # plt.pcolor(lon_new2d, lat_new2d, tem_int[it, iz, :, :], cmap=plt.cm.jet)
            # plt.colorbar()
            
           
        # lev = 5# 700
        # iz  = np.argmin(np.abs(dep-lev))

        # plt.figure(figsize=(10,5))
        # plt.subplot(121)
        # plt.pcolor(lon, lat, tem[0,iz], 
        #            vmin=np.nanmin(tem_int[0,iz]), 
        #            vmax=np.nanmax(tem_int[0,iz]),
        #            cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.subplot(122)
        # plt.pcolor(lon_new2d, lat_new2d, tem_int[0,iz], 
        #            vmin=np.nanmin(tem_int[0,iz]), 
        #            vmax=np.nanmax(tem_int[0,iz]),
        #            cmap=plt.cm.jet)
        # plt.colorbar()
        
        
    print('end start griddata  [s]...', 
                  counter_time.process_time() - start)   
    
    return tem_int, var_ori_mask

def interp_original_eNATL60_UV(var_ori, Tori_mask, tem_int,lon, lat, time, lon_new2d, lat_new2d):    
    
    ''' 
    Interpolate original eNATL60 U and V data 
    to a regular horizontal grid using griddata.
    
    input:
        var_ori: variable to interpolate dimensions: (time, dep, lat, lon)
        Tori_mask: is the mask of the original T or S data (masks 0 and nans). 
                     This mask is applied to the original U and V 
                     to exclude invalid values before interpolation.
        tem_int: interpolated T or S 
                     (to use its mask (nans) to mask the interpolated U and V)
        lon, lat: 2D longitude and latitude of the variable to interpolate
        time: 1D time of the variable to interpolate

        lon_new2d, lat_new2d: 2D new horizontal axis
    
    output: 
        var_int_all : variable interpolated
    '''
    
    start = counter_time.process_time()
            
    print('time start griddata  [s]...', 
                  counter_time.process_time() - start)
    
    # points  = (lon[~np.isnan(lon)], lat[~np.isnan(lon)])
    var_int_all = np.ones((var_ori.shape[0], var_ori.shape[1], 
                        lon_new2d.shape[0], lon_new2d.shape[1]))
    
    
    for it in np.arange(var_ori.shape[0]):
        
        print('interpolating data on... ', mdates.num2date(time[it]).strftime("%Y-%m-%d %H:%M"))
        
        for iz in np.arange(var_ori.shape[1]):
            
            var    = var_ori[it, iz, :, :]
            
            # nan where T or S is masked
            var_mk_nan = np.copy(var)
            var_mk_nan[Tori_mask[it, iz, :, :]==True] = np.nan
            
            # plt.figure()
            # plt.pcolor(lon, lat, var_mk_nan)
            
            # remove nans for the interpolation
            values = var_mk_nan[~np.isnan(var_mk_nan)] 
            points = (lon[~np.isnan(var_mk_nan)], lat[~np.isnan(var_mk_nan)])

            var_int = griddata(points, values, (lon_new2d, lat_new2d), 
                               method='linear')
            
            # The interpolated field has a bit of extrapolation, 
            # apply the mask of the interpolated T and S
            # that solves this issue
    
            var_int_mked  = np.ma.masked_where(np.isnan(tem_int[it, iz]), var_int)
            var_int_final = np.copy(var_int_mked.data)
            var_int_final[var_int_mked.mask==True] = np.nan

            
            var_int_all[it, iz, :, :] = var_int_final
            
            # # Figure to check
            # plt.figure(figsize=(10,5))
            # plt.subplot(121)
            # plt.pcolor(lon, lat, var_mk_nan,
            #         cmap=plt.cm.jet)
            # plt.colorbar()
            # plt.subplot(122)
            # # plt.pcolor(lon_new2d, lat_new2d, tem_int[it, iz, :, :],
            # #        cmap=plt.cm.jet)
            # plt.pcolor(lon_new2d, lat_new2d, var_int_all[it, iz, :, :], cmap=plt.cm.jet)
            # plt.colorbar()
        
            
           
        # lev = 5# 700
        # iz  = np.argmin(np.abs(dep-lev))

        # plt.figure(figsize=(10,5))
        # plt.subplot(121)
        # plt.pcolor(lon, lat, tem[0,iz], 
        #            vmin=np.nanmin(tem_int[0,iz]), 
        #            vmax=np.nanmax(tem_int[0,iz]),
        #            cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.subplot(122)
        # plt.pcolor(lon_new2d, lat_new2d, tem_int[0,iz], 
        #            vmin=np.nanmin(tem_int[0,iz]), 
        #            vmax=np.nanmax(tem_int[0,iz]),
        #            cmap=plt.cm.jet)
        # plt.colorbar()
        
        
    print('end start griddata  [s]...', 
                  counter_time.process_time() - start)   
    
    return var_int_all

def interp_original_2D_eNATL60(region, dir_outputs, SSHavgo, lono, lato):
    
            '''
            Interpolate the variables from the 2D (surf.) eNATL60 outputs 
            onto the 2D regular grid to which we have interpolated
            previously the 3D outputs. Remember: the original fields are in 
            an oblique grid. 
            '''
            
            # Open lon and lat of the interpolated 3D data

            if region == 'Med':    
        
                dir_save    = dir_outputs + 'eNATL60_MED_3D_int/'
                name_fileT  = 'eNATL60MEDBAL-BLB002_Sep_T.nc'

            elif region == 'Atl':
        
                dir_save    = dir_outputs + 'eNATL60_ATL_3D_int/'
                name_fileT  = 'eNATL60COSNWA-BLB002_Sep_T.nc' 

            nc   = netcdf.Dataset(dir_save + name_fileT, 'r')
            lati  = nc.variables['lat'][:].data
            loni  = nc.variables['lon'][:].data
            nc.close()             

            
            # interpolate data to the regular grid used for the 3D fields
            
            # remove nans for the interpolation
            values = SSHavgo[~np.isnan(SSHavgo)] 
            points = (lono[~np.isnan(SSHavgo)], lato[~np.isnan(SSHavgo)])

            SSHavg_int = griddata(points, values, (loni, lati), 
                               method='linear')            
            
            return loni, lati, SSHavg_int

def interp_original_2D_eNATL60_whole_domain(SSHavgo, lono, lato, loni, lati, dmethod):
    
            '''
            Interpolate the variables from the 2D (surf.) eNATL60 outputs 
            onto a 2D regular grid for the whole domain. 
            Remember: the original fields are in an oblique grid. 
            '''
 
            # interpolate data to the regular grid (loni, lati)
            
            # remove nans for the interpolation
            values = SSHavgo[~np.isnan(SSHavgo)] 
            points = (lono[~np.isnan(SSHavgo)], lato[~np.isnan(SSHavgo)])

            SSHavg_int = griddata(points, values, (loni, lati), 
                               method=dmethod)            
            
            return SSHavg_int  
        
        
def create_nc_interpolated_eNATL60_T(dir_save, name_file, time, dep, lon_new2d):
    
    '''
    Create a netcdf file to save the interpolated eNATL60 model outputs
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '_T.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', time.shape[0]) 
    nc.createDimension('ddep',  dep.shape[0])     
    nc.createDimension('dlat', lon_new2d.shape[0])   
    nc.createDimension('dlon', lon_new2d.shape[1]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime')) #'f8' (64-bit floating point)
    nc.createVariable('dep',  'f4', ('ddep'))
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    nc.createVariable('tem',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('sal',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('u',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('v',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['dep'].long_name  = 'Depth of the model data'
    nc.variables['dep'].units      = 'm'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    nc.variables['tem'].long_name  = 'sea_water_potential_temperature'
    nc.variables['tem'].units      = 'degC'    

    # nc.variables['sal'].long_name  = 'sea_water_practical_salinity'
    # nc.variables['sal'].units      = '1e-3' 

    # nc.variables['u'].long_name  = 'sea_water_x_velocity'
    # nc.variables['u'].units      = 'm/s' 

    # nc.variables['v'].long_name  = 'sea_water_y_velocity'
    # nc.variables['v'].units      = 'm/s'   
    
    nc.close()   
    
    print('File created! >>>', name_file + '_T.nc')    

def create_nc_interpolated_eNATL60_S(dir_save, name_file, time, dep, lon_new2d):
    
    '''
    Create a netcdf file to save the interpolated eNATL60 model outputs
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '_S.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', time.shape[0]) 
    nc.createDimension('ddep',  dep.shape[0])     
    nc.createDimension('dlat', lon_new2d.shape[0])   
    nc.createDimension('dlon', lon_new2d.shape[1]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime'))
    nc.createVariable('dep',  'f4', ('ddep'))
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    # nc.createVariable('tem',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    nc.createVariable('sal',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('u',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('v',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['dep'].long_name  = 'Depth of the model data'
    nc.variables['dep'].units      = 'm'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    # nc.variables['tem'].long_name  = 'sea_water_potential_temperature'
    # nc.variables['tem'].units      = 'degC'    

    nc.variables['sal'].long_name  = 'sea_water_practical_salinity'
    nc.variables['sal'].units      = '1e-3' 

    # nc.variables['u'].long_name  = 'sea_water_x_velocity'
    # nc.variables['u'].units      = 'm/s' 

    # nc.variables['v'].long_name  = 'sea_water_y_velocity'
    # nc.variables['v'].units      = 'm/s'   
    
    nc.close()   
    
    print('File created! >>>', name_file + '_S.nc') 

def create_nc_interpolated_eNATL60_U(dir_save, name_file, time, dep, lon_new2d):
    
    '''
    Create a netcdf file to save the interpolated eNATL60 model outputs
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '_U.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', time.shape[0]) 
    nc.createDimension('ddep',  dep.shape[0])     
    nc.createDimension('dlat', lon_new2d.shape[0])   
    nc.createDimension('dlon', lon_new2d.shape[1]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime'))
    nc.createVariable('dep',  'f4', ('ddep'))
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    # nc.createVariable('tem',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('sal',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    nc.createVariable('u',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('v',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['dep'].long_name  = 'Depth of the model data'
    nc.variables['dep'].units      = 'm'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    # nc.variables['tem'].long_name  = 'sea_water_potential_temperature'
    # nc.variables['tem'].units      = 'degC'    

    # nc.variables['sal'].long_name  = 'sea_water_practical_salinity'
    # nc.variables['sal'].units      = '1e-3' 

    nc.variables['u'].long_name  = 'sea_water_x_velocity'
    nc.variables['u'].units      = 'm/s' 

    # nc.variables['v'].long_name  = 'sea_water_y_velocity'
    # nc.variables['v'].units      = 'm/s'   
    
    nc.close()   
    
    print('File created! >>>', name_file + '_U.nc') 


def create_nc_interpolated_eNATL60_V(dir_save, name_file, time, dep, lon_new2d):
    
    '''
    Create a netcdf file to save the interpolated eNATL60 model outputs
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '_V.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', time.shape[0]) 
    nc.createDimension('ddep',  dep.shape[0])     
    nc.createDimension('dlat', lon_new2d.shape[0])   
    nc.createDimension('dlon', lon_new2d.shape[1]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime'))
    nc.createVariable('dep',  'f4', ('ddep'))
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    # nc.createVariable('tem',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('sal',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    # nc.createVariable('u',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    nc.createVariable('v',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['dep'].long_name  = 'Depth of the model data'
    nc.variables['dep'].units      = 'm'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    # nc.variables['tem'].long_name  = 'sea_water_potential_temperature'
    # nc.variables['tem'].units      = 'degC'    

    # nc.variables['sal'].long_name  = 'sea_water_practical_salinity'
    # nc.variables['sal'].units      = '1e-3' 

    # nc.variables['u'].long_name  = 'sea_water_x_velocity'
    # nc.variables['u'].units      = 'm/s' 

    nc.variables['v'].long_name  = 'sea_water_y_velocity'
    nc.variables['v'].units      = 'm/s'   
    
    nc.close()   
    
    print('File created! >>>', name_file + '_V.nc') 
    
def create_nc_interpolated_2D_eNATL60(dir_save, name_file, time, lon_new2d):
    
    '''
    Create a netcdf file to save the interpolated 2D eNATL60 model outputs.
    variables: SSH, U and V
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', time.shape[0])  
    nc.createDimension('dlat', lon_new2d.shape[0])   
    nc.createDimension('dlon', lon_new2d.shape[1]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime')) #'f8' (64-bit floating point)
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    nc.createVariable('ssh',  'f4', ('dtime', 'dlat', 'dlon'))
    nc.createVariable('u',    'f4', ('dtime', 'dlat', 'dlon'))
    nc.createVariable('v',    'f4', ('dtime', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    nc.variables['ssh'].long_name  = 'sea surface height'
    nc.variables['ssh'].units      = 'm'    

    nc.variables['u'].long_name  = 'sea_water_x_velocity'
    nc.variables['u'].units      = 'm/s' 

    nc.variables['v'].long_name  = 'sea_water_y_velocity'
    nc.variables['v'].units      = 'm/s'    
    
    nc.close()   
    
    print('File created! >>>', name_file + '.nc')   

def create_nc_SSHi_2D_eNATL60_whole_domain(dir_save, name_file, sshi):
    
    '''
    Create a netcdf file to save the interpolated 2D eNATL60 SSH.
    For the whole domain and for each time step.
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', 1)  
    nc.createDimension('dlat', sshi.shape[0])   
    nc.createDimension('dlon', sshi.shape[1]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime')) #'f8' (64-bit floating point)
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    nc.createVariable('ssh',  'f4', ('dlat', 'dlon'))
    # nc.createVariable('u',    'f4', ('dtime', 'dlat', 'dlon'))
    # nc.createVariable('v',    'f4', ('dtime', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    nc.variables['ssh'].long_name  = 'sea surface height'
    nc.variables['ssh'].units      = 'm'    

    # nc.variables['u'].long_name  = 'sea_water_x_velocity'
    # nc.variables['u'].units      = 'm/s' 

    # nc.variables['v'].long_name  = 'sea_water_y_velocity'
    # nc.variables['v'].units      = 'm/s'    
    
    nc.close()   
    
    print('File created! >>>', name_file + '.nc') 
    
def create_nc_SSHi_2D_eNATL60_whole_domain_all(dir_save, name_file, sshi):
    
    '''
    Create a netcdf file to save the interpolated 2D eNATL60 SSH.
    For the whole domain.
    
    Valid also for the smoothed data.
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', sshi.shape[0])  
    nc.createDimension('dlat',  sshi.shape[1])   
    nc.createDimension('dlon',  sshi.shape[2]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime')) #'f8' (64-bit floating point)
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    nc.createVariable('ssh',  'f4', ('dtime', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    nc.variables['ssh'].long_name  = 'sea surface height'
    nc.variables['ssh'].units      = 'm'    

    
    nc.close()   
    
    print('File created! >>>', name_file + '.nc') 




    
def save2nc_interpolated_2D_eNATL60(dir_save, name_file, time,  
                                   lat, lon, ssh, u, v):

    ''' 
    Save the interpolated 2D eNATL60 model data to a nc. file. 
    lon and lat are 2D and define the new grid
    ssh, u and v (3D) are the interpolated fields.
    '''
     
    nc = netcdf.Dataset(dir_save + name_file + '.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    nc.variables['ssh'][:]   = ssh 
    nc.variables['u'][:]     = u
    nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file + '.nc')  

def save2nc_SSHi_2D_eNATL60_whole_domain(dir_save, name_file, time, lat, lon, ssh):

    ''' 
    Save the interpolated 2D eNATL60 model data to a nc. file. 
    lon and lat are 2D and define the new grid
    ssh is the interpolated field.
    
    Interpolation done in the whole domain.
    One file for each time step.
    '''
     
    nc = netcdf.Dataset(dir_save + name_file + '.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    nc.variables['ssh'][:]   = ssh 
    # nc.variables['u'][:]     = u
    # nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file + '.nc') 
    
def save2nc_SSHi_2D_eNATL60_whole_domain_all(dir_save, name_file, time, lat, lon, ssh):

    ''' 
    Save the interpolated 2D eNATL60 model data to a nc. file. 
    lon and lat are 2D and define the new grid
    ssh is the interpolated field.
    
    Interpolation done in the whole domain.
    One file for each time step.
    '''
     
    nc = netcdf.Dataset(dir_save + name_file + '.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    nc.variables['ssh'][:]   = ssh 
    # nc.variables['u'][:]     = u
    # nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file + '.nc') 

def create_nc_SSHi_2D_eNATL60_whole_domain_1year(dir_save, name_file, sshi):
    
    '''
    Create a netcdf file to save the interpolated 2D eNATL60 SSH.
    2D fields for the whole domain. 
    Save 1 file for each original file (1 day).
    For all the data in 1 year. 
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', sshi.shape[0])  
    nc.createDimension('dlat',  sshi.shape[1])   
    nc.createDimension('dlon',  sshi.shape[2]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime')) #'f8' (64-bit floating point)
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    nc.createVariable('ssh',  'f4', ('dtime', 'dlat', 'dlon'))

    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    nc.variables['ssh'].long_name  = 'sea surface height'
    nc.variables['ssh'].units      = 'm'     
    
    nc.close()   
    
    print('File created! >>>', name_file + '.nc') 

def save2nc_SSHi_2D_eNATL60_whole_domain_1year(dir_save, name_file, time, lat, lon, ssh):
    
    ''' 
    Save the interpolated 2D eNATL60 model data to a nc. file. 
    lon and lat are 2D and define the new grid
    ssh is the interpolated field.
    
    Interpolation done in the whole domain.
    Save 1 file for each original file (1 day).
    For all the data in 1 year. 
    
    Valid also for the smoothed fields.
    '''
     
    nc = netcdf.Dataset(dir_save + name_file + '.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    nc.variables['ssh'][:]   = ssh 

    nc.close() 
    
    print('Data saved! >>>', name_file + '.nc') 
    
def create_nc_SSHif_2D_eNATL60_whole_domain_all(dir_save, name_file, sshi):
    
    '''
    Create a netcdf file to save the interpolated and smoothed 2D eNATL60 SSH.
    
    This creates a netcdf file to save: SSH_original, SSH_largescale,
    SSH_swotscales (from Step00f*.py). 
    
    SSH_swotscales is computed as follows (in Step00f*.py)): 
        - Filter SSH_original with L = 200 km (large scale) --> SSH_largescale
        - Compute SSH_smallscale = SSH_original - SSH_largescale
        - Filter SSH_smallscale with L = 30 km --> SSH_swotscales 
          (keep swot scales between 20-150 km wavelength)
    
    '''
    
    nc = netcdf.Dataset(dir_save + name_file + '.nc' , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', sshi.shape[0])  
    nc.createDimension('dlat',  sshi.shape[1])   
    nc.createDimension('dlon',  sshi.shape[2]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime')) #'f8' (64-bit floating point)
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    nc.createVariable('ssh_original',    'f4', ('dtime', 'dlat', 'dlon'))
    nc.createVariable('ssh_largescale',  'f4', ('dtime', 'dlat', 'dlon'))
    nc.createVariable('ssh_swotscales',  'f4', ('dtime', 'dlat', 'dlon'))
    # nc.createVariable('u',    'f4', ('dtime', 'dlat', 'dlon'))
    # nc.createVariable('v',    'f4', ('dtime', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    nc.variables['ssh_original'].long_name  = 'original sea surface height'
    nc.variables['ssh_original'].units      = 'm'    

    nc.variables['ssh_largescale'].long_name  = 'filtered sea surface height, represents scales > 200km (150 km wavelength)'
    nc.variables['ssh_largescale'].units      = 'm'    

    nc.variables['ssh_swotscales'].long_name  = 'filtered sea surface height, represents 30 km < scales < 200 km (20-150 km wavelength, the SWOT scales)'
    nc.variables['ssh_swotscales'].units      = 'm'  
    
    # nc.variables['u'].long_name  = 'sea_water_x_velocity'
    # nc.variables['u'].units      = 'm/s' 

    # nc.variables['v'].long_name  = 'sea_water_y_velocity'
    # nc.variables['v'].units      = 'm/s'    
    
    nc.close()   
    
    print('File created! >>>', name_file + '.nc') 
    
    
def save2nc_SSHif_2D_eNATL60_whole_domain_all(dir_save, name_file, time, 
                                              lat, lon, ssh_original,
                                              ssh_largescale, ssh_swotscales):

    ''' 
    Save the interpolated and smoothed 2D eNATL60 model data to a nc. file. 
    lon and lat are 2D and define the new grid
    
    Save: SSH_original, SSH_largescale,
    SSH_swotscales (from Step00f*.py). 
    
    SSH_swotscales is computed as follows (in Step00f*.py)): 
        - Filter SSH_original with L = 200 km (large scale) --> SSH_largescale
        - Compute SSH_smallscale = SSH_original - SSH_largescale
        - Filter SSH_smallscale with L = 30 km --> SSH_swotscales 
          (keep swot scales between 20-150 km wavelength)
    '''
     
    nc = netcdf.Dataset(dir_save + name_file + '.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    nc.variables['ssh_original'][:]   = ssh_original 
    nc.variables['ssh_largescale'][:] = ssh_largescale 
    nc.variables['ssh_swotscales'][:] = ssh_swotscales 

    nc.close() 
    
    print('Data saved! >>>', name_file + '.nc') 
    
def open_eNATL60_2D_interp_outputs(dir_outputs, region, period_for_model):
    
    '''
    Open interpolated eNATL60 2D model outputs. 
    
    For the comparison between different configurations and 
    the ocean truth (Step 10b, 10c).
    '''
    
    if mdates.num2date(period_for_model[1]).month == 9:
        period = 'Sep'
        
    elif mdates.num2date(period_for_model[1]).month == 1:
        period = 'Jan'

    elif mdates.num2date(period_for_model[1]).month == 8:
        period = 'Aug'

    elif mdates.num2date(period_for_model[1]).month == 10:
        period = 'Oct'
        
    else:
        print('No valid period')
        
    if region == 'Med':    
        
        dir_save    = dir_outputs + 'eNATL60_MED_2D_int/'
        name_file  = 'eNATL60MEDWEST-BLB002_' + period + '.nc'
 
        
        print('')
        print('Opening...', name_file)

    elif region == 'Atl':
        
        dir_save    = dir_outputs + 'eNATL60_ATL_2D_int/'
        name_file  = 'eNATL60NANFL-BLB002_' + period + '.nc' 
    
        print('')
        print('Opening...', name_file)
        
 
    nc   = netcdf.Dataset(dir_save + name_file, 'r')
    time = nc.variables['time'][:].data  # days since 0001-01-01 00:00:00 UTC, plus one
    lat  = nc.variables['lat'][:].data
    lon  = nc.variables['lon'][:].data
    ssh  = nc.variables['ssh'][:].data 
    u    = nc.variables['u'][:].data 
    v    = nc.variables['v'][:].data 
    nc.close() 
    
    
    return time, lon[0,:], lat[:,0], u, v, ssh #ssh nan 



     
def create_nc_interpolated_eNATL60(dir_save, name_file, time, dep, lon_new2d):
    
    '''
    Create a netcdf file to save the interpolated 3D eNATL60 model outputs.
    (not used, data saved in individual files for each variable).
    '''
    
    nc = netcdf.Dataset(dir_save + name_file , 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dtime', time.shape[0]) 
    nc.createDimension('ddep',  dep.shape[0])     
    nc.createDimension('dlat', lon_new2d.shape[0])   
    nc.createDimension('dlon', lon_new2d.shape[1]) 
    
    
    # Create the variables...
    
    nc.createVariable('time', 'f8', ('dtime'))
    nc.createVariable('dep',  'f4', ('ddep'))
    nc.createVariable('lat',  'f4', ('dlat', 'dlon'))
    nc.createVariable('lon',  'f4', ('dlat', 'dlon'))

    nc.createVariable('tem',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    nc.createVariable('sal',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    nc.createVariable('u',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))
    nc.createVariable('v',  'f4', ('dtime', 'ddep', 'dlat', 'dlon'))


    # Write in variable attributes...
    nc.variables['time'].long_name = 'Time of the model data'
    nc.variables['time'].units     = 'Number of days (fraction part represents hours, minutes, seconds, ms) since 0001-01-01 00:00:00 UTC, plus one.'
        
    nc.variables['dep'].long_name  = 'Depth of the model data'
    nc.variables['dep'].units      = 'm'
        
    nc.variables['lat'].long_name  = 'Latitude of the interpolated data'
    nc.variables['lat'].units      = 'degrees_north'

    nc.variables['lon'].long_name  = 'Longitude of the interpolated data'
    nc.variables['lon'].units      = 'degrees_east'
    
    nc.variables['tem'].long_name  = 'sea_water_potential_temperature'
    nc.variables['tem'].units      = 'degC'    

    nc.variables['sal'].long_name  = 'sea_water_practical_salinity'
    nc.variables['sal'].units      = '1e-3' 

    nc.variables['u'].long_name  = 'sea_water_x_velocity'
    nc.variables['u'].units      = 'm/s' 

    nc.variables['v'].long_name  = 'sea_water_y_velocity'
    nc.variables['v'].units      = 'm/s'   
    
    nc.close()   
    
    print('File created! >>>', name_file)    

def save2nc_interpolated_eNATL60_T(dir_save, name_file, time, dep, 
                                   lat, lon, tem):

    ''' 
    Save the interpolated eNATL60 model data to an nc. file. 
    lon and lat are 2D and define the new grid
    tem is the interpolated field
    '''
     
    nc = netcdf.Dataset(dir_save + name_file + '_T.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['dep'][:]   = dep
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    nc.variables['tem'][:]   = tem
    # nc.variables['sal'][:]   = sal    
    # nc.variables['u'][:]     = u
    # nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file + '_T.nc')  

def save2nc_interpolated_eNATL60_S(dir_save, name_file, time, dep, lat, lon, sal):

    ''' 
    Save the interpolated eNATL60 model data to an nc. file. 
    lon and lat are 2D and define the new grid
    sal is the interpolated field
    '''
     
    nc = netcdf.Dataset(dir_save + name_file + '_S.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['dep'][:]   = dep
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    # nc.variables['tem'][:]   = tem
    nc.variables['sal'][:]   = sal    
    # nc.variables['u'][:]     = u
    # nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file + '_S.nc')  

def save2nc_interpolated_eNATL60_U(dir_save, name_file, time, dep, lat, lon, u):

    ''' 
    Save the interpolated eNATL60 model data to an nc. file. 
    lon and lat are 2D and define the new grid
    u is the interpolated field
    '''  
    
    nc = netcdf.Dataset(dir_save + name_file + '_U.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['dep'][:]   = dep
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    # nc.variables['tem'][:]   = tem
    # nc.variables['sal'][:]   = sal    
    nc.variables['u'][:]     = u
    # nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file + '_U.nc') 
    
def save2nc_interpolated_eNATL60_V(dir_save, name_file, time, dep, lat, lon, v):

    ''' 
    Save the interpolated eNATL60 model data to an nc. file. 
    lon and lat are 2D and define the new grid
    v is the interpolated field
    '''  
     
    nc = netcdf.Dataset(dir_save + name_file + '_V.nc', 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['dep'][:]   = dep
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    # nc.variables['tem'][:]   = tem
    # nc.variables['sal'][:]   = sal    
    # nc.variables['u'][:]     = u
    nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file + '_V.nc')
    
def save2nc_interpolated_eNATL60(dir_save, name_file, time, dep, lat, lon,
                                 tem, sal, u, v):

    
    ''' Save the interpolated eNATL60 model data to an nc. file '''
     
    nc = netcdf.Dataset(dir_save + name_file, 'a', format='NETCDF3_CLASSIC')
    
    nc.variables['time'][:]  = time
    nc.variables['dep'][:]   = dep
    nc.variables['lat'][:]   = lat
    nc.variables['lon'][:]   = lon
    
    nc.variables['tem'][:]   = tem
    nc.variables['sal'][:]   = sal    
    nc.variables['u'][:]     = u
    nc.variables['v'][:]     = v

    nc.close() 
    
    print('Data saved! >>>', name_file)     


def open_eNATL60_interp_outputs(dir_outputs, region, period_for_model):
    
    '''
    Open interpolated eNATL60 model outputs. 
    '''
    
    if mdates.num2date(period_for_model[1]).month == 9:
        period = 'Sep'
        
    elif mdates.num2date(period_for_model[1]).month == 1:
        period = 'Jan'

    elif mdates.num2date(period_for_model[1]).month == 8:
        period = 'Aug'

    elif mdates.num2date(period_for_model[1]).month == 10:
        period = 'Oct'
        
    else:
        print('No valid period')
        
    if region == 'Med':    
        
        dir_save    = dir_outputs + 'eNATL60_MED_3D_int/'
        name_fileT  = 'eNATL60MEDBAL-BLB002_' + period + '_T.nc'
        name_fileS  = 'eNATL60MEDBAL-BLB002_' + period + '_S.nc'
        name_fileU  = 'eNATL60MEDBAL-BLB002_' + period + '_U.nc'
        name_fileV  = 'eNATL60MEDBAL-BLB002_' + period + '_V.nc'
        
        print('')
        print('Opening...', name_fileT)

    elif region == 'Atl':
        
        dir_save    = dir_outputs + 'eNATL60_ATL_3D_int/'
        name_fileT  = 'eNATL60COSNWA-BLB002_' + period + '_T.nc' 
        name_fileS  = 'eNATL60COSNWA-BLB002_' + period + '_S.nc' 
        name_fileU  = 'eNATL60COSNWA-BLB002_' + period + '_U.nc' 
        name_fileV  = 'eNATL60COSNWA-BLB002_' + period + '_V.nc' 
    
        print('')
        print('Opening...', name_fileT)
        
    # T    
    nc   = netcdf.Dataset(dir_save + name_fileT, 'r')
    time = nc.variables['time'][:].data  # days since 0001-01-01 00:00:00 UTC, plus one
    dep  = nc.variables['dep'][:] .data 
    lat  = nc.variables['lat'][:].data
    lon  = nc.variables['lon'][:].data
    tem  = nc.variables['tem'][:].data 
    nc.close() 
    
    # S
    nc   = netcdf.Dataset(dir_save + name_fileS, 'r')
    sal  = nc.variables['sal'][:].data
    nc.close() 
    
    # U
    nc   = netcdf.Dataset(dir_save + name_fileU, 'r')
    u  = nc.variables['u'][:].data
    nc.close()     
    
    # V
    nc   = netcdf.Dataset(dir_save + name_fileV, 'r')
    v  = nc.variables['v'][:].data 
    nc.close()     
    
    ssh = np.ones((u.shape[0], u.shape[2], u.shape[3])) * np.nan
    
    # There is an error in the interpolated files. 
    # We need to mask where salinity < 35 (irreal data)
    
    # for it in np.arange(sal.shape[0]):
    #     for iz in np.arange(sal.shape[1]):
    #         var = sal[it, iz,:,:]
    #         ilat, ilon = np.where(var>37.5)
    #         sal[it, iz, ilat, ilon]
    #         plt.figure()
    #         lon2d, lat2d = np.meshgrid(lon, lat)
    #         plt.scatter(lon2d[ilat, ilon], lat2d[ilat, ilon], 
    #                     c=sal[it, iz, ilat, ilon], cmap=plt.cm.jet)
    #         plt.colorbar()
            
            
    
    return time, lon[0,:], lat[:,0], dep, tem, sal, u, v, ssh  #ssh nan
    
def open_eNATL60_SSHi_surf_whole_domain(dir_outputs, region, period_for_model):
    
    '''
    Open interpolated SSH from the eNATL60 2D files 
    (data at surface and in the big regions). 
    
    '''
    
    if mdates.num2date(period_for_model[1]).month == 9:
        period = 'Sep'
        
    elif mdates.num2date(period_for_model[1]).month == 1:
        period = 'Jan'

    # elif mdates.num2date(period_for_model[1]).month == 8:
    #     period = 'Aug'

    # elif mdates.num2date(period_for_model[1]).month == 10:
    #     period = 'Oct'
        
    else:
        print('No valid period')                
            
    if region == 'Med':    
        
        dir_save        = dir_outputs + 'eNATL60_MED_2D_int/'
        name_files_SSH  = 'eNATL60MEDWEST-BLB002_SSHi_wd_' + period + '.nc'

        print('')
        print('Opening...', name_files_SSH)

    elif region == 'Atl':
        
        dir_save        = dir_outputs + 'eNATL60_ATL_2D_int/'
        name_files_SSH  = 'eNATL60NANFL-BLB002_SSHi_wd_'+ period + '.nc'

        print('')
        print('Opening...', name_files_SSH)
    
    # open file    
    nc   = netcdf.Dataset(dir_save + name_files_SSH, 'r')
    time = nc.variables['time'][:].data  # days since 0001-01-01 00:00:00 UTC, plus one
    lat  = nc.variables['lat'][:].data
    lon  = nc.variables['lon'][:].data
    ssh  = nc.variables['ssh'][:].data 
    nc.close() 
      
    
    return time, lon[0,:], lat[:,0], ssh  

def open_eNATL60_SSHi_surf_whole_domain_filter(dir_outputs, region, 
                                               period_for_model, scale):
    
    '''
    Open interpolated and filtered SSH from the eNATL60 2D files 
    (data at surface and in the big regions). 
    
    '''
    
    if mdates.num2date(period_for_model[1]).month == 9:
        period = 'Sep'
        
    elif mdates.num2date(period_for_model[1]).month == 1:
        period = 'Jan'

    # elif mdates.num2date(period_for_model[1]).month == 8:
    #     period = 'Aug'

    # elif mdates.num2date(period_for_model[1]).month == 10:
    #     period = 'Oct'
        
    else:
        print('No valid period')                
            
    if region == 'Med':    
        
        dir_save        = dir_outputs + 'eNATL60_MED_2D_int/'
        name_files_SSH  = 'eNATL60MEDWEST-BLB002_SSHi_wd_' + period + \
                           '_filtered_' +np.str(scale)+'km.nc'

        print('')
        print('Opening...', name_files_SSH)

    elif region == 'Atl':
        
        dir_save        = dir_outputs + 'eNATL60_ATL_2D_int/'
        name_files_SSH  = 'eNATL60NANFL-BLB002_SSHi_wd_'+ period +  \
                           '_filtered_' +np.str(scale)+'km.nc'

        print('')
        print('Opening...', name_files_SSH)
    
    # open file    
    nc   = netcdf.Dataset(dir_save + name_files_SSH, 'r')
    time = nc.variables['time'][:].data  # days since 0001-01-01 00:00:00 UTC, plus one
    lat  = nc.variables['lat'][:].data
    lon  = nc.variables['lon'][:].data
    ssh  = nc.variables['ssh'][:].data 
    nc.close() 
      
    
    return time, lon[0,:], lat[:,0], ssh  

def open_eNATL60_2D_wd_SSHi_rmean_filtered(dir_outputs, region, 
                                               period_for_model, scale):
    
    '''
    open_eNATL60_2D_wd_SSHi_rmean_filtered:
    
    - eNATL60_2D_wd: eNATL60 model outputs for the surface (2D) in the
                     whole domains (wd)
    - SSHi: SSH has been interpolated to a regular grid
    - rmean_filtered: then SSHi has been filtered with L=30km after removing
                      the spatial mean. 
    
    Open interpolated and filtered SSH from the eNATL60 2D files 
    (data at surface and in the big regions). 
    
    Note: this file has been created in Step00g, in which
    the interpolated files are opened, and SSHi is smoothed 
    with L=30 km after removing the spatial mean (to remove
    the contamination from the large scale that introduces
    high temporal variability).
    
    '''
    
    if mdates.num2date(period_for_model[1]).month == 9:
        period = 'Sep'
        
    elif mdates.num2date(period_for_model[1]).month == 1:
        period = 'Jan'

        
    else:
        print('No valid period')                
            
    if region == 'Med':    
        
        dir_save        = dir_outputs + 'eNATL60_MED_2D_int/'
        name_files_SSH  = 'eNATL60MEDWEST-BLB002_SSHi_wd_' + period + \
                           '_rmean_filtered_' +np.str(scale)+'km.nc'

        print('')
        print('Opening...', name_files_SSH)

    elif region == 'Atl':
        
        dir_save        = dir_outputs + 'eNATL60_ATL_2D_int/'
        name_files_SSH  = 'eNATL60NANFL-BLB002_SSHi_wd_'+ period +  \
                           '_rmean_filtered_' +np.str(scale)+'km.nc'

        print('')
        print('Opening...', name_files_SSH)
    
    # open file    
    nc   = netcdf.Dataset(dir_save + name_files_SSH, 'r')
    time = nc.variables['time'][:].data  # days since 0001-01-01 00:00:00 UTC, plus one
    lat  = nc.variables['lat'][:].data
    lon  = nc.variables['lon'][:].data
    ssh  = nc.variables['ssh'][:].data 
    nc.close() 
      
    
    return time, lon[0,:], lat[:,0], ssh  

def open_eNATL60_interp_outputs_grid(dir_outputs, region, period_for_model):
    
    time, lon, lat, dep, tem, sal, u, v, ssh = open_eNATL60_interp_outputs(dir_outputs, region, period_for_model)
    
    return lon, lat

def open_4D_model_outputs(dir_outputs, model, region, period_for_model):
    
    '''
    Open 4D model ouputs (3D + time)
    
    model options: 'CMEMS', 'WMOP', 'eNATL60'
    region options: 'Med', 'Atl'
    
    outputs: time, lon, lat, dep, tem, sal, u, v, ssh
    
    '''
    
    if model == 'CMEMS':
        
        time, lon, lat, dep, tem, sal, u, v, ssh = \
                         open_CMEMS_outputs(dir_outputs, region)

    elif model == 'WMOP':        
        time, lon, lat, dep, tem, sal, u, v, ssh = \
                         open_WMOP_3D_outputs(dir_outputs, region, period_for_model)

    elif model == 'eNATL60':    
        
        time, lon, lat, dep, tem, sal, u, v, ssh = \
                         open_eNATL60_interp_outputs(dir_outputs, region, period_for_model)        

                        
    return time, lon, lat, dep, tem, sal, u, v, ssh

def open_surf_model_outputs(dir_outputs, model, region, period_for_model):
    
    '''
    Open model outputs at the ocean surface (bigger region files): 
    files named 2D because the data is only at the surface (lon, lat), 
    but we also have the time variable.
    
    model options: 'CMEMS', 'WMOP', 'eNATL60'
    region options: 'Med', 'Atl'
    
    outputs: time, lon, lat, dep, tem, sal, u, v, ssh
    
    '''
    
    if model == 'CMEMS':
        #update for 2D
        print('no data for now')
        # time, lon, lat, dep, tem, sal, u, v, ssh = \
        #                   open_CMEMS_outputs(dir_outputs, region)

    elif model == 'WMOP':        

        time, lon, lat, tem, sal, u, v, ssh = \
                         open_WMOP_2D_outputs(dir_outputs, region, period_for_model)

    elif model == 'eNATL60': 
        #update for 2D
        print('no data for now') 
        # time, lon, lat, dep, tem, sal, u, v, ssh = \
        #                   open_eNATL60_interp_outputs(dir_outputs, region, period_for_model)        

                        
    return time, lon, lat, tem, sal, u, v, ssh

def limit_model_data(num_conf, tem, sal, u, v, lon, lat, time, dep, 
                     lon_ctd, lat_ctd, time_ctd,
                     lon_adcp=None, lat_adcp=None, time_adcp=None):
    
    ''' Limit model data with some extra margin '''
    
    if num_conf != '5':   # CTD and ADCP
        
        print('Limiting model data with ADCP and CTD data') 
        
        cond_lonm = np.logical_and(lon > min(lon_ctd.min(), lon_adcp.min()) - 0.25,
                               lon < max(lon_ctd.max(), lon_adcp.max()) + 0.25)
    
        cond_latm = np.logical_and(lat > min(lat_ctd.min(), lat_adcp.min()) - 0.25,
                               lat < max(lat_ctd.max(), lat_adcp.max()) + 0.25)
    
        cond_time = np.logical_and(time > min(time_ctd.min(), time_adcp.min()) - 1,
                               time < max(time_ctd.max(), time_adcp.max()) + 1)
        
    elif num_conf == '5': # only glider CTD data 
        
        print('Limiting model data only with CTD data') 
        
        cond_lonm = np.logical_and(lon > lon_ctd.min() - 0.25,
                                   lon < lon_ctd.max() + 0.25)
    
        cond_latm = np.logical_and(lat > lat_ctd.min() - 0.25,
                                   lat < lat_ctd.max() + 0.25)
    
        cond_time = np.logical_and(time > time_ctd.min() - 1,
                                   time < time_ctd.max() + 1)

        
    lonm_reg  = lon[cond_lonm]
    latm_reg  = lat[cond_latm]
    timem_reg = time[cond_time]
    depm_reg  = dep
    temm_reg  = tem[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    salm_reg  = sal[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    um_reg    = u[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    vm_reg    = v[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    
    return lonm_reg, latm_reg, timem_reg, depm_reg, temm_reg, salm_reg, \
           um_reg, vm_reg
           
def limit_exact_model_data(num_conf, tem, sal, u, v, lon, lat, time, dep, 
                     lon_ctd, lat_ctd, time_ctd,
                     lon_adcp=None, lat_adcp=None, time_adcp=None):
    
    ''' Limit model data within the sampling domain and time,
        no extra margin, to compute L empirical '''
    
    if num_conf != '5':   # CTD and ADCP
        
        print('Limiting model data with ADCP and CTD data') 
        
        cond_lonm = np.logical_and(lon >= min(lon_ctd.min(), lon_adcp.min()),
                               lon <= max(lon_ctd.max(), lon_adcp.max()))
    
        cond_latm = np.logical_and(lat >= min(lat_ctd.min(), lat_adcp.min()),
                               lat <= max(lat_ctd.max(), lat_adcp.max()))
    
        cond_time = np.logical_and(time >= min(time_ctd.min(), time_adcp.min()),
                               time <= max(time_ctd.max(), time_adcp.max()))
        
    elif num_conf == '5': # only glider CTD data 
        
        print('Limiting model data only with CTD data') 
        
        cond_lonm = np.logical_and(lon >= lon_ctd.min(),
                                   lon <= lon_ctd.max())
    
        cond_latm = np.logical_and(lat >= lat_ctd.min(),
                                   lat <= lat_ctd.max())
    
        cond_time = np.logical_and(time >= time_ctd.min(),
                                   time <= time_ctd.max())

        
    lonm_reg  = lon[cond_lonm]
    latm_reg  = lat[cond_latm]
    timem_reg = time[cond_time]
    depm_reg  = dep
    temm_reg  = tem[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    salm_reg  = sal[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    um_reg    = u[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    vm_reg    = v[cond_time][:,:, cond_latm,:][:,:,:,cond_lonm]
    
    return lonm_reg, latm_reg, timem_reg, depm_reg, temm_reg, salm_reg, \
           um_reg, vm_reg

def limit_exact_2D_model_data_all_vars(num_conf, tem, sal, u, v, ssh, lon, lat, time,  
                     lon_ctd, lat_ctd, time_ctd,
                     lon_adcp=None, lat_adcp=None, time_adcp=None):
    
    ''' Limit 2D model data with no extra margins, including all variables 
    in CMEMS and WMOP model data. Not used. '''
    
    if num_conf != '5':   # CTD and ADCP
        
        print('Limiting model data with ADCP and CTD data') 
        
        cond_lonm = np.logical_and(lon >= min(lon_ctd.min(), lon_adcp.min()),
                               lon <= max(lon_ctd.max(), lon_adcp.max()))
    
        cond_latm = np.logical_and(lat >= min(lat_ctd.min(), lat_adcp.min()),
                               lat <= max(lat_ctd.max(), lat_adcp.max()))
    
        cond_time = np.logical_and(time >= min(time_ctd.min(), time_adcp.min()),
                               time <= max(time_ctd.max(), time_adcp.max()))
        
    elif num_conf == '5': # only glider CTD data 
        
        print('Limiting model data only with CTD data') 
        
        cond_lonm = np.logical_and(lon >= lon_ctd.min(),
                                   lon <= lon_ctd.max())
    
        cond_latm = np.logical_and(lat >= lat_ctd.min(),
                                   lat <= lat_ctd.max())
    
        cond_time = np.logical_and(time >= time_ctd.min(),
                                   time <= time_ctd.max())

        
    lonm_reg  = lon[cond_lonm]
    latm_reg  = lat[cond_latm]
    timem_reg = time[cond_time]
    temm_reg  = tem[cond_time][:,cond_latm,:][:,:,cond_lonm]
    salm_reg  = sal[cond_time][:,cond_latm,:][:,:,cond_lonm]
    um_reg    = u[cond_time][:,cond_latm,:][:,:,cond_lonm]
    vm_reg    = v[cond_time][:,cond_latm,:][:,:,cond_lonm]
    ssh_reg   = ssh[cond_time][:,cond_latm,:][:,:,cond_lonm]
    
    return lonm_reg, latm_reg, timem_reg, temm_reg, salm_reg, \
           um_reg, vm_reg, ssh_reg


def limit_exact_2D_model_data(u, v, ssh, lon, lat, time,  
                     lon_ctd, lat_ctd, time_ctd):
    
    ''' Limit 2D model data to the domain limited by lon_ctd 
        and lat_ctd, and for a period +-1 day time_ctd. 
        Variables: u, v and ssh. 
        Can be applied to CMEMS, WMOP, eNATL60 outputs. 
        Used in Step 10.'''

    print('Limiting model data...''') 
        
    cond_lonm = np.logical_and(lon >= lon_ctd.min(),
                                   lon <= lon_ctd.max())
    
    cond_latm = np.logical_and(lat >= lat_ctd.min(),
                                   lat <= lat_ctd.max())
    
    # time range bigger because in Step10b we interpolate  
    # to have the fields to the OI map date    
    
    ind_time_min = np.argmin(np.abs(time-(time_ctd.min()-1)))
    ind_time_max = np.argmin(np.abs(time-(time_ctd.max()+1)))         
                
    lonm_reg  = lon[cond_lonm]
    latm_reg  = lat[cond_latm]
    timem_reg = time[ind_time_min:ind_time_max+1] 
    um_reg    = u[ind_time_min:ind_time_max+1][:,cond_latm,:][:,:,cond_lonm]
    vm_reg    = v[ind_time_min:ind_time_max+1][:,cond_latm,:][:,:,cond_lonm]
    ssh_reg   = ssh[ind_time_min:ind_time_max+1][:,cond_latm,:][:,:,cond_lonm]
    
    
    return lonm_reg, latm_reg, timem_reg,  \
           um_reg, vm_reg, ssh_reg

def limit_bigger_2D_model_data(u, v, ssh, lon, lat, time,  
                     lon_ctd, lat_ctd, time_ctd, dbigger):
    
    ''' 
       Extract model data:
               * in a domain bigger than the sampling domain, 
                 defined by "dbigger""
               * for the sampling period +- 1 day     
       Variables: u, v and ssh. 
       Can be applied to CMEMS, WMOP, eNATL60 outputs. 
       Used in Step 10c.'''

    print('Limiting model data...''') 
        
    cond_lonm = np.logical_and(lon >= (lon_ctd.min()-dbigger),
                                   lon <= (lon_ctd.max()+dbigger))
    
    cond_latm = np.logical_and(lat >= (lat_ctd.min()-dbigger),
                                   lat <= (lat_ctd.max()+dbigger))
    
    # time range bigger because in Step10b we interpolate  
    # to have the fields to the OI map date    
    
    ind_time_min = np.argmin(np.abs(time-(time_ctd.min()-1)))
    ind_time_max = np.argmin(np.abs(time-(time_ctd.max()+1)))         
                
    lonm_reg  = lon[cond_lonm]
    latm_reg  = lat[cond_latm]
    timem_reg = time[ind_time_min:ind_time_max+1] 
    um_reg    = u[ind_time_min:ind_time_max+1][:,cond_latm,:][:,:,cond_lonm]
    vm_reg    = v[ind_time_min:ind_time_max+1][:,cond_latm,:][:,:,cond_lonm]
    ssh_reg   = ssh[ind_time_min:ind_time_max+1][:,cond_latm,:][:,:,cond_lonm]
    
    
    return lonm_reg, latm_reg, timem_reg,  \
           um_reg, vm_reg, ssh_reg
           
def coord_SWOT_swath_Med():
    
    ''' 
    Open swath of SWOT coordinates in the Mediterranean Sea. 
    Files from Laura Gómez Navarro
    '''
    
    dir_swaths = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/swath_swot/'
    
    file_swaths1 = 'MED_fastPhase_1km_swotFAST_grid_p009.nc'
    file_swaths2 = 'MED_fastPhase_1km_swotFAST_grid_p022.nc'
    
    nc    = netcdf.Dataset(dir_swaths + file_swaths1, 'r')
    latsw1  = nc.variables['lat'][:]   
    lonsw1  = nc.variables['lon'][:]  
    x_ac1   = nc.variables['x_ac'][:]  # "Across track distance from nadir"
    
    lonnd1  = nc.variables['lon_nadir'][:] 
    latnd1  = nc.variables['lat_nadir'][:] 
    nc.close()     

    nc    = netcdf.Dataset(dir_swaths + file_swaths2, 'r')
    latsw2  = nc.variables['lat'][:]   
    lonsw2  = nc.variables['lon'][:]  
    x_ac2   = nc.variables['x_ac'][:]  # "Across track distance from nadir"

    lonnd2  = nc.variables['lon_nadir'][:] 
    latnd2  = nc.variables['lat_nadir'][:]     
    
    nc.close()  
    
    return lonsw1, latsw1, lonsw2, latsw2, lonnd1, latnd1, lonnd2, latnd2 

def coord_SWOT_swath_Atl():

    ''' 
    Open swath of SWOT coordinates in the northwest Atlantic. 
    Files from Laura Gómez Navarro
    '''
    
    dir_swaths = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/swath_swot/'
    
    file_swaths1 = 'eNAtl60_wT_gst_hourly_FMA_fast_grid_p013.nc'
    file_swaths2 = 'eNAtl60_wT_gst_hourly_FMA_fast_grid_p026.nc'  
    
    nc    = netcdf.Dataset(dir_swaths + file_swaths1, 'r')
    latsw1  = nc.variables['lat'][:]   
    lonsw1_360  = nc.variables['lon'][:]  
    #x_ac1   = nc.variables['x_ac'][:]  # "Across track distance from nadir"
    
    lonnd1_360  = nc.variables['lon_nadir'][:] 
    latnd1  = nc.variables['lat_nadir'][:] 
    nc.close()     

    nc    = netcdf.Dataset(dir_swaths + file_swaths2, 'r')
    latsw2  = nc.variables['lat'][:]   
    lonsw2_360  = nc.variables['lon'][:]  
    #x_ac2   = nc.variables['x_ac'][:]  # "Across track distance from nadir"

    lonnd2_360  = nc.variables['lon_nadir'][:] 
    latnd2  = nc.variables['lat_nadir'][:]     
    
    
    nc.close()  
    
    lonsw1 = np.copy(lonsw1_360)
    lonsw1[lonsw1_360>180] = lonsw1_360[lonsw1_360>180] - 360 
    
    lonsw2 = np.copy(lonsw2_360)
    lonsw2[lonsw2_360>180] = lonsw2_360[lonsw2_360>180] - 360 

    lonnd1 = np.copy(lonnd1_360)
    lonnd1[lonnd1_360>180] = lonnd1_360[lonnd1_360>180] - 360 
    
    lonnd2 = np.copy(lonnd2_360)
    lonnd2[lonnd2_360>180] = lonnd2_360[lonnd2_360>180] - 360 

    
    return lonsw1, latsw1, lonsw2, latsw2, lonnd1, latnd1, lonnd2, latnd2 

def coord_SWOT_swath(region):
    
    if region == 'Med':
        
        lonsw1, latsw1, lonsw2, latsw2, lonnd1, latnd1, lonnd2, latnd2 = \
                                                       coord_SWOT_swath_Med()
    elif region == 'Atl':
        lonsw1, latsw1, lonsw2, latsw2, lonnd1, latnd1, lonnd2, latnd2 = \
                                                       coord_SWOT_swath_Atl()
        
    return lonsw1, latsw1, lonsw2, latsw2, lonnd1, latnd1, lonnd2, latnd2



def length_lon_lat_degs(lat_mean_deg):
      
  ''' 
  Function to infer the length of a degree of longitude
  and the length of a degree of latitude
  at a specific latitude position. 
  Assuming that the Earth is an ellipsoid.
  
  input:  latitude in DEGREES!!!
  output: length_deg_lon, length_deg_lat in meters
  
  from:
      https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude
      https://en.wikipedia.org/wiki/Latitude#Length_of_a_degree_of_latitude
  '''  
  
  ''' Earth parameters '''
  lat_mean_rad = lat_mean_deg*(np.pi/180)  #in radians
  
  a = 6378137.0                # m (equatorial radius)
  b = 6356752.3142             # m  (polar radius)
  
  ecc_2 = (a**2 - b**2) / a**2 # eccentricity squared


  ''' The length of a degree of longitude is... '''
  
  divident_lon = (a*np.pi/180) * np.cos(lat_mean_rad)
  divisor_lon  = np.sqrt(1 - (ecc_2*np.sin(lat_mean_rad)*np.sin(lat_mean_rad)))
  
  length_deg_lon = divident_lon / divisor_lon 
  
  
  ''' The length of a degree of latitude is... '''
  
  divident_lat = (a*np.pi/180) * (1 - ecc_2)
  divisor_lat  = (1 - (ecc_2 * np.sin(lat_mean_rad) * np.sin(lat_mean_rad)))**(3/2)
  
  length_deg_lat = divident_lat / divisor_lat
  
  
  return length_deg_lon, length_deg_lat   

def compute_vgeo_from_mdt(lon_mdt, lat_mdt, mdt):

    ''' 
    Calculate geostrophic velocity from MDT, ADT, SLA, dh (in m) 
    u = -g/f (dMDT/dy)
    v = g/f (dMDT/dx)
    '''
    
    
    p    = gsw.p_from_z(0, lat_mdt.mean())
    grav = gsw.grav(lat_mdt.mean(), p)
    f    = gsw.f(lat_mdt.mean())
    
    dlon = np.diff(lon_mdt).min() # in deg
    dlat = np.diff(lat_mdt).min() # in deg

    # length [m] of a degree of longitude and latitude
    length_deg_lon, length_deg_lat = length_lon_lat_degs(lat_mdt.mean())    

    dx = dlon * length_deg_lon # in m
    dy = dlat * length_deg_lat # in m 

    grad_mdt = dt.grad2d_2d(mdt,dx,dy) # computes d(data)/dx , d(data)/dy 
    
    dMDTdx = grad_mdt.real
    dMDTdy = grad_mdt.imag
    
    ug = - (grav/f) * dMDTdy
    vg = (grav/f) * dMDTdx
    
    lon_mdt2d, lat_mdt2d = np.meshgrid(lon_mdt, lat_mdt)

    return lon_mdt2d, lat_mdt2d, ug, vg
