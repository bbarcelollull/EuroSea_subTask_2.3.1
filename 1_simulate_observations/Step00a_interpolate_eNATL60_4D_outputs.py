#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
from matplotlib              import dates as mdates
import EuroSea_toolbox       as to



'''
Code to open eNATL60 4D files, extract desired data, and interpolate them
onto a regular 2D grid for each time step and depth (griddata). 
Then save interpolated model outputs in a .nc file. 
Use interpolated data to extract pseudo-observations in Step 2. 

written by Bàrbara Barceló-Llull on 02-03-2021 at IMEDEA (Mallorca, Spain)
'''

def define_new_grid(hor_res, lonp, latp):
    
    hor_res = 1/60 #deg
    lon_new = np.arange(np.nanmin(lonp), np.nanmax(lonp), hor_res)
    lat_new = np.arange(np.nanmin(latp), np.nanmax(latp), hor_res)
    
    lon_new2d, lat_new2d = np.meshgrid(lon_new, lat_new)
        
    return lon_new2d, lat_new2d


if __name__ == '__main__':
    
    
    
    # in laptop:
    dir_outputs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/model_outputs/'    
    # in lluna:
    # dir_outputs = '/home/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/model_outputs/'    
        
    
    ''' ----- If running in lluna uncomment this ----- '''
    
    # # Number of days since 0001-01-01 00:00:00 UTC, plus one.
    # old_epoch = '0000-12-31T00:00:00'
    # # configure mdates with the old_epoch
    # mdates.set_epoch(old_epoch)  # old epoch (pre MPL 3.3)

    # print('')
    # print('mdates works with days since... ', mdates.get_epoch())    
    
    ''' ----------------------------------------- '''
    
    model       = 'eNATL60' 
    regions     = ['Med']#['Med', 'Atl'] # Med or 'Atl' 
    periods     = ['Sep'] #['Aug'] #['Oct']#['Sep', 'Jan'] # 'Sep' or 'Jan' , on 26 April 2021 add August and October for the Med
    
    for region in regions:
        for period in periods:
            
            print('')
            print('------------------------------------------')
            print('')
            print('Interpolating... ', model)
            print('in the region... ', region)
            print('for the period... ', period)
            print('')
            print('------------------------------------------')
            print('')
            
            ''' Period of the outputs '''
    
            # Sep: from 31-8-2009 to 6-9-2009
            # Jan: from 31-12-2009 to 5-1-2010
    
            if period == 'Sep':
        
                d_ini_Sep    = mdates.date2num(datetime(2009, 8, 31, 0, 0, 0)) 
                d_fin_Sep    = mdates.date2num(datetime(2009, 9, 6, 0, 0, 0)) 
        
                period_for_model = [d_ini_Sep, d_fin_Sep]
        
            elif period == 'Jan':
        
                d_ini_Jan    = mdates.date2num(datetime(2009, 12, 31, 0, 0, 0)) 
                d_fin_Jan    = mdates.date2num(datetime(2010, 1, 5, 0, 0, 0)) 
    
                period_for_model = [d_ini_Jan, d_fin_Jan]

            elif period == 'Aug':
                # Added on 26-4-2021, only in the Med -> for Baptiste reconstruction
                
                d_ini_Aug    = mdates.date2num(datetime(2009, 7, 31, 0, 0, 0)) 
                d_fin_Aug    = mdates.date2num(datetime(2009, 8, 5, 0, 0, 0)) 
    
                period_for_model = [d_ini_Aug, d_fin_Aug]    

            elif period == 'Oct':
                # Added on 26-4-2021, only in the Med -> for Baptiste reconstruction
                
                d_ini_Oct    = mdates.date2num(datetime(2009, 9, 30, 0, 0, 0)) 
                d_fin_Oct    = mdates.date2num(datetime(2009, 10, 5, 0, 0, 0)) 
    
                period_for_model = [d_ini_Oct, d_fin_Oct]   
                
                
            ''' Open original eNATL60 model outputs '''
    
            time, lon, lat, dep, tem, sal, u, v, ssh = \
                    to.open_original_eNATL60_outputs(dir_outputs, region, period_for_model)      
    
 
    
            ''' Define new grid: '''
    
            hor_res = 1/60 #deg
            lon_new2d, lat_new2d = define_new_grid(hor_res, lon, lat)
    
            #check grid
            # plt.figure()
            # plt.scatter(lon.flatten(),lat.flatten()) 
            # plt.scatter(lon_new2d.flatten(),lat_new2d.flatten(), c='r',
            #             marker='x') 
            

            ''' Save data to an .nc file '''
    
            # Define directory and file name
            if region == 'Med':    
        
        
                dir_save   = dir_outputs + 'eNATL60_MED_3D_int/'
                name_file  = 'eNATL60MEDBAL-BLB002_' + period # + '.nc'

            elif region == 'Atl':
          
        
                dir_save   = dir_outputs + 'eNATL60_ATL_3D_int/'
                name_file  = 'eNATL60COSNWA-BLB002_' + period # + '.nc' 
        
    
            ''' Interpolate data in each time step and depth to a regular 2D axis '''

            #>>>>>>>>>>> TEMPERATURE <<<<<<<<<<<<
    
            
            print ('Interpolating temperature...')
            tem_int, Tori_mask = to.interp_original_eNATL60_TS(tem, lon, lat, time, lon_new2d, lat_new2d)
            
            # Create file
            to.create_nc_interpolated_eNATL60_T(dir_save, name_file, time, dep, lon_new2d)
    
            # save data
            print('')
            print ('Saving temperature...')    
            to.save2nc_interpolated_eNATL60_T(dir_save, name_file, time, dep, 
                                      lat_new2d, lon_new2d, tem_int)
     
            
    
            # >>>>>>>>>>> SALINITY <<<<<<<<<<<<
     
            print('')
            print ('Interpolating salinity...')
            sal_int, Sori_mask = to.interp_original_eNATL60_TS(sal, lon, lat, time, lon_new2d, lat_new2d)

            # Create file
            to.create_nc_interpolated_eNATL60_S(dir_save, name_file, time, dep, lon_new2d)   
    
            # save data
            print('')
            print ('Saving salinity...')      
            to.save2nc_interpolated_eNATL60_S(dir_save, name_file, time, dep, 
                                        lat_new2d, lon_new2d, sal_int)    
    
    
    
            # >>>>>>>>>>> U <<<<<<<<<<<<
      
            print('')
            print ('Interpolating U...')
            u_int   = to.interp_original_eNATL60_UV(u, Tori_mask, tem_int, lon, lat, time, lon_new2d, lat_new2d)

            # Create file
            to.create_nc_interpolated_eNATL60_U(dir_save, name_file, time, dep, lon_new2d)  
    
            # save data
            print('')
            print ('Saving U...')      
            to.save2nc_interpolated_eNATL60_U(dir_save, name_file, time, dep,  
                                        lat_new2d, lon_new2d, u_int)        
    
    
    
            # >>>>>>>>>>> V <<<<<<<<<<<<
               
            print('')
            print ('Interpolating V...')    
            v_int   = to.interp_original_eNATL60_UV(v, Tori_mask, tem_int, lon, lat, time, lon_new2d, lat_new2d)

            # Create file
            to.create_nc_interpolated_eNATL60_V(dir_save, name_file, time, dep, lon_new2d) 
    
            # save data
            print('')
            print ('Saving V...')      
            to.save2nc_interpolated_eNATL60_V(dir_save, name_file, time, dep,  
                                       lat_new2d, lon_new2d, v_int) 

  
    