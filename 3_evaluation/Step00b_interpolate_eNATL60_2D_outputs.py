#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
from matplotlib              import dates as mdates
import EuroSea_toolbox       as to


'''
Code to open eNATL60 2D files, extract desired data, and interpolate them
onto a regular 2D grid for each time step (griddata). 
Then save interpolated model outputs in a .nc file. 
Use interpolated data to compare with reconstructed fields. 

written by Bàrbara Barceló-Llull on 03-11-2021 at IMEDEA (Mallorca, Spain)
'''


if __name__ == '__main__':
    
    
    
    # a mac:
    dir_outputs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/model_outputs/'    
    # a lluna:
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
    regions     = ['Atl'] #['Med', 'Atl'] # Med or 'Atl' 
    periods     = ['Jan'] #['Aug'] #['Oct']#['Sep', 'Jan'] # 'Sep' or 'Jan' , on 26 April 2021 add August and October for the Med
    
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
    
            # time, lon, lat, dep, tem, sal, u, v, ssh = \
            #         to.open_original_eNATL60_outputs(dir_outputs, region, period_for_model)      
    
            timeo, lono, lato, ssho, uo, vo = \
              to.open_original_2D_eNATL60_outputs(dir_outputs, region, period_for_model)
             
            print('')
            print('Time period of the model outputs...')
            print(mdates.num2date(timeo[0]).strftime("%Y-%m-%d %H:%M"))
            print(mdates.num2date(timeo[-1]).strftime("%Y-%m-%d %H:%M"))
            


            ''' Save data to an .nc file '''
    
            # Define directory and file name
            if region == 'Med':    

                dir_save   = dir_outputs + 'eNATL60_MED_2D_int/'
                #name_file  = 'eNATL60MEDBAL-BLB002_' + period # + '.nc' #3D Med files
                name_file  = 'eNATL60MEDWEST-BLB002_' + period           #2D Med files
            
            elif region == 'Atl':

                dir_save   = dir_outputs + 'eNATL60_ATL_2D_int/'
                #name_file  = 'eNATL60COSNWA-BLB002_' + period # + '.nc' 
                name_file  = 'eNATL60NANFL-BLB002_' + period
        
    
            ''' Interpolate data (ssho, uo, vo) in each time step to a regular 2D axis '''

            for it in np.arange(timeo.shape[0]):
                
                print('')
                print('interpolating data, it = ' + np.str(it) + '/' + np.str(timeo.shape[0]))

                loni, lati, ssh_int_it = to.interp_original_2D_eNATL60( \
                                    region, dir_outputs, ssho[it], lono, lato)
                
                loni, lati, u_int_it   = to.interp_original_2D_eNATL60( \
                                     region, dir_outputs, uo[it], lono, lato)
                
                loni, lati, v_int_it   = to.interp_original_2D_eNATL60( \
                                     region, dir_outputs, vo[it], lono, lato)
                    
                if it == 0:
                    ssh_int = np.ones((ssho.shape[0], ssh_int_it.shape[0],
                                       ssh_int_it.shape[1])) * np.nan
                    u_int   = np.ones((ssho.shape[0], ssh_int_it.shape[0],
                                       ssh_int_it.shape[1])) * np.nan
                    v_int   = np.ones((ssho.shape[0], ssh_int_it.shape[0],
                                       ssh_int_it.shape[1])) * np.nan    
                    
                ssh_int[it] = ssh_int_it
                u_int[it]   = u_int_it
                v_int[it]   = v_int_it
               
               
                
            ''' Save interpolated data '''
            
            # Create .nc file (1 file with all interpolated variables)
                
            to.create_nc_interpolated_2D_eNATL60(dir_save, name_file, timeo, loni)
    
           
            # save data into the .nc file
            
            to.save2nc_interpolated_2D_eNATL60(dir_save, name_file, timeo,  
                                   lati, loni, ssh_int, u_int, v_int)
