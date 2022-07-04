#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import netCDF4               as netcdf
import pickle


"""
Step 9b: Extract reconstructed data at the upper layer
         to do comparisons in Step 12.
         
         Save data with the same format as the ocean truth .pkl file.
    

written by Bàrbara Barceló-Llull on 06-04-2022 at IMEDEA (Mallorca, Spain)

"""

            
if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directories '''
    
    dir_OIdata    = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/reconstructed_fields/spatio-temporal_OI_all_conf/'
    dir_pseudoobs = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'
    dir_figures   = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/reconstructed_fields/spatio-temporal_OI/'
    dir_dic       = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/comparison/spatio-temporal_OI/'

    ''' Which model and region? '''
    
    model         = 'WMOP' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Med' #'Atl' or 'Med'
 
    ''' Start dictionary where to save the data '''
    
    dic_all  = {}

    
    '''
    >>>>>> Interpolated fields <<<<<<
    '''
    
    oi_files  = sorted(glob.glob(dir_OIdata + region + '*_'+model + '*T.nc'))
    
    if  np.logical_and(region == 'Med', model == 'eNATL60'):
        
        oi_files_selec = oi_files[:-3] + [oi_files[-1]]
        
        oi_files = np.copy(oi_files_selec)
        

    for icf, file in enumerate(oi_files): #[oi_files[4]]): #oi_files): #[oi_files[4]]): #oi_files): 


        name_conf  = file[96:-5]
        num_conf   = name_conf[:33] 
        
        print('')
        print('--------------------------------------')
        print('')
        print('Configuration file...', name_conf)
        print('')
        print('configuration...', num_conf)
        print('')
                
        
        ''' Read .nc file with interpolated fields '''
           
        ncT      = netcdf.Dataset(dir_OIdata + name_conf + '_T.nc', 'r')
        ptem     = ncT.variables['ptem'][:].data 
        eptem    = ncT.variables['error'][:].data    
        lon      = ncT.variables['longitude'][:].data
        lat      = ncT.variables['latitude'][:].data   
        dep      = ncT.variables['depth'][:].data  
        time_map = ncT.variables['time'][:].data[0]
        ncT.close() 
        
        ncS   = netcdf.Dataset(dir_OIdata + name_conf + '_S.nc', 'r')
        psal  = ncS.variables['psal'][:].data 
        epsal = ncS.variables['error'][:].data          
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
        
        SPg = np.sqrt(ug**2 + vg**2)
        
        ''' Depth layer for the comparison '''
        
        iz = np.argmin(dep)

        print('Comparison done at DEPTH [m] = ', dep[iz])  

        
        ''' Save data '''
            
        dic_ref  = {}

        dic_ref.update({'lon'     : lon,
                        'lat'     : lat,
                        'dh'      : dh[iz], 
                        'ug'      : ug[iz], 
                        'vg'      : vg[iz],
                        'speedg'  : SPg[iz],
                        'Rog'     : Rog[iz],
                        'dep_map' : dep[iz],
                        'time_map': time_map}) 
        
            
        dic_all.update({name_conf[:-13]: dic_ref})
        
    
    ''' Save data to do statistics '''
    
    f_rec = open(dir_dic + region + '_' + model + '_rec_fields_upper_layer.pkl','wb')
    pickle.dump(dic_all,f_rec)
    f_rec.close()   
    
    