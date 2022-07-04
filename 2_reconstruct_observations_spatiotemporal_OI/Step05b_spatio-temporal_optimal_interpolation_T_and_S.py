#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import netCDF4               as netcdf
from matplotlib              import dates as mdates
import EuroSea_toolbox       as to
import Tools_OI              as toi 
from scipy                   import interpolate


"""
Step 5b: Reconstruct pseudo-observations of 
T and S applying linear interpolation on the vertical and
the spatio-temporal optimal interpolation horizontally. 

For all configurations, in the Atlantic and Mediterranean,
for all models. 

written by Bàrbara Barceló-Llull on 01-04-2022 at IMEDEA (Mallorca, Spain)

"""


def spatiotemporal_OI_3Dvar(dep, lono, lato, timeo, varo,
                            loni, lati, timei,
                            Lx, Ly, Lt,
                            ang, eps, mfield):
    
    
        ''' Optimal interpolation for each depth layer '''
        
        varint = np.ones((dep.shape[0], lati.shape[0], loni.shape[0]))*np.nan
        errint = np.ones((dep.shape[0], lati.shape[0], loni.shape[0]))*np.nan
        print('Start optimal interpolation...')


        for izoi, dd in enumerate(dep):
            print(dd)

            lonikm2d, latikm2d, timeOI, varint[izoi], errint[izoi] = \
                          toi.compute_OI_2d_time(lono, lato, timeo, 
                                                 varo[:, izoi],
                                                 loni, lati, timei, 
                                                 Lx, Ly, Lt, 
                                                  ang, eps, mfield)


        
        return lonikm2d, latikm2d, timeOI, varint, errint    


def create_nc(savedirnc, filename, svar, lonOI, latOI, dep):
    
    '''
    Create a netcdf file
    '''
    
    nc = netcdf.Dataset(savedirnc + filename, 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dep', dep.shape[0])   
    nc.createDimension('lat', lonOI.shape[0])    
    nc.createDimension('lon', lonOI.shape[1])    
    nc.createDimension('dtime', 1)
    
    # Create the variables...
    nc.createVariable(svar,        'f4', ('dep', 'lat', 'lon'))
    nc.createVariable('error',     'f4', ('dep', 'lat', 'lon'))
    nc.createVariable('longitude', 'f4', ('lat', 'lon'))
    nc.createVariable('latitude',  'f4', ('lat', 'lon'))
    nc.createVariable('depth',     'f4', ('dep'))
    nc.createVariable('time',      'f4', ('dtime'))

    
    # Write in variable attributes...
    if svar == 'ptem':
        nc.variables[svar].long_name  = 'Potential Temperature objectively interpolated'
        nc.variables[svar].units      = '[deg C]'
        
    elif svar == 'psal':
        nc.variables[svar].long_name  = 'Practical Salinity objectively interpolated'
        nc.variables[svar].units      = '[PSU]'
    
    elif svar == 'u':
        
        nc.variables[svar].long_name  = 'Meridional component of the ADCP velocity objectively interpolated'
        nc.variables[svar].units      = '[m/s]'

    elif svar == 'v':
        
        nc.variables[svar].long_name  = 'Zonal component of the ADCP velocity objectively interpolated'
        nc.variables[svar].units      = '[m/s]'
        
    else:
        print ('Provide the variable!')
        
     
    nc.variables['error'].long_name    = 'Normalized error of the OI'
    nc.variables['error'].units        = ''      
        
    nc.variables['longitude'].long_name    = 'Longitude'
    nc.variables['longitude'].units        = '[degrees]'  
    
    nc.variables['latitude'].long_name     = 'Latitude'
    nc.variables['latitude'].units         = '[degrees]' 
    
    nc.variables['depth'].long_name        = 'Depth'
    nc.variables['depth'].units            = '[m]' 

    nc.variables['time'].long_name        = 'Time of the map'
    nc.variables['time'].units            = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.' 
    
    nc.close()   
    
    print('File created! >>>', filename)

def save_to_nc(savedirnc, filename, \
               svar, varint, errint, \
               lonOI, latOI, timei, dep):

    # ------------------------------------------
    # Save data
    # ------------------------------------------
     
    nc = netcdf.Dataset(savedirnc + filename, 'a', format='NETCDF3_CLASSIC')
    nc.variables[svar][:]           = varint
    nc.variables['error'][:]        = errint
    nc.variables['longitude'][:]    = lonOI
    nc.variables['latitude'][:]     = latOI
    nc.variables['depth'][:]        = dep
    nc.variables['time'][:]         = timei
    nc.close() 
    
    print('Data saved! >>>', filename)


            
if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directory to save figures '''
    
    dir_fig       = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/OI/'
    dir_OIdata    = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/reconstructed_fields/spatio-temporal_OI_all_conf/'
    dir_save      = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'

    # in lluna: 
    # dir_fig       = '/home/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/OI/'
    # dir_OIdata    = '/home/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/reconstructed_fields/spatio-temporal_OI_all_conf/'
    # dir_save      = '/home/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'    

    ''' ----- If running in lluna uncomment these lines ----- '''
    
    # # Number of days since 0001-01-01 00:00:00 UTC, plus one.
    # old_epoch = '0000-12-31T00:00:00'
    # # configure mdates with the old_epoch
    # mdates.set_epoch(old_epoch)  # old epoch (pre MPL 3.3)

    # print('')
    # print('mdates works with days since... ', mdates.get_epoch())     
    # print('')
    
    ''' Which model and region? '''
    
    model         = 'eNATL60' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Atl' #'Atl' or 'Med'
    
    print('')
    print('---------------------')
    print(model)
    print(region)
    print('---------------------')
    print('')
    
    ''' Spatial scales ''' 
        
    lx     = 20 # km
    ly     = lx
        
    ''' Temporal correlation scale '''
    
    lt = 10 # days
    
    ''' Noise-to-signal error '''
    eps_sc = 0.03 # as in PRE-SWOT 
    eps_vc = 0.18 # as in PRE-SWOT 
            
    ''' Parameter for the OI '''
    ang_sc = 0
    ang_vc = 0
    
    # 'scalar' or a 'plane' (scalar for vector 
    # components and plane for a scalar quantity)
    mfield_sc  = 'plane' 
    mfield_vc  = 'scalar'     
    
    ''' Vertical resolution (for vertical interpolation) '''
    
    dz = 5 #m
    
    ''' Horizontal resolution new grid for OI '''
    
    dx_km = 2 # km 
    
    ''' Date of the resulting map: central or first '''
    
    date2use = 'central'
 
    print('')
    print('------ OI parameters ------')
    print('')
    print('using Lx = Ly = [km] ', lx)
    print('using Lt = [days] ', lt)
    print('using noise-to-signal error (T and S)...', eps_sc)
    print('using noise-to-signal error (U and V)...', eps_vc)
    print('')       
    print('dz = [m] ', dz)
    print('dx = dy = [km] ', dx_km)
    print('')
    print('interpolation to the central date')
    print('')
    
    
    '''
    >>>>>> OPEN PSEUDO-OBSERVATIONS <<<<<<
    '''
    
    pobs_files  = sorted(glob.glob(dir_save + region + '*_'+model+'.nc'))
    
    for file in pobs_files: #[pobs_files[-1]]: #pobs_files: #[pobs_files[-1]]: #start with the reference configuration
        
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
        time_ctd  = nc.variables['time_ctd'][:].data   
        dep_ctd   = nc.variables['dep_ctd'][:].data 
        lat_ctd   = nc.variables['lat_ctd'][:].data   
        lon_ctd   = nc.variables['lon_ctd'][:].data 
    
        tem_ctd   = nc.variables['tem_ctd'][:].data      
        sal_ctd   = nc.variables['sal_ctd'][:].data  

        if num_conf != '5':  
            
            print('opening ADCP pseudo-observations')
            time_adcp = nc.variables['time_adcp'][:].data  
            dep_adcp  = nc.variables['dep_adcp'][:].data  
            lat_adcp  = nc.variables['lat_adcp'][:].data   
            lon_adcp  = nc.variables['lon_adcp'][:].data      
    
            u_adcp    = nc.variables['u_adcp'][:].data  
            v_adcp    = nc.variables['v_adcp'][:].data
                        
            nc.close() 
        
        ''' Vertical interpolation for T and S '''
        
        # dz = 5 #m
        dep_ctd_new = np.arange(dep_ctd.min(),dep_ctd.max()+1, dz)
        
        tem_ctd_vi = np.zeros((tem_ctd.shape[0], dep_ctd_new.shape[0]))
        sal_ctd_vi = np.zeros((sal_ctd.shape[0], dep_ctd_new.shape[0]))

    
        for ip in np.arange(tem_ctd.shape[0]): #for each CTD profile
            
          # Interp1d (lineal) 
          f_interp_T = interpolate.interp1d(dep_ctd, tem_ctd[ip,:],  \
                                  fill_value=np.nan,bounds_error=False)

          f_interp_S = interpolate.interp1d(dep_ctd, sal_ctd[ip,:],  \
                                  fill_value=np.nan,bounds_error=False)
              
          tem_ctd_vi[ip,:]= f_interp_T(dep_ctd_new)
          sal_ctd_vi[ip,:]= f_interp_S(dep_ctd_new)

        ''' Define new grid for the optimal interpolation '''
        
        loni_min = lon_ctd.min()
        loni_max = lon_ctd.max()
    
        lati_min = lat_ctd.min()
        lati_max = lat_ctd.max()
    

        # length of a degree of longitude and latitude
        lat_ref = 40
        length_deg_lonp, length_deg_latp = to.length_lon_lat_degs(lat_ref)    
    
        # cast resolution from km to longitude and latitude degrees
        dlon_deg = dx_km /(length_deg_lonp/1000)
        dlat_deg = dx_km /(length_deg_latp/1000)

        
        loni = np.arange(loni_min, loni_max, dlon_deg)
        lati = np.arange(lati_min, lati_max, dlat_deg)
    
        [lonOI, latOI] = np.meshgrid(loni, lati)
        
        # plt.figure()
        # plt.scatter(lon_ctd, lat_ctd, marker='o', c='k')
        # plt.scatter(lonOI.flatten(), latOI.flatten(), 
        #             marker='x', c='r')

        ''' Time of the optimally interpolated map '''
        
        if date2use == 'central':
            
            
            timei = ((time_ctd.max() - time_ctd.min())/2) + time_ctd.min() 
    
            print('')
            print('------------------------------------')
            print('Time of the OI maps = ', mdates.num2date(timei).strftime("%Y-%m-%d %H:%M"))
            print('Time min CTD = ', mdates.num2date(time_ctd.min()).strftime("%Y-%m-%d %H:%M"))
            print('Time max CTD = ', mdates.num2date(time_ctd.max()).strftime("%Y-%m-%d %H:%M"))
            print('------------------------------------')
            print('')
        
        
        ''' Test for noise-to-signal errors '''
        
        # ang_sc = 0
        # eps_sc = 0.03
        
        # From PRE-SWOT report for SALINITY: We believe the final salinities 
        # are good to ± 0.002, as can be seen in Figure 1.
        std_inst_error_S = 0.002
        std_inst_error_T = 0.01 # Gasparin et al., 2019 Argo T error
        
        print('')
        print('------ noise-to-signal error analysis ------')
        
        print('')
        print('>>>>> Using all data ')
        print('')
        print('STD of the original data - T...', np.nanstd(tem_ctd_vi))
        print('STD of the original data - S...', np.nanstd(sal_ctd_vi))
        print('')
        print('Assuming an instrumental error with a standard deviation of (T)', std_inst_error_T)
        print('eps recomended for T OI...', (std_inst_error_T/np.nanstd(tem_ctd_vi))**2)
        print('')
        print('Assuming an instrumental error with a standard deviation of (S)...', std_inst_error_S)
        print('eps recomended for S OI...', (std_inst_error_S/np.nanstd(sal_ctd_vi))**2)
        print('')
        print('And you are using an eps of...', eps_sc)
        print('')
        
        # Check with std only in level = 100 m 
        
        lev_test = 100
        izz = np.argmin(np.abs(dep_ctd_new-lev_test))
        
        
        print('')
        print('>>>>> Using only data at dep =', lev_test)
        print('')
  
        print('STD of the original data - T...', np.nanstd(tem_ctd_vi[:,izz]))
        print('STD of the original data - S...', np.nanstd(sal_ctd_vi[:,izz]))
        print('')
        print('Assuming an instrumental error with a standard deviation of (T)', std_inst_error_T)
        print('eps recomended for T OI...', (std_inst_error_T/np.nanstd(tem_ctd_vi[:,izz]))**2)
        print('')
        print('Assuming an instrumental error with a standard deviation of (S)...', std_inst_error_S)
        print('eps recomended for S OI...', (std_inst_error_S/np.nanstd(sal_ctd_vi[:,izz]))**2)
        print('')
        print('And you are using an eps of...', eps_sc)
        print('')        


        
        
        ''' Spatio-temporal OI Temperature '''
        
        print ('')
        print ('Interpolating temperature...')
        print ('')
        

            
        lonikm2d, latikm2d, timeOI, temOI, err_tem  = \
              spatiotemporal_OI_3Dvar(dep_ctd_new, \
                 lon_ctd, lat_ctd, time_ctd, tem_ctd_vi,\
                 loni, lati, timei, 
                 lx, ly, lt, 
                 ang_sc, eps_sc, mfield_sc)
                 

        # Save interpolated Temperature
                
        print ('Saving temperature...')
        print ('')
        
        filenameT = name_conf + '_stOI_Lx' + np.str(int(lx))+ 'km_Lt' +\
                    np.str(lt) + 'days_cd_'+\
                    mdates.num2date(timei).strftime("%Y%m%d%H%M") +\
                    '_T.nc'
    
        # Create a nc file and save data into it:
        create_nc(dir_OIdata, filenameT, 'ptem', lonOI, latOI, dep_ctd_new)
        save_to_nc(dir_OIdata, filenameT, 'ptem',\
                    temOI, err_tem, \
                    lonOI, latOI, timei, dep_ctd_new)
            
            
        # lev_plot = 50
            
        # iz_orig = np.argmin(np.abs(dep_ctd_new - lev_plot))
        
        # # check OI T
        # plt.figure(figsize=(12,10))
        # pc = plt.pcolor(lonOI, latOI, temOI[iz_orig],
        #             vmin=temOI[iz_orig].min(), 
        #             vmax=temOI[iz_orig].max(), 
        #             cmap=plt.cm.jet)
        # sc = plt.scatter(lon_ctd, lat_ctd, 
        #                   c=tem_ctd_vi[:, iz_orig],
        #             vmin=temOI[iz_orig].min(), 
        #             vmax=temOI[iz_orig].max(),
        #             cmap=plt.cm.jet,
        #             linewidths=0.5, edgecolors='w', s=60)
        # plt.colorbar(pc, orientation='horizontal')
        # #plt.colorbar(sc, orientation='horizontal')
        # plt.axis('image')
        # plt.tight_layout()
            
            
        ''' OI Salinity '''
        
        print ('Interpolating salinity')
        print ('')
        
                
        lonikm2d, latikm2d, timeOI, salOI, err_sal  = \
              spatiotemporal_OI_3Dvar(dep_ctd_new, \
                 lon_ctd, lat_ctd, time_ctd, sal_ctd_vi,\
                 loni, lati, timei, 
                 lx, ly, lt, 
                 ang_sc, eps_sc, mfield_sc)        
                
        # Save interpolated Salinity
                
        print ('Saving salinity...')
        print ('')

        filenameS = name_conf + '_stOI_Lx' + np.str(int(lx))+ 'km_Lt' +\
                    np.str(lt) + 'days_cd_'+\
                    mdates.num2date(timei).strftime("%Y%m%d%H%M") +\
                    '_S.nc'
                    
    
        # Create a nc file and save data into it:
        create_nc(dir_OIdata, filenameS, 'psal', lonOI, latOI, dep_ctd_new)
        save_to_nc(dir_OIdata, filenameS, 'psal',\
                    salOI, err_sal, \
                    lonOI, latOI, timei, dep_ctd_new)
            

            
        # lev_plot = 50
            
        # iz_orig = np.argmin(np.abs(dep_ctd_new - lev_plot))
        
        # # check OI S
        # plt.figure(figsize=(12,10))
        # pc = plt.pcolor(lonOI, latOI, salOI[iz_orig],
        #             vmin=salOI[iz_orig].min(), 
        #             vmax=salOI[iz_orig].max(), 
        #             cmap=plt.cm.jet)
        # sc = plt.scatter(lon_ctd, lat_ctd, 
        #                   c=sal_ctd_vi[:, iz_orig],
        #             vmin=salOI[iz_orig].min(), 
        #             vmax=salOI[iz_orig].max(),
        #             cmap=plt.cm.jet,
        #             linewidths=0.5, edgecolors='w', s=60)
        # plt.colorbar(pc, orientation='horizontal')
        # #plt.colorbar(sc, orientation='horizontal')
        # plt.axis('image')
        # plt.tight_layout()
            


        # if num_conf != '5':       
            

            
        #     print('')
        #     print('------ noise-to-signal error analysis ------')
        
        #     print('')
        #     print('>>>>> Using all data ')
        #     print('')
        #     print('STD of the original data - U...', np.nanstd(u_adcp))
        #     print('STD of the original data - V...', np.nanstd(v_adcp))
        #     print('')
        #     print('Assuming an instrumental error with a standard deviation of 0.1 m/s')
        #     print('eps recomended for U OI...', (0.01/np.nanstd(u_adcp))**2)
        #     print('eps recomended for V OI...', (0.01/np.nanstd(v_adcp))**2)
        #     print('')
        #     print('And you are using an eps of...', eps_vc)
        #     print('')
            
        #     izz_adcp = np.argmin(np.abs(dep_adcp-lev_test))

        #     print('')
        #     print('>>>>> Using only data at dep =', lev_test)
        #     print('')
      
        #     print('STD of the original data - U...', np.nanstd(u_adcp[:,izz_adcp]))
        #     print('STD of the original data - V...', np.nanstd(v_adcp[:,izz_adcp]))
        #     print('')
        #     print('Assuming an instrumental error with a standard deviation of 0.1 m/s')
        #     print('eps recomended for U OI...', (0.01/np.nanstd(u_adcp[:,izz_adcp]))**2)
        #     print('eps recomended for V OI...', (0.01/np.nanstd(v_adcp[:,izz_adcp]))**2)
        #     print('')
        #     print('And you are using an eps of...', eps_vc)
        #     print('')            

            
    
     
        #     ''' OI U '''
        
        #     print ('Interpolating U')
        #     print ('')
        

        #     lonikm2d, latikm2d, timeOI, uOI, err_u  = \
        #           spatiotemporal_OI_3Dvar(dep_adcp, \
        #              lon_adcp, lat_adcp, time_adcp, u_adcp,\
        #              loni, lati, timei, 
        #              lx, ly, lt, 
        #              ang_vc, eps_vc, mfield_vc)      
                  
        #     # Save interpolated U
                
        #     print ('Saving U...')
        #     print ('')
  
        #     filenameU = name_conf + '_stOI_Lx' + np.str(int(lx))+ 'km_Lt' +\
        #                 np.str(lt) + 'days_cd_'+\
        #                 mdates.num2date(timei).strftime("%Y%m%d%H%M") +\
        #                 '_U.nc'
                    
        #     # Create an nc file and save data into it:
        #     create_nc(dir_OIdata, filenameU, 'u', lonOI, latOI, dep_adcp)
        #     save_to_nc(dir_OIdata, filenameU, 'u', uOI, err_u, \
        #             lonOI, latOI, timei, dep_adcp)


            
        #     ''' OI V '''
        
        #     print ('Interpolating V')
        #     print ('')
        

        #     lonikm2d, latikm2d, timeOI, vOI, err_v  = \
        #           spatiotemporal_OI_3Dvar(dep_adcp, \
        #              lon_adcp, lat_adcp, time_adcp, v_adcp,\
        #              loni, lati, timei, 
        #              lx, ly, lt, 
        #              ang_vc, eps_vc, mfield_vc)    
                      
        #     # Save interpolated V
                
        #     print ('Saving V...')
        #     print ('')

        #     filenameV = name_conf + '_stOI_Lx' + np.str(int(lx))+ 'km_Lt' +\
        #                 np.str(lt) + 'days_cd_'+\
        #                 mdates.num2date(timei).strftime("%Y%m%d%H%M") +\
        #                 '_V.nc'
                        
        #     # Create an nc file and save data into it:
        #     create_nc(dir_OIdata, filenameV, 'v', lonOI, latOI, dep_adcp)
        #     save_to_nc(dir_OIdata, filenameV, 'v', vOI, err_v, \
        #             lonOI, latOI, timei, dep_adcp)                
       
       
       
       
       
       
       