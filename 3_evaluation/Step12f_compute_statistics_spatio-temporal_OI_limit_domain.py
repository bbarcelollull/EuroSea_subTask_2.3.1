#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
from matplotlib              import dates as mdates
import pickle
from scipy.interpolate       import griddata


"""

Step 12f: Compute statistics (RMSE-based score) 
         between reconstructed and model fields.

This codes does:

1.	Open reconstructed and model fields (saved in a bigger domain 
    than the corresponding configuration in Step10c).
2.	Interpolate model fields (ssh, ut, vt, speed, Ro) onto the 
    reconstruction grid.
3.	Limit model and reconstructed data within the configuration 2a 
    domain.
4.	Compute DH anomaly and SSH anomaly. Spatial average over 
    configuration 2a domain. 
5.	Calculate RMSE-based score between reconstructed and model fields.                  https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60/blob/master/notebooks/example_data_eval.ipynb
6.	Save RMSE-based score in a .pkl file for each field.
7.	Plot table and figure of the RMSE-based score. (Only plot figures 
    for DH and geostrophic velocity magnitude.)

        
    dir_dic   = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/comparison/spatio-temporal_OI/' 
        
    f_RMSEs_DH = open(dir_dic + 'RMSEs_stOIcd_DH_same_domain.pkl','wb')
    f_RMSEs_SP = open(dir_dic + 'RMSEs_stOIcd_SP_same_domain.pkl','wb')
    f_RMSEs_Ro = open(dir_dic + 'RMSEs_stOIcd_Ro_same_domain.pkl','wb')

     
    dir_fig   = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/comparison/spatio-temporal_OI/'


    Model options: CMEMS, WMOP, eNATL60
    Region options: Med, Atl

written by Bàrbara Barceló-Llull on 31-05-2022 at IMEDEA (Mallorca, Spain)

"""

def interp_2d_model_data(var, lonm, latm, lonr, latr):            
            
    # remove nans for the interpolation
    values = var[~np.isnan(var)] 

    lonm2d, latm2d = np.meshgrid(lonm, latm)
    points = (lonm2d[~np.isnan(var)], latm2d[~np.isnan(var)])

    var_int = griddata(points, values, (lonr, latr), 
                               method='linear')  
            
    # plt.figure()
    # plt.subplot(211)
    # plt.pcolor(lonm2d, latm2d, var, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.subplot(212)
    # plt.pcolor(lonr, latr, var_int, cmap=plt.cm.jet)
    # plt.colorbar()
            
    return var_int
        
def RMSE(var1, var2):

    RMSE = np.sqrt(np.nanmean((var1-var2)**2))
    
    return RMSE  

def RMSE_based_score(sshOI, sshNR):
    
    # RMSE based score: https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60/blob/master/src/mod_eval.py
    
    RMSEs = 1. - (np.sqrt(np.nanmean((sshOI-sshNR)**2))/np.sqrt(np.nanmean((sshNR)**2)))
    
    return RMSEs
        

def make_table(title_text, footer_text, data2table_ordered, dir_fig, save_title):
    

    data = data2table_ordered[:] 
    
    ''' 
    Code from: 
    https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4: 
    '''
    
    fig_background_color = 'skyblue'
    fig_border = 'steelblue'

    # Data should have this format
    # data =  [
    #         [         'Freeze', 'Wind', 'Flood', 'Quake', 'Hail'],
    #         [ '5 year',  66386, 174296,   75131,  577908,  32015],
    #         ['10 year',  58230, 381139,   78045,   99308, 160454],
    #         ['20 year',  89135,  80552,  152558,  497981, 603535],
    #         ['30 year',  78415,  81858,  150656,  193263,  69638],
    #         ['40 year', 139361, 331509,  343164,  781380,  52269],
    #     ]   

    # Pop the headers from the data array
    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]
    
    # Table data needs to be non-numeric text. Format the data
    # while I'm at it.
    cell_text = []
    for row in data:
        cell_text.append(['{0:.4f}'.format(x) for x in row])
        
    # Get some lists of color specs for row and column headers
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    
    # Create the figure. Setting a small pad on tight_layout
    # seems to better regulate white space. Sometimes experimenting
    # with an explicit figsize here can produce better outcome.
    plt.figure(linewidth=2,
               edgecolor=fig_border,
               facecolor=fig_background_color,
               tight_layout={'pad':1},
               #tight_layout={'rect':[0, 0.01, 1, 0.99]}, #(left, bottom, right, top)
               #figsize=(6,5)
              )
    
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          cellLoc='center',
                      rowLabels=row_headers,
                      rowColours=rcolors,
                      rowLoc='left',
                      colColours=ccolors,
                      colLabels=column_headers,
                      loc='center')
    
    # Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)
    
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Hide axes border
    plt.box(on=None)
    
    # Add title
    plt.suptitle(footer_text + '\n' + title_text )
    
    # Add footer
    #plt.figtext(0.95, 0.05, footer_text, horizontalalignment='center', size=10, weight='light')
    
    # Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    #plt.draw()
    
    # Create image. plt.savefig ignores figure edge and face colors, so map them.
    fig = plt.gcf()
    plt.savefig(dir_fig + save_title,
            #bbox='tight',
            bbox_inches='tight',
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=150
            )
        


if __name__ == '__main__': 
    
    plt.close('all')
    
    ''' Directories '''
    
    dir_dic   = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/comparison/spatio-temporal_OI/' 
    dir_fig   = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/comparison/spatio-temporal_OI/'
    dir_pobs  = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/pseudo_observations/'
    

    ''' Which region? All models in a loop. '''
    
    regions        = ['Med', 'Atl'] #['Med', 'Atl'] #'Atl' or 'Med'
        
    dicDH_region = {}
    dicSP_region = {}
    dicRo_region = {} 
    
    for region in regions:

        ''' * Coordinates of configuration 2a reconstructed fields for this region * '''
    
        # Reconstructed fields configuration 2 (5 km 1000 m) --> extract domain

        f_rec_domain    = open(dir_dic + region + '_CMEMS_rec_fields_upper_layer.pkl','rb')
        rec_data_domain = pickle.load(f_rec_domain)
        f_rec_domain.close()  
        
        key_domain = region + '_conf_2_dep_1000m_res_05km_Sep_CMEMS_stOI_Lx20km_Lt10days_cd'
        
        lon_grid_domain = rec_data_domain[key_domain]['lon']
        lat_grid_domain = rec_data_domain[key_domain]['lat']
        
        lon_domain_min, lon_domain_max = np.min(lon_grid_domain), np.max(lon_grid_domain)
        lat_domain_min, lat_domain_max = np.min(lat_grid_domain), np.max(lat_grid_domain)
        
        
        ''' Initialize figure with RMSEs for this region '''


        figdh = plt.figure(figsize =(4,4))
        ax1=plt.subplot(111)
        
        figsp = plt.figure(figsize =(4,4))
        ax2=plt.subplot(111)
        
        fsize = 7
        
        if region == 'Med':
            
            models = ['CMEMS', 'WMOP', 'eNATL60']
            
            # marker type and color for the plot
            mkr = ['ob', '^m', 'sg']
            ii  = 0
            
            
        elif region == 'Atl': 
            
            models = ['CMEMS',  'eNATL60']            
            
            # marker type and color for the plot
            mkr = ['ob', 'sg']
            ii  = 0

        dicDH_model = {}
        dicSP_model = {}
        dicRo_model = {}
             
        for model in models: 
             
             print('')
             print('region ', region)
             print('model ', model)

             ''' Start the list with data to plot in a table '''
             
             table_headers_columns = []
             table_headers_columns = ['DH', 'Ug', 'Vg', 'Speed', 'Rog']
             
             data2table = []
             data2table.append(table_headers_columns)
  
             data2plot = [] 
             data2plot.append(table_headers_columns)
             
             
             '''
             1) Open reconstructed and model fields (saved in a bigger 
             domain than the corresponding configuration in Step10c).                          
             '''
             
             ''' Open model data '''
    
             f_model    = open(dir_dic + region + '_' + model +\
                               '_ocean_truth_SSH_speed_Ro_bigger.pkl','rb')
             model_data = pickle.load(f_model)
             f_model.close()         
    
             keys = model_data.keys()
             
             ''' Open reconstructed fields '''
    
             f_rec    = open(dir_dic + region + '_' + model +\
                             '_rec_fields_upper_layer.pkl','rb')
             rec_data = pickle.load(f_rec)
             f_rec.close()          
             
             
             dicDH_confs = {}
             dicSP_confs = {}
             dicRo_confs = {}
             
             for iconf, key in enumerate(keys):
                 
                 ''' For each configuration extract text for the title '''
                 
                 num_conf = key[9]
                
                 if num_conf == '2':
                     
                     if key[21:29] == 'res_05km':
                         num_conft = '2a (5km 1000m)'  
                         num_confp = r'$\bf{2a}$' + '\nCTD\n5km\n1000m\nSep.' 
                         
                     elif key[21:29] == 'res_08km':
                         num_conft = '2b (8km 1000m)'
                         num_confp = r'$\bf{2b}$' + '\nCTD\n8km\n1000m\nSep.'  

                     elif key[21:29] == 'res_12km':
                         num_conft = '2c (12km 1000m)'
                         num_confp = r'$\bf{2c}$' + '\nCTD\n12km\n1000m\nSep.'  

                     elif key[21:29] == 'res_15km':
                         num_conft = '2d (15km 1000m)'     
                         num_confp = r'$\bf{2d}$' + '\nCTD\n15km\n1000m\nSep.'  
                
                 elif  num_conf == '3':  
                     
                     if key[21:29] == 'res_02.5':
                         num_conft = '3b (2.5km 200m)'
                         num_confp = r'$\bf{3b}$' + '\nuCTD\n2.5km\n200m\nSep.'  
                
                     elif key[21:29] == 'res_06.0':
                         num_conft = '3a (6km 500m)'   
                         num_confp = r'$\bf{3a}$' + '\nuCTD\n6km\n500m\nSep.'  
                         
                 elif  num_conf == '1':  
                     
                      num_conft = '1 (10km 500m)' 
                         
                      num_confp = r'$\bf{1}$' + '\nCTD\n10km\n500m\nSep.'   

                      
                 elif  num_conf == '5':  
                     
                      num_conft = '5 (6km 500m)'  
                      num_confp = r'$\bf{5}$' + '\nGliders\n6km\n500m\nSep.'  

                 elif  num_conf == '4':  
                     
                      num_conft = '4 (10km 1000m)' 
                      num_confp = r'$\bf{4}$' + '\nCTD\n10km\n1000m\nJan.' 
                      
                 elif  num_conf == 'r':  
                     
                      num_conft = 'r (10km 1000m)' 
                      num_confp = r'$\bf{r}$' + '\nCTD\n10km\n1000m\nSep.'  
                        
                 print('')        
                 print(key)
                 print('num conf...', num_conft)
                 
                 ''' Reconstructed fields for this configuration '''

                 lonr       = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['lon']
                 latr       = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['lat']
                 dhr_all    = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['dh']
                 ugr_all    = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['ug']
                 vgr_all    = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['vg']
                 spgr_all   = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['speedg']
                 Rogr_all   = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['Rog']
                 dep_OImap  = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['dep_map'] 
                 time_OImap = rec_data[key + '_stOI_Lx20km_Lt10days_cd']['time_map'] 
      
                 print('')
                 print('Date of the OI map...')
                 print(mdates.num2date(time_OImap).strftime("%Y-%m-%d %H:%M"))
                 print('')
                 
                 
                 ''' Model data for the OI date of this configuration '''
  
                 lonm       = model_data[key]['lon']
                 latm       = model_data[key]['lat']
                 sshm       = model_data[key]['ssh_int']
                 utm        = model_data[key]['u_int']
                 vtm        = model_data[key]['v_int']
                 sptm       = model_data[key]['speed_int'] # total velocity
                 Rotm       = model_data[key]['Ro_int']    # total Ro 
                 time_mmap  = model_data[key]['time_map'][0]
                 
                 print('')
                 print('Date of the model map...')
                 print(mdates.num2date(time_mmap).strftime("%Y-%m-%d %H:%M"))
                 print('')

                 '''
                 2) Interpolate model fields onto reconstruction grid 
                 (linear, griddata).                          
                 '''
                 #sshai_all = interp_2d_model_data(ssha, lonm, latm, lonr, latr)
                 sshmi_all = interp_2d_model_data(sshm, lonm, latm, lonr, latr)
                 utmi_all  = interp_2d_model_data(utm,  lonm, latm, lonr, latr)
                 vtmi_all  = interp_2d_model_data(vtm,  lonm, latm, lonr, latr)
                 sptmi_all = interp_2d_model_data(sptm, lonm, latm, lonr, latr)
                 Rotmi_all = interp_2d_model_data(Rotm, lonm, latm, lonr, latr)             

                 # check interpolation
                 # plt.figure(figsize=(12,4))
                 # plt.subplot(121)
                 # plt.pcolor(lonm, latm, sshm, cmap=plt.cm.jet)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' ' + num_conf + ' ' + 'original model SSH')
                 # plt.axis('image')

                 # plt.subplot(122)
                 # plt.pcolor(lonr, latr, sshmi_all, cmap=plt.cm.jet)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' ' + num_conf + ' ' +  'interpolated model SSH')
                 # plt.axis('image')
                 # plt.tight_layout()
                 
                 # plt.figure(figsize=(12,4))
                 # plt.subplot(121)
                 # plt.pcolor(lonm, latm, sptm, cmap=plt.cm.jet)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' ' + num_conf + ' ' +  'original model speed')
                 # plt.axis('image')
                 
                 # plt.subplot(122)
                 # plt.pcolor(lonr, latr, sptmi_all, cmap=plt.cm.jet)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' ' + num_conf + ' ' +  'interpolated model speed')
                 # plt.axis('image') 
                 # plt.tight_layout()
                 
                 '''
                 3) Limit model and reconstructed data within the configuration 2a domain.
                 '''

                 cond_lond = np.logical_and(lonr[0,:] >= lon_domain_min,
                                            lonr[0,:] <= lon_domain_max)
    
                 cond_latd = np.logical_and(latr[:,0] >= lat_domain_min,
                                            latr[:,0] <= lat_domain_max)   
                 
                 #limit region
                 lond = lonr[cond_latd,:][:,cond_lond]
                 latd = latr[cond_latd,:][:,cond_lond]

                 # limit reconstructed data
                 dhr  = dhr_all[cond_latd,:][:,cond_lond] 
                 ugr  = ugr_all[cond_latd,:][:,cond_lond] 
                 vgr  = vgr_all[cond_latd,:][:,cond_lond] 
                 spgr = spgr_all[cond_latd,:][:,cond_lond] 
                 Rogr = Rogr_all[cond_latd,:][:,cond_lond]              
                 
                 # limit model data
                 sshmi = sshmi_all[cond_latd,:][:,cond_lond] 
                 utmi  = utmi_all[cond_latd,:][:,cond_lond] 
                 vtmi  = vtmi_all[cond_latd,:][:,cond_lond] 
                 sptmi = sptmi_all[cond_latd,:][:,cond_lond] 
                 Rotmi = Rotmi_all[cond_latd,:][:,cond_lond]        

                 
                 '''
                 4) Compute DH anomaly and SSH anomaly. 
                 Spatial average over configuration 2a domain. 
                 '''
                         
                 dha   = dhr - np.nanmean(dhr)
                 sshai = sshmi - np.nanmean(sshmi)


                 ''' Plot fields compared and save figure '''
                 
                 # figc = plt.figure(figsize=(12,8))
                 # plt.subplot(221)
                 # plt.contourf(lond, latd, dha, 20, cmap=cmo.cm.balance)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' - conf ' + num_conf + ' - rec. DHa [dyn m]')
                 # plt.xlim([lon_domain_min, lon_domain_max])
                 # plt.ylim([lat_domain_min, lat_domain_max])  
                 
                 # plt.subplot(222)
                 # plt.contourf(lond, latd, sshai, 20, cmap=cmo.cm.balance)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' - conf ' + num_conf + ' - ocean truth SSHa [m]')
                 # plt.xlim([lon_domain_min, lon_domain_max])
                 # plt.ylim([lat_domain_min, lat_domain_max])
                 
                 # plt.subplot(223)
                 # plt.contourf(lond, latd, spgr, 20, cmap=plt.cm.Spectral_r)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' - conf ' + num_conf + ' - rec. geostr. speed [m/s]')
                 # plt.xlim([lon_domain_min, lon_domain_max])
                 # plt.ylim([lat_domain_min, lat_domain_max])
                 
                 # plt.subplot(224)
                 # plt.contourf(lond, latd, sptmi, 20, cmap=plt.cm.Spectral_r)
                 # plt.colorbar()
                 # plt.title(region + ' ' + model + ' - conf ' + num_conf + ' - ocean truth horiz. speed [m/s]')
                 # plt.xlim([lon_domain_min, lon_domain_max])
                 # plt.ylim([lat_domain_min, lat_domain_max])   
                 
                 # plt.tight_layout()
                 # figc.savefig(dir_fig + '/Figures_compared_fields/Fields_' + key + '.png', dpi=500 )
                 
                 # plt.close('all')
                 
                 '''
                 5) Calculate RMSE-based score between reconstructed 
                 and model fields. 
                 
                 https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60/blob/master/notebooks/example_data_eval.ipynb

                 '''
                    
                 dh_rmse_mo  = RMSE_based_score(dha,  sshai)
                 ug_rmse_mo  = RMSE_based_score(ugr,  utmi)
                 vg_rmse_mo  = RMSE_based_score(vgr,  vtmi)
                 spg_rmse_mo = RMSE_based_score(spgr, sptmi)
                 Rog_rmse_mo = RMSE_based_score(Rogr, Rotmi)
                 
                 ''' Save RMSEs in a dictionary to save it into a file '''
                 
                 dicDH_confs.update({key:  dh_rmse_mo})
                 dicSP_confs.update({key:  spg_rmse_mo})
                 dicRo_confs.update({key:  Rog_rmse_mo})
             
                 ''' Save results to plot in a table '''
                 
                 rmse2table = []
                 rmse2table = ['conf ' +num_conft, dh_rmse_mo, 
                               ug_rmse_mo, vg_rmse_mo,
                               spg_rmse_mo, Rog_rmse_mo]

                 rmse2plot = []
                 rmse2plot = [num_confp, dh_rmse_mo, 
                                ug_rmse_mo, vg_rmse_mo,
                                spg_rmse_mo, Rog_rmse_mo]                 
                 
        
                 data2table.append(rmse2table)
                 data2plot.append(rmse2plot)
                 
                 # print(data2table)
                 
             ''' Save dicXX_confs dictionary into dicXX_model dictionary '''
             
             dicDH_model.update({model:  dicDH_confs})
             dicSP_model.update({model:  dicSP_confs})
             dicRo_model.update({model:  dicRo_confs})
              
             ''' Plot 1 table for each region and model '''
    
             # Change order table rows so that 'conf r is the first one
             # and 'conf 4' the last one

             data2table_ordered = [data2table[0]] + [data2table[-1]] + data2table[1:-3] + \
                         [data2table[-2]] + [data2table[-3]]
             
             # note: data2table_ordered will be modified after making the table
             make_table('RMSE between model data and reconstructed fields, same domain', 
                        region + '  ' + model,
                        data2table_ordered, 
                        dir_fig, 
                        region + '_' + model +\
                        '_spatio-temporal_OI_cd_RMSEs_table_same_domain.png')


             ''' Plot figure with the RMSEs for all conf, variables and models '''
             
             data2plot_or = [data2plot[0]] + [data2plot[-1]] + data2plot[1:-5] + \
                         [data2plot[-4]] + [data2plot[-5]] + \
                         [data2plot[-2]] + [data2plot[-3]]
             
             # plot DH RMSE for each configuration (x-axis)
             
             x_axis_conf = []
             dh_conf     = []
             ug_conf     = []
             vg_conf     = []
             spg_conf    = []
             Rog_conf    = []
             
             rmse_vars = data2plot_or[0].copy()
             
             for row in data2plot_or[1:]:
                 #print(row)
                 
                 x_axis_conf.append(row[0])
                 dh_conf.append(row[1])
                 ug_conf.append(row[2])
                 vg_conf.append(row[3])
                 spg_conf.append(row[4])
                 Rog_conf.append(row[5])
                 
            
             # plot data in figure
             xx = np.arange(len(x_axis_conf))
             
             ax1.plot(xx, dh_conf,  mkr[ii], alpha = 0.7, markersize=fsize-2, label= model)
             ax2.plot(xx, spg_conf, mkr[ii], alpha = 0.7, markersize=fsize-2, label= model)
             # ax3.plot(xx, Rog_conf, mkr[ii], alpha = 0.7, label= model)
             
             ii = ii + 1

  
             
        ax1.set_xticks(np.arange(len(x_axis_conf)))
        #ax1.set_xticklabels(x_axis_conf,rotation=80, fontsize=fsize-1)
        ax1.set_xticklabels(x_axis_conf, fontsize=fsize-1)
        
        ax1.legend(loc='lower left' ,fontsize=fsize-1)
        ax1.set_ylabel('RMSEs', fontsize=fsize)
        ax1.tick_params(axis='y', labelsize=fsize-1)       
        ax1.set_title(r'DHa$_{rec}$ vs. SSHa$_{truth}$',
                      fontsize=fsize+1)             
        
        ax2.set_xticks(np.arange(len(x_axis_conf)))
        ax2.set_xticklabels(x_axis_conf, fontsize=fsize-1)
        ax2.legend(loc='lower left' ,fontsize=fsize-1)
        ax2.set_ylabel('RMSEs', fontsize=fsize)
        ax2.tick_params(axis='y', labelsize=fsize-1)       
        ax2.set_title(r'u$^{g}_{rec}$ vs. u$^{t}_{truth}$',
                      fontsize=fsize+1)  

        # ax3.set_xticks(np.arange(len(x_axis_conf)))
        # ax3.set_xticklabels(x_axis_conf,rotation=80, fontsize=fsize-1)
        # ax3.legend(loc='lower left' , fontsize=fsize-1)
        # ax3.set_ylabel('RMSEs', fontsize=fsize)
        # ax3.tick_params(axis='y', labelsize=fsize-1)       
        # ax3.set_title(r'Ro$^{g}_{rec}$ vs. Ro$^{t}_{truth}$',
        #               fontsize=fsize+2)   
        ax1.set_ylim([0,1])
        ax2.set_ylim([0,1])
        # ax3.set_ylim([0,1])   
        #fig.suptitle(' [' + region + ', same domain]', x=0.08, y=0.03)
        # fig.tight_layout()
        # fig.savefig(dir_fig + region + '_all_spatio-temporal_OI_cd_RMSEs_fig_same_domain.png', dpi=500 ) 
        figdh.tight_layout()
        figdh.savefig(dir_fig + region + '_all_spatio-temporal_OI_cd_RMSEs_fig_same_domain_DH.png', dpi=500 ) 
        figsp.tight_layout()
        figsp.savefig(dir_fig + region + '_all_spatio-temporal_OI_cd_RMSEs_fig_same_domain_SP.png', dpi=500 ) 
                
        
        
        
        ''' Save dicXX_model dictionary into dicXX_region dictionary '''
 
        dicDH_region.update({region:  dicDH_model})
        dicSP_region.update({region:  dicSP_model})
        dicRo_region.update({region:  dicRo_model})
        
    ''' Save each dictionary into a file '''

    f_RMSEs_DH = open(dir_dic + 'RMSEs_stOIcd_DH_same_domain_improved.pkl','wb')
    f_RMSEs_SP = open(dir_dic + 'RMSEs_stOIcd_SP_same_domain_improved.pkl','wb')
    f_RMSEs_Ro = open(dir_dic + 'RMSEs_stOIcd_Ro_same_domain_improved.pkl','wb')
   
    pickle.dump(dicDH_region, f_RMSEs_DH)
    pickle.dump(dicSP_region, f_RMSEs_SP)
    pickle.dump(dicRo_region, f_RMSEs_Ro)
    
    f_RMSEs_DH.close()  
    f_RMSEs_SP.close()  
    f_RMSEs_Ro.close()  

    