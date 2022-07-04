#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import pickle


"""
***********************************
CODE USED FOR THE LEADERBOARD D2.3
***********************************

Step 13a: 
  
    Previous step:
        Step12f: Compute statistics (RMSE-based score) 
         between reconstructed and model fields.
         
Here: make ranking of the best configurations depending on the RMSE-based score
computed in Step 12f.

        1 ranking per region and variable.
        
        Model options: CMEMS, WMOP, eNATL60
        Region options: Med, Atl

written by Bàrbara Barceló-Llull on 06-04-2022 at IMEDEA (Mallorca, Spain)

"""

def extract_RMSEs_for_region(keyregion, RMSEs_DH):    
 
        ''' Create array for all values '''

        RMSEs_all  = [] 
        model_all  = [] 
        conf_all   = [] 
           
        for keymodel in RMSEs_DH[keyregion].keys():
            
            print('Model... ', keymodel)
            
            for keyconf in RMSEs_DH[keyregion][keymodel].keys():
                
                ''' Don't include conf 4 in the ranking '''
                
                if keyconf[4:10] != 'conf_4':
                    
                    print('Conf... ', keyconf)
                 
                    RMSEs_all = np.append(RMSEs_all, 
                                      RMSEs_DH[keyregion][keymodel][keyconf])
                    model_all = np.append(model_all, keymodel)
                    conf_all  = np.append(conf_all, keyconf)
                
        return RMSEs_all, model_all, conf_all  

def print_ranking_region(RMSEs_all_DH, model_DH, conf_DH):
    
    # Make ranking by ordering the values of RMSEs 

    ind_sorted  = np.argsort(RMSEs_all_DH)[::-1]
        
    RMSEs_all_DH_order = RMSEs_all_DH[ind_sorted]
    model_DH_order     = model_DH[ind_sorted]
    conf_DH_order      = conf_DH[ind_sorted]


    
    print('----------------------------------')
    print('LEADERBOARD')
    print('(Rank, RMSEs, conf)')
    print('----------------------------------')
    print('')
    
    n = len(RMSEs_all_DH_order)
    for ii in np.arange(n):
        
        num_conf = conf_DH_order[ii][9]

    
        if num_conf == '2':
                     
            if conf_DH_order[ii][21:29] == 'res_05km':
                         num_conft = '2a (CTD 5km 1000m Sep.)'  
                         
            elif conf_DH_order[ii][21:29] == 'res_08km':
                         num_conft = '2b (CTD 8km 1000m Sep.)'
                         
            elif conf_DH_order[ii][21:29] == 'res_12km':
                         num_conft = '2c (CTD 12km 1000m Sep.)'

            elif conf_DH_order[ii][21:29] == 'res_15km':
                         num_conft = '2d (CTD 15km 1000m Sep.)'     
                
        elif  num_conf == '3':  
                     
            if conf_DH_order[ii][21:29] == 'res_02.5':
                         num_conft = '3b (uCTD 2.5km 200m Sep.)'
                
            elif conf_DH_order[ii][21:29] == 'res_06.0':
                         num_conft = '3a (uCTD 6km 500m Sep.)'   
                         
        elif  num_conf == '1':  
                     
                      num_conft = '1 (CTD 10km 500m Sep.)' 
     
        elif  num_conf == '5':  
                     
                      num_conft = '5 (Gliders 6km 500m Sep.)'  

        elif  num_conf == '4':  
                     
                      num_conft = '4 (CTD 10km 1000m Jan.)' 
                      
        elif  num_conf == 'r':  
                     
                      num_conft = 'r (CTD 10km 1000m Sep.)' 
            
        print('  ' + '{:02d}'.format(ii+1) + '      ' +
              '{:.4f}'.format(RMSEs_all_DH_order[ii]) + '      ' + 
              num_conft + ' ' + model_DH_order[ii])
        
    print('----------------------------------')
    print('')
    
if __name__ == '__main__': 
    
    plt.close('all')
    
    ''' Directories '''
    
    dir_dic   = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/comparison/spatio-temporal_OI/' 
    dir_fig   = '/Users/bbarcelo/HOME_SCIENCE/Figures/2020_EuroSea/comparison/spatio-temporal_OI/'

    ''' Files with the RMSE-based score for each field '''   

    # From Step 12f
    f_RMSEs_DH  = open(dir_dic + 'RMSEs_stOIcd_DH_same_domain_improved.pkl','rb')
    f_RMSEs_SP  = open(dir_dic + 'RMSEs_stOIcd_SP_same_domain_improved.pkl','rb')
    f_RMSEs_Ro  = open(dir_dic + 'RMSEs_stOIcd_Ro_same_domain_improved.pkl','rb')    
    
    RMSEs_DH = pickle.load(f_RMSEs_DH)
    RMSEs_SP = pickle.load(f_RMSEs_SP)
    RMSEs_Ro = pickle.load(f_RMSEs_Ro)
    
    f_RMSEs_DH.close()  
    f_RMSEs_SP.close()  
    f_RMSEs_Ro.close()  
    
    ''' 1 ranking for each region including all models '''
    
    # Which region?
    keyregion = 'Atl' #'Med' or 'Atl'
    
    # Concatenate all the data for each region 
    RMSEs_all_DH, model_DH, conf_DH = \
                     extract_RMSEs_for_region(keyregion, RMSEs_DH)
    RMSEs_all_SP, model_SP, conf_SP = \
                     extract_RMSEs_for_region(keyregion, RMSEs_SP)
    RMSEs_all_Ro, model_Ro, conf_Ro = \
                     extract_RMSEs_for_region(keyregion, RMSEs_Ro)

    # make ranking and print it
    print('')
    print('==============================')   
    print('RANKING FOR THE...', keyregion)
    print('==============================')              
    print('')
    print('//////////////////////////////')   
    print('RANKING FOR DH')
    print('//////////////////////////////') 
    print('')
    
    print_ranking_region(RMSEs_all_DH, model_DH, conf_DH)
    
    print('')
    print('//////////////////////////////') 
    print('RANKING FOR SPEED')
    print('//////////////////////////////') 
    print('')
    
    print_ranking_region(RMSEs_all_SP, model_SP, conf_SP)    
    
    print('')
    print('//////////////////////////////') 
    print('RANKING FOR Ro')
    print('//////////////////////////////') 
    print('')
    
    print_ranking_region(RMSEs_all_Ro, model_Ro, conf_Ro)        