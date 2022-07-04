#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import matplotlib.pyplot     as plt
import glob
import netCDF4               as netcdf
import EuroSea_toolbox       as to
import gsw
import deriv_tools           as dt 
import time 

"""
Step 7b: Calculate dh and vgeo from T and S objectively 
interpolated in Step 5b. 

Also calculate other derived variables from density: Rog

For all configurations, in the Atlantic and Mediterranean,
for all models. 

written by Bàrbara Barceló-Llull on 04-04-2022 at IMEDEA (Mallorca, Spain)

"""

def converter_deg2m(lat):   
    ''' 
    Function to convert lon and lat in degrees to meters
    also infers gravity and coriolis frequency.
    '''
        
    ''' Parameters '''
    # factor pi , earth radio, and mean latitude
    dpi         = 4*np.arctan(1)
    factorpi    = dpi/180
    radio       = 6378137 # meters 6378.137km
    glatitude   = lat.mean() # deg
    print ('at latitude...', glatitude)
    # Coriolis parameter at mean latitude
    plat        = glatitude * factorpi # mean latitude in rad
    omega_earth = 7.2921159e-5         # Earth's angular velocity
    coriol_mean = 2 * omega_earth * np.sin(plat)
    factorlat   = np.cos(plat)
    print ('Mean Coriolis', coriol_mean)
    # Computes gravity at mean latitude
    X1          = np.sin(plat)
    X           = X1 * X1
    gravedad    = 9.780318e0 * (1+(5.2788e-3 + 2.36e-5*X)*X)
        
    #Latitude:  1 deg = 110.54 km
    #Longitude: 1 deg = 111.320*cos(latitude in rad) km
    
    convert_lon2m = radio*factorlat*factorpi 
    convert_lat2m = radio          *factorpi
        
    return convert_lon2m, convert_lat2m, gravedad, coriol_mean

def compute_CT_SA_sigma(p, lat, lon, psal, ptem):
    
    ''' Compute conservative temperature, absolute salinity and
    potential density anomaly (sigma-theta) '''
    
    p_3d = np.repeat(np.repeat(p[:,np.newaxis,np.newaxis], psal.shape[1], axis=1), 
                     psal.shape[2], axis=2)
    
    # Absolute Salinity from Practical Salinity [ g/kg ]
    SA = gsw.SA_from_SP(psal,p_3d,lon.mean(),lat.mean())
        
    # Conservative Temperature from potential temperature
    CT = gsw.CT_from_pt(SA, ptem)    

    # Calculate potential density anomaly (sigma-theta) 
        
    # potential density anomaly, being potential density
    # minus 1000 kg/m^3 (75-term equation)
    sig = gsw.sigma0(SA, CT)    # variable that we want!  
    
    return SA, CT, sig    

def compute_dh(p, SA, CT):
    
    '''   Compute dynamic height with TEOS10 and with my own code  
    
    Atention: 
            TEOS-10 function: Calculates dynamic height anomaly as the integral 
            of specific volume anomaly from the pressure p of the "bottle" to 
            the reference pressure p_ref.
            
            While with my code I use specific volume, not the anomaly!
            
    '''
    
    dp_inv = np.abs(np.diff(p[::-1]))
    
    specvol = np.zeros(SA.shape)
    geo_strf_dyn_height = np.zeros(SA.shape)
    dh_hand = np.zeros(SA.shape)
    dh_teos = np.zeros(SA.shape)
    
    for jj in np.arange(SA.shape[1]):
        for ii in np.arange(SA.shape[2]):
            # From https://glossary.ametsoc.org/wiki/Dynamic_meter
            # The standard unit of dynamic height, defined as 10 m2 s−2: 1dyn m = 10m2 s−2
            
            # Compute dh with TEOS-10 function [ m^2/s^2 ]
            geo_strf_dyn_height[:,jj,ii] = gsw.geo_strf_dyn_height(SA[:,jj,ii],CT[:,jj,ii],p,p[-1])
            
            dh_teos[:,jj,ii]  = geo_strf_dyn_height[:,jj,ii]/10 # in [dyn m]
            
            
            
            # Compute specific volume = 1 /rho
            specvol[:,jj,ii] = gsw.specvol(SA[:,jj,ii], CT[:,jj,ii], p)
 
            # Compute dh to compare with TEOS-10, from the bottom to the surface (invert)   
            specvol_inv = specvol[:,jj,ii][::-1]
            dh_pf_inv   = np.zeros((len(p)))
           
            for ip in np.arange(1,len(p)):
               
                dp_Pa = dp_inv[ip-1] * 10000 # in [Pa = N/m^2] 1dbar = 10000 Pa
                dh_pf_inv[ip] = dh_pf_inv[ip-1] + specvol_inv[ip] * dp_Pa
              
            dh_hand[:,jj,ii] = dh_pf_inv[::-1]/10 # from m^2/s^2 or J/kg to dyn m (1dyn m = 10 J/kg)
              
    return dh_hand, dh_teos

def calculate_vgeo_from_DH(dh_hand, deltax, deltay, grav, coriol_mean):
        
        grad2d_DH = dt.grad2d_3d(dh_hand, deltax, deltay) #(dDH/dx, dDH/dy)
        
        vgeo = (grav/coriol_mean) * grad2d_DH.real
        ugeo = - (grav/coriol_mean) * grad2d_DH.imag
        
        speed = np.sqrt(vgeo**2 + ugeo**2)
        
        return ugeo, vgeo, speed

def compute_buoyancy(gravedad, sig, sigma_mean):
    
    # ===========================================
    #        1) Buoyancy, b = g*rho/rho0
    # ===========================================
    
    b = gravedad*sig/(1000+sigma_mean)
    
    # from the bottom to the surface 
    binv   = b[::-1,:,:]
    depinv = dep[::-1]
    
    return b, binv, depinv
    
    
def compute_N2(binv, deltaz): 
    
    ''' Compute Brunt-Vaisala frequency '''
    
    # ===========================================
    #      2) Compute Brunt-Vaisala from b
    #    brnv2 = -g/rho0 * drho/dz = - db/dz 
    # ===========================================
    
    brnv2inv      = - dt.deriv1d_3d(binv, 'K', deltaz)
    brnv2_mean    = brnv2inv.mean()
    
    # Where N^2 < 0 --> N^2 = 0 (Romain also follows this criterion)
    brnv2inv[np.where(brnv2inv<0)] = 0
    
    brnvinv = np.sqrt(brnv2inv)
    
    return brnv2inv, brnv2_mean, brnvinv


def compute_vgeo(binv, deltax, deltay, deltaz, coriol_mean): 
    
    # ===========================================            
    #       3) Horizontal gradient of b
    #           gradb=(db/dx, db/dy)  
    #            UNITS: [s^-1]   
    # ===========================================
  
    gradsinv = dt.grad2d_3d(binv, deltax, deltay) #(db/dx, db/dy)
    gradbinv = gradsinv/coriol_mean # b=g*sigma/(rho0*f) 
    
    
    #==============================================
    # 4) Computing Geostrophic velocity from gradb 
    #==============================================
    #
    # -kXd(vgeos)/dz=gradb
    #   UNITS=[m s^-1]

    vgeosinv         = dt.vgeos_from_gradb(gradbinv,deltaz)
    uginv            = vgeosinv.real
    vginv            = vgeosinv.imag
    modvgeosinv      = np.sqrt(uginv**2+vginv**2);
    
    print('Speed min...', modvgeosinv.min())
    print('Speed max...', modvgeosinv.max())
    
    return gradsinv, gradbinv, uginv, vginv, modvgeosinv

def compute_Rog(uginv, vginv, deltax, deltay):    
        
    # Horizontal gradient of the geostrophic velocity
    graduginv   = dt.grad2d_3d(uginv,deltax,deltay)
    gradvginv   = dt.grad2d_3d(vginv,deltax,deltay)
    
    # Geostrophic vertical relative vorticity
    vvortinv = gradvginv.real - graduginv.imag
    #vvort    = vvortinv # Redefine to not change anything in the iterative method
    
    # Gesotrophic Rossby number
    Roginv   = vvortinv /coriol_mean
    
    print('Rog min...', Roginv.min())
    print('Rog max...', Roginv.max())
    
    return Roginv
        
def compute_wQG(uginv, vginv, gradsinv, deltax, deltay): 
    
    # ===========================================            
    # 13) quasi-geostrophic Q vector --> QG  
    # ===========================================  
    vgeos = uginv + 1j * vginv
    grads = gradsinv
   
    QG = dt.Q_from_v_gradb(vgeos,grads,deltax,deltay)
   
    # ===========================================            
    # 16) divergence of QG-vector 
    # ===========================================      
    divQG  = dt.divQ_from_Q(QG,deltax,deltay)   
    
    # ===============================================         
    # 18) Computes the RHS of the QG omega equation
    # ===============================================

    rhsQG = 2*divQG   
   
    # Which forcing
    rhs = rhsQG
   
   
    ''' Parameters for the relaxation method '''
    # NOTE: this parameter must be selected 
    #       conviniently to ensure convergence
    #       the relaxation method. Use this default 
    #       value. If solution does not converge (see displays in 
    #       screen) then the value must be reduced.
    delta_tau = 1e8
  
    # New values of retol and itermx to better estimate w (06.05.16)
    retol     = 1e-4#1e-2    # Maximum error between iterations
    itermx    = 200000#150000 2018-06-05 more interations for PRE-SWOT#100000  # Maximum number of iterations

    ''' Relaxation methor to infer w-QG '''
    # ===============================================    
    # Relaxation method: 
    # =============================================== 
  
    #  Continue iterations if error > retol 
  
    # starting error value
    error     = 2*retol 
    # set the counter to zero
    iteration = 0
  
    initime = time.time()
  
    while np.logical_and(error>retol, iteration<=itermx): #itermx):
   
      # Initial condition is an array of zero vertical velocity
      if iteration==0:
          ww = np.zeros(rhs.shape)
          b  = np.zeros(rhs.shape)
          iteration = 1
      else:
          ww = b
          iteration = iteration + 1
          print ('iteration ', iteration)
          
      # Boundary conditions  
      #  
      # NOTE: Here boundary conditions are zero. The first 2-3 grid points 
      # will be influenced by this boundary condition. Interior solution is
      # fine. To reduce the sensitivity of the solution to the lateral 
      # boundary conditions, the domain canbe enlarged and the boundaries moved 
      # far away from the study region 
      
      ww[:, :, 0]   = 0
      ww[:, :, -1]  = 0
      ww[:, 0, :]   = 0
      ww[:, -1, :]  = 0      
      ww[0, :, :]   = 0
      ww[-1, :, :]  = 0
      
      #  computes new w called b(IZ,IY,IX) (next iteration)
      #  we need a and c:
      #       
      #     a = N2*(w_xx+w_yy) using ww
      #     c=  coriol_mean*vvort*w_zz
      #     b=  (rhs-a-c)*delta_tau + w 
      
      # Compute the laplacian of ww
      lapww = dt.lap2d_3d(ww, deltax, deltay)
      
#      print ('   ')
#      print (' min lapwwx= ', lapww.real.min())
#      print (' max lapwwx= ', lapww.real.max())
#      print ('   ')
#      print (' min lapwwy= ', lapww.imag.min())
#      print (' max lapwwy= ', lapww.imag.max())  
      
      # Computes a = N2*(w_xx+w_yy) using ww
      a = brnv2_mean * (lapww.real + lapww.imag)     
      
      # Compute the vertical component of the laplacian of ww
      lapzz = dt.lapzz_3d(ww, deltaz)
      
#      print ('   ')
#      print (' min lapwwz= ', lapzz.min())
#      print (' max lapwwz= ', lapzz.max())      
      
      
      # Computes c=coriol_mean*vvort*w_zz using ww
      #c = coriol_mean * vvort * lapzz # For the generalized omega eq.
      c = coriol_mean * coriol_mean * lapzz # For the QG omega eq. 2018-06-05
      
      #    computes b=  (rhs-a-c)*delta_tau+ ww 
      #    (rhs - N2 (ww(N)_xx+ww(N)_yy)-f(zeta+f) ww(N)_zz)*delta_tau +
      #     ww(N)=ww(N+1)
      s  = -rhs + a + c
      b  = ww + delta_tau*s      
      
      anb = a[1:-1, 1:-1, 1:-1]
      bnb = b[1:-1, 1:-1, 1:-1]
      cnb = c[1:-1, 1:-1, 1:-1]
      snb = s[1:-1, 1:-1, 1:-1]
      
      
      aandc = anb+cnb
      
      print ('  ')
      print (' iteration = ', iteration)
#      print (' min rhs = ', rhs[1:-1, 1:-1, 1:-1].min())
#      print (' max rhs = ', rhs[1:-1, 1:-1, 1:-1].max()) 
#      print (' rhs rang = ', rhs[1:-1, 1:-1, 1:-1].max() - rhs[1:-1, 1:-1, 1:-1].min())
#      print (' a+c rang = ', aandc.max() - aandc.min())    
#      print (' s rang = ', snb.max() - snb.min())
#      print (' min ww = ', ww.min())
#      print (' max ww = ', ww.max())
#      print (' min a = ', anb.min())
#      print (' max a = ', anb.max())
#      print (' min c = ', cnb.min())
#      print (' max c = ', cnb.max())      
#      print (' min s = ', snb.min())
#      print (' max s = ', snb.max())            
      
      # maximum difference: betwen a, c, rhs?
      rmag = np.max([np.abs(anb.max()-anb.min()), 
                     np.abs(cnb.max()-cnb.min()),
                     np.abs(rhs.max()-rhs.min())])
                     
      error = np.abs(snb.max()- snb.min())/rmag
      
#      print (' ')
#      print (' rmag = ', rmag)
#      print (' smax-smin = ', snb.max()- snb.min())
      print ('  ')
      print (' error = ', error)
      
      diff = np.max(b-ww)  
      print ('  ')
      print (' diff w (ww[N+1]-ww[N]) max =', diff ) 
      
  
    # Iteration time
    fintime = time. time()
    total_time = fintime - initime
    print (' ')
    print (' total time relaxing method ', total_time)
      
    ww = b         

    return ww, rhs

def createnc_derivedvar(savedir,nx,ny,nz):
    
    # Cream el fitxer netcdf   
    nc = netcdf.Dataset(savedir, 'w', format='NETCDF3_CLASSIC')
    
    # Create the dimensions...
    nc.createDimension('dimlon',nx)    
    nc.createDimension('dimlat',ny) 
    nc.createDimension('dimdep',nz) 
    nc.createDimension('dtime', 1)
        
    # Create the variables...
    nc.createVariable('longitude','f4', ('dimlat', 'dimlon'))
    nc.createVariable('latitude','f4', ('dimlat', 'dimlon'))
    nc.createVariable('depth', 'f4',('dimdep'))
    nc.createVariable('time',      'f4', ('dtime'))
    nc.createVariable('sig', 'f4', ('dimdep', 'dimlat', 'dimlon')) 
    nc.createVariable('dh', 'f4', ('dimdep', 'dimlat', 'dimlon')) 
    nc.createVariable('ug', 'f4', ('dimdep', 'dimlat', 'dimlon'))   
    nc.createVariable('vg', 'f4', ('dimdep', 'dimlat', 'dimlon')) 
    nc.createVariable('Rog', 'f4', ('dimdep', 'dimlat', 'dimlon')) 
    nc.createVariable('N', 'f4', ('dimdep', 'dimlat', 'dimlon')) 
    
    # Write in variable attributes...
    nc.variables['longitude'].long_name   = 'longitude after O.I.'
    nc.variables['longitude'].units       = 'degrees'
    nc.variables['latitude'].long_name    = 'latitude after O.I.'
    nc.variables['latitude'].units        = 'degrees'
    nc.variables['depth'].long_name       = 'Depth from the surface to the bottom'
    nc.variables['depth'].units           = 'meters' 
    nc.variables['time'].long_name        = 'Time of the map'
    nc.variables['time'].units            = 'Number of days since 0001-01-01 00:00:00 UTC, plus one.' 
        
    nc.variables['sig'].long_name     = 'Potential density anomaly, sigma-theta'
    nc.variables['sig'].units         = 'kg/m^3' 
    nc.variables['dh'].long_name      = 'Dynamic height'
    nc.variables['dh'].units          = 'dynamic meter'     
    nc.variables['ug'].long_name      = 'U-component geostrophic velocity'
    nc.variables['ug'].units          = 'm/s' 
    nc.variables['vg'].long_name      = 'V-component geostrophic velocity'
    nc.variables['vg'].units          = 'm/s'     
    nc.variables['Rog'].long_name     = 'geostrophic Rossby number'
    nc.variables['Rog'].units         = ''  
    nc.variables['N'].long_name       = 'Brunt-Vaisala frequency, N'
    nc.variables['N'].units           = '1/s'  
     
    nc.close()  
            
if __name__ == '__main__':        
        
    plt.close('all')
    
    ''' Directories '''
    
    dir_OIdata    = '/Users/bbarcelo/HOME_SCIENCE/Data/2020_EuroSea/reconstructed_fields/spatio-temporal_OI_all_conf/'


    ''' Which model and region? '''
    
    model         = 'CMEMS' # 'CMEMS', 'WMOP', 'eNATL60'
    region        = 'Atl' #'Atl' or 'Med'
    
    '''
    >>>>>> OPEN interpolated fields <<<<<<
    '''
    oi_files  = sorted(glob.glob(dir_OIdata + region + '*_'+model + '*T.nc'))
    
    
    for file in oi_files: #[oi_files[-6]]: #oi_files: 
        

        name_conf = file[96:-5]
        num_conf  = name_conf[:33] 
        
        print('')
        print('--------------------------------------')
        print('')
        print('Configuration file...', name_conf)
        print('')
        
        print('configuration...', num_conf)
        
        
        ''' Read .nc file with interpolated fields '''
           
        ncT      = netcdf.Dataset(dir_OIdata + name_conf + '_T.nc', 'r')
        ptem     = ncT.variables['ptem'][:].data #'ptem_mk'][:].data 
        eptem    = ncT.variables['error'][:].data     
        lon      = ncT.variables['longitude'][:].data  
        lat      = ncT.variables['latitude'][:].data    
        dep      = ncT.variables['depth'][:].data  
        time_map = ncT.variables['time'][:].data 
        ncT.close() 
        
        ncS   = netcdf.Dataset(dir_OIdata + name_conf + '_S.nc', 'r')
        psal  = ncS.variables['psal'][:].data #'psal_mk'][:].data  
        epsal = ncS.variables['error'][:].data           
        ncS.close() 
        
        
        ''' Parameters we need '''
        
        p           = gsw.p_from_z(-dep,lat.mean()) # in dbar
        grav        = np.mean(gsw.grav(lat.mean(), p[::-1]))
        coriol_mean = gsw.f(lat.mean()) 
        lat_ref = 40
        
        length_deg_lonp, length_deg_latp = to.length_lon_lat_degs(lat_ref)    


        deltax     = (lon[0,1]-lon[0,0])*length_deg_lonp # meters
        deltay     = (lat[1,0]-lat[0,0])*length_deg_latp # meters
        deltaz     = dep[1]-dep[0]  
        

        ''' Compute CT, SA, and sigma '''
        
        SA, CT, sig = compute_CT_SA_sigma(p, lat, lon, psal, ptem)
        sigma_mean = np.nanmean(sig)
        
        

        # lev = 990
        # izlev = np.argmin(np.abs(dep-lev))

        # plt.figure()
        # plt.pcolor(lon, lat, ptem[izlev], cmap=plt.cm.jet,
        #            vmin=np.nanmin(CT[izlev]), 
        #            vmax=np.nanmax(ptem[izlev]))
        # plt.title('Potential temperature')
        # plt.colorbar()
        
        # plt.figure()
        # plt.pcolor(lon, lat, CT[izlev], cmap=plt.cm.jet,
        #            vmin=np.nanmin(CT[izlev]), 
        #            vmax=np.nanmax(ptem[izlev]))
        # plt.title('Conservative temperature')
        # plt.colorbar()

        # plt.figure()
        # plt.pcolor(lon, lat, psal[izlev], cmap=plt.cm.jet,
        #            vmin=np.nanmin(psal[izlev]), 
        #            vmax=np.nanmax(psal[izlev]))
        # plt.title('Practical salinity')
        # plt.colorbar()
        
        # plt.figure()
        # plt.pcolor(lon, lat, SA[izlev], cmap=plt.cm.jet,
        #            vmin=np.nanmin(SA[izlev]), 
        #            vmax=np.nanmax(SA[izlev]))
        # plt.title('Absolute salinity')
        # plt.colorbar()
        
        # plt.figure()
        # plt.pcolor(lon, lat, sig[izlev], cmap=plt.cm.jet)
        # plt.title('Potential density anomaly (sigma)')
        # plt.colorbar()        
        
        
        ''' 
        Compute DH
        '''
        
        dh_hand, dh_teos = compute_dh(p, SA, CT)
            
        
        # Note, dh_hand and dh_teos are different in absolute values
        # because the TEOS-10 function uses the specific volume anomaly
        

        # plt.figure()
        # plt.subplot(211)
        # plt.pcolor(dh_hand[0], cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.subplot(212)
        # plt.pcolor(dh_teos[0], cmap=plt.cm.jet)
        # plt.colorbar()
        
        
        ''' 
        Compute (ugeo, vgeo) from DH 
        
        '''
        
        ugeo, vgeo, speed = calculate_vgeo_from_DH( \
                                dh_hand, deltax, deltay, grav, coriol_mean)
            
        # This gives the same result!    
        ugeo_teos, vgeo_teos, speed_teos = calculate_vgeo_from_DH( \
                                dh_teos, deltax, deltay, grav, coriol_mean)
        
        # plt.figure()
        # plt.quiver(lon, lat, ugeo[0], vgeo[0], scale=10, color='k')
        # plt.quiver(lon, lat, ugeo_teos[0], vgeo_teos[0], scale=10, color='y')

        # plt.figure()
        # plt.subplot(211)
        # plt.quiver(lon, lat, ugeo[0], vgeo[0], scale=10, color='k')
        # plt.subplot(212)
        # plt.quiver(lon, lat, ugeo_teos[0], vgeo_teos[0], scale=10, color='y')        
        
        ''' Compute Brunt-Vaisala frequency '''
        
        # compute buoyancy
        b, binv, depinv = compute_buoyancy(grav, sig, sigma_mean)
        
        # compute brunt-vaisala frequency 
        brnv2inv, brnv2_mean, brnvinv = compute_N2(binv, deltaz)
    

        ''' Geostrophic velocity from buoyancy to compare ''' 
        
        gradsinv, gradbinv, uginv, vginv, modvgeosinv = compute_vgeo(binv, deltax, deltay, deltaz, coriol_mean)


        ''' Figure to compare all fields '''
        
        # lev = 0 
        # sc  = 5
        # iz     = np.argmin(np.abs(dep-lev))
        
        # plt.figure(figsize=(15,5))
        # plt.subplot(131)
        # plt.pcolor(lon, lat, ptem[iz], cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.title('Temperature original')
        # plt.subplot(132)
        # plt.pcolor(lon, lat, speed[iz,:,:], cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.quiver(lon, lat, ugeo[iz,:,:], vgeo[iz,:,:], color='k', 
        #            scale=sc)
        # plt.quiver(lon, lat, uginv[::-1][iz,:,:], vginv[::-1][iz,:,:], color='r', 
        #            scale=sc, alpha=0.7)
        
        # plt.subplot(133)
        # plt.pcolor(lon, lat, speed_teos[iz,:,:], cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.quiver(lon, lat, ugeo_teos[iz,:,:], vgeo_teos[iz,:,:], color='k', 
        #            scale=sc)
        # plt.quiver(lon, lat, uginv[::-1][iz,:,:], vginv[::-1][iz,:,:], color='r', 
        #            scale=sc, alpha=0.7)
        
    
        ''' Geostrophic Ro '''
        
        uginv  = ugeo[::-1,:,:]
        vginv  = vgeo[::-1,:,:]
        Roginv = compute_Rog(uginv, vginv, deltax, deltay)
        
        ''' Invert z axis '''
        
        Rog  = Roginv[::-1,:,:] 
        brnv = brnvinv[::-1,:,:]

        # exRo = max(np.nanmax(Rog),  np.abs(np.nanmin(Rog)))
        
        # plt.figure()
        # plt.pcolor(lon, lat, Rog[iz], cmap = plt.cm.RdBu_r,
        #            vmin=-exRo , vmax=exRo )
        # plt.colorbar()
        # plt.title('Ro at depth = ' + np.str(dep[iz]))
   
       
        # --------------------------------------------------
        # Save data: ug, vg, Rog, N, sig, lon, lat, dep
        # -------------------------------------------------
        [nz, ny, nx] = dh_hand.shape 
    
        fileDH  = name_conf + '_derived_variables.nc' 
                 
    
        print (' Creant arxiu... ', fileDH)
  
        createnc_derivedvar(dir_OIdata + fileDH, nx, ny, nz) 
       
        # Save data
        nc = netcdf.Dataset(dir_OIdata + fileDH, 'a', format='NETCDF3_CLASSIC')
        nc.variables['longitude'][:] = lon
        nc.variables['latitude'][:]  = lat
        nc.variables['depth'][:]     = dep
        nc.variables['time'][:]      = time_map
        nc.variables['sig'][:]       = sig
        nc.variables['dh'][:]        = dh_hand
        nc.variables['ug'][:]        = ugeo
        nc.variables['vg'][:]        = vgeo
        nc.variables['Rog'][:]       = Rog
        nc.variables['N'][:]         = brnv
        nc.close()
  
