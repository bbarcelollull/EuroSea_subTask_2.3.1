#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy                 as np
import scipy.linalg


"""
Created on Thu May 24 09:59:19 2018


>>> Optimal interpolation functions <<<

Author: Bàrbara Barceló-Llull, IMEDEA, May 2018

"""

def compute_OI_2d(lon,lat,U,loni,lati,lx,ly,ang,eps,T_mean) :
    '''
    From function: omapping1d.m (Enric Pallas, CICESE)
    
    omapping1d.m interpolates irregular data using objective mapping with 
    decorrelation scales lx and ly. 

              lon=longitude of the stations 1d array --> np.flatten()
              lat=latitude of the stations  1d array --> np.flatten()
              loni=new longitude points  1d array --> only lon points!
              lati=new latitude points   1d array --> only lat points!
              lx=decorrelation scale in x-direction
              ly=decorrelation scale in y-direction
              ang: angle of the ellipses of decorrelation (ang)
              eps: error of measurements (default=0.05)
              T_mean: anomaly remove a 'scalar' or a 'plane' (scalar for vector 
                      components and plane for a scalar quantity sigma, T, S,etc..)
    
    Adapted to python by: Bàrbara Barceló Llull, IMEDEA, May 2018
    '''
    
    
    ''' Remove nans '''
    Ud    = U[~np.isnan(U)]
    lond  = lon[~np.isnan(U)]
    latd  = lat[~np.isnan(U)]    
    
    ''' From degrees to km '''
    # Parameters:
    # factor pi , earth radio, and mean latitude
    dpi         = 4*np.arctan(1)
    factorpi    = dpi/180
    radio       = 6378 # km 
    
    # Mean latitude
    glatitude   = latd.mean() # deg

    # Coriolis parameter at mean latitude
    factorlat   = np.cos(glatitude*factorpi)
 
    # Computes gravity at mean latitude
    X1          = np.sin(glatitude*factorpi)
    X           = X1 * X1
    gravedad    = 9.780318e0 * (1+(5.2788e-3 + 2.36e-5*X)*X)

    # Convert degrees to distance in Km (1 degree ---> 111.12Km)
    # old grid in km
    lonkm  = (lond-np.min(lond))*radio*factorlat*factorpi
    latkm  = (latd-np.min(latd))*radio          *factorpi
    
    # size of the original data (= number of CTD casts)
    # hence, latkm.shape = lonkm.shape and A is squared
    nlatlon=len(lond)
    
    # new grid in km with respect to the minimum point (lon, lat)
    lonkmi = (loni-np.min(lond))*radio*factorlat*factorpi
    latkmi = (lati-np.min(latd))*radio          *factorpi
    
    Xin         = lonkmi
    Yin         = latkmi 
    [XXin,YYin] = np.meshgrid(Xin,Yin)
    Xint        = np.zeros(XXin.shape)

    # For reshapes
    nxg        = len(loni)
    nyg        = len(lati) 
    

    ''' Compute correlations with the function Fcov1 '''
    
    # Decorrelation scales
    coef=[lx,ly,ang]
    
    # Compute the function of covariance between real (original) positions [lat(i),lon(j)]
    A = np.zeros((nlatlon,nlatlon))
    for i in np.arange(nlatlon):
        for j in np.arange(nlatlon):
            if j!=i:
                A[i,j]=Fcov1(lonkm[i],latkm[i],lonkm[j],latkm[j],coef)
            else:
                A[i,j]=Fcov1(lonkm[i],latkm[i],lonkm[j],latkm[j],coef) + eps
                
    # Compute inverse of the covariance matrix function            
    Ainv = np.linalg.inv(A)
    
    
    # Computing covariance function between grid points and real positions
    C = np.zeros((nxg*nyg,nlatlon))
    for i in np.arange(nlatlon):
        for j in np.arange(nxg*nyg):
             C[j,i]=Fcov1(XXin.flatten()[j],YYin.flatten()[j],lonkm[i],latkm[i],coef)
             # in matlab is this:
             #C(j,i)=Fcov1(XXin(j),YYin(j),lonkm(i),latkm(i),coef);
             # I suppose that YYin(j) does the job XXin.flatten()[j]
       
        
    ''' Compute anomalies of the original data removing the mean or a plane'''
    # scalar
    if T_mean == 'scalar':
       Umean1d = np.mean(Ud)

    #plane
    if T_mean == 'plane':
       a1 = fitplane(lonkm, latkm, Ud)
      
       #Z = C[0]*X + C[1]*Y + C[2]
       #Umean1d = a1[0] + a1[1]*lonkm + a1[2]*latkm
       Umean1d = a1[0]*lonkm + a1[1]*latkm + a1[2]
    
    # Anomaly of the original data
    Uanomaly = Ud - Umean1d
    
    ''' OPTIMAL INTERPOLATION '''
    #uint = C*(A\Uanomaly)
    #uint=C*(A\Uanomaly)
    
    # Through the interpolation we obtain the anomaly onto the new grid points
    uint = np.matmul(C, np.matmul(Ainv, Uanomaly))
    
    
    ''' Mean or plane at the new grid points (to add to the interpolated anomaly)'''  
    # scalar
    if T_mean == 'scalar':
         # the same as before
        Umean2d = Umean1d      #constant value

    # mean plane
    if T_mean == 'plane':
        # apply the previous plane adjustment to the interpolation grid points
        # interpolation grid 2d, so Umean2d is 2d and does not need a reshape
        Umean2d = a1[0]*XXin + a1[1]*YYin + a1[2]
    
    ''' Add the mean value or mean plane to the interpolated anomaly '''
    # Xint=reshape(uint,nxg,nyg)+Umean2d --> I don't know what happens with Umean2d
    Xint = np.reshape(uint, XXin.shape)+ Umean2d
    
    
    ''' Infer the normalized error '''
    # Computing normalized squared percent error relative to the "true"
    numErr = np.matmul(C, np.matmul(Ainv, np.transpose(C))) #[total num grid points, total num grid points]
    
    # For each grid point i: error(i) = Diag(1 - C*Ainv*C.T)(i) = 1 - C*Ainv*C.T(i,i)
    # So: error(5) = 1 - C*Ainv*C.T(5,5), and so on.
    # Total num grid points = N = numErr.shape[0]
    Err = np.zeros(numErr.shape[0])
    for i in np.arange(numErr.shape[0]):
      Err[i] = 1 - numErr[i,i]
    
    # scaled to the observation variance(proves comparaciÛ ananda)
    # Err = std(U) * (Err.^0.5);    
    
    epsilon = np.reshape(Err, XXin.shape)
    
    return XXin, YYin, Xint, epsilon
    


def  Fcov1(x1,y1,x2,y2,a):
    
    '''
    From function:
        
    [f]=Fcov1(x1,y1,x2,y2,a) computes anisotropic tilted gaussian covariance 
                            between points (x1,y1) and (x2,y2)
 
    INPUT:  (x1,y1),         coordinates of point 1
           (x2,y2),         coordinates of point 2
           a=[lx,ly,theta], parameters of gaussian function 
             
             -(lx,ly) are the decorralation scales
             -theta is the angle of the gaussian function vs y-axes
   
    Author: Enric Pallàs Sanz 
           Scripps Institution of Oceanography 
           CASPO 2009

    Adapted to python by: Bàrbara Barceló Llull, IMEDEA, May 2018
    '''

    x1 =  x1*np.cos(a[2]) + y1*np.sin(a[2])
    y1 = -x1*np.sin(a[2]) + y1*np.cos(a[2])
    x2 =  x2*np.cos(a[2]) + y2*np.sin(a[2])
    y2 = -x2*np.sin(a[2]) + y2*np.cos(a[2])

    Tx2 =(x1-x2)**2 / (2*(a[0]**2))
    Ty2 =(y1-y2)**2 / (2*(a[1]**2))
    f   = np.exp(-Tx2-Ty2)
    
    return f

def compute_OI_2d_time(lon,lat,time,U,loni,lati,timei,lx,ly,lt,ang,eps,T_mean) :
    
    '''
    From function: omapping1d.m (Enric Pallas, CICESE)
    
    omapping1d.m interpolates irregular data using objective mapping with 
    decorrelation scales lx and ly. 
    >>> Now we also include the time variable of the observations and
    the time correlation in the correlation function. <<<

              lon=longitude of the stations 1d array --> np.flatten()
              lat=latitude of the stations  1d array --> np.flatten()
              time= time of the observations [in days]
              U = variable to interpolate
              loni=new longitude points  1d array --> only lon points!
              lati=new latitude points   1d array --> only lat points!
              timei = time of the interpolated field, 1 float [in days]
              lx=decorrelation scale in x-direction
              ly=decorrelation scale in y-direction
              lt= temporal correlation scale [in days]
              ang: angle of the ellipses of decorrelation (ang)
              eps: error of measurements (default=0.05) --> noise-to-signal variance
              T_mean: anomaly remove a 'scalar' or a 'plane' (scalar for vector 
                      components and plane for a scalar quantity sigma, T, S,etc..)
    
    Adapted to python by: Bàrbara Barceló Llull, IMEDEA, May 2018
    
    15 November 2021: include the time variable and correlation temporal scale
    following Escudier et al., 2013.

    '''
    
    
    ''' Remove nans '''
    Ud    = U[~np.isnan(U)]
    lond  = lon[~np.isnan(U)]
    latd  = lat[~np.isnan(U)]  
    timed = time[~np.isnan(U)] # in days
    
    ''' From degrees to km '''
    # Parameters:
    # factor pi , earth radio, and mean latitude
    dpi         = 4*np.arctan(1)
    factorpi    = dpi/180
    radio       = 6378 # km 
    
    # Mean latitude
    glatitude   = latd.mean() # deg

    # Coriolis parameter at mean latitude
    factorlat   = np.cos(glatitude*factorpi)
 
    # Computes gravity at mean latitude
    X1          = np.sin(glatitude*factorpi)
    X           = X1 * X1
    gravedad    = 9.780318e0 * (1+(5.2788e-3 + 2.36e-5*X)*X)

    # Convert degrees to distance in Km (1 degree ---> 111.12Km)
    # old grid in km
    lonkm  = (lond-np.min(lond))*radio*factorlat*factorpi
    latkm  = (latd-np.min(latd))*radio          *factorpi
    
    # size of the original data (= number of CTD casts)
    # hence, latkm.shape = lonkm.shape and A is squared
    nlatlon=len(lond)
    
    # new grid in km with respect to the minimum point (lon, lat)
    lonkmi = (loni-np.min(lond))*radio*factorlat*factorpi
    latkmi = (lati-np.min(latd))*radio          *factorpi
    
    Xin         = lonkmi
    Yin         = latkmi 
    [XXin,YYin] = np.meshgrid(Xin,Yin)
    Xint        = np.zeros(XXin.shape)
    

    # I will need this later to reshape
    nxg        = len(loni)
    nyg        = len(lati) 
    

    # time of the interpolated field will be the same for all the map
    TTin = np.ones(XXin.shape) * timei
    
    
    ''' Compute correlations between original points with the function Fcov1_time '''
    
    # Decorrelation scales
    coeft = [lx,ly,lt,ang]
    
    # Compute the function of covariance between real (original) positions 
    # including the time variable
    # Escudier et al., 2013: This correlation scheme is used to determine 
    # the weights for the data interpolation.
    
    A = np.zeros((nlatlon,nlatlon))
    for i in np.arange(nlatlon):
        for j in np.arange(nlatlon):
            if j!=i:
                A[i,j]=Fcov1_time(lonkm[i],latkm[i],timed[i],
                                  lonkm[j],latkm[j],timed[j],coeft)
            else:
                A[i,j]=Fcov1_time(lonkm[i],latkm[i],timed[i],
                                  lonkm[j],latkm[j],timed[j],coeft) + eps
                
    # Compute inverse of the covariance matrix function            
    Ainv = np.linalg.inv(A)
    
    ''' Compute correlations between grid and original points with the function Fcov1 '''
    
    #coef  = [lx,ly,ang]


    # Computing covariance function between grid points and real positions
    C = np.zeros((nxg*nyg,nlatlon))
    for i in np.arange(nlatlon):
        for j in np.arange(nxg*nyg):
             C[j,i]=Fcov1_time(XXin.flatten()[j],YYin.flatten()[j],TTin.flatten()[j],
                               lonkm[i],latkm[i],timed[i],coeft)
             #C[j,i]=Fcov1(XXin.flatten()[j],YYin.flatten()[j],lonkm[i],latkm[i],coef)
             
       
        
    ''' Compute anomalies of the original data removing the mean or a plane'''
    # scalar
    if T_mean == 'scalar':
       Umean1d = np.mean(Ud)

    #plane
    if T_mean == 'plane':
       a1 = fitplane(lonkm, latkm, Ud)
      
       #Z = C[0]*X + C[1]*Y + C[2]
       #Umean1d = a1[0] + a1[1]*lonkm + a1[2]*latkm
       Umean1d = a1[0]*lonkm + a1[1]*latkm + a1[2]
    
    # Anomaly of the original data
    Uanomaly = Ud - Umean1d
    
    ''' OPTIMAL INTERPOLATION '''
    #uint = C*(A\Uanomaly)
    #uint=C*(A\Uanomaly)
    
    # Through the interpolation we obtain the anomaly onto the new grid points
    uint = np.matmul(C, np.matmul(Ainv, Uanomaly))
    
    
    ''' Mean or plane at the new grid points (to add to the interpolated anomaly)'''  
    # scalar
    if T_mean == 'scalar':
         # the same as before
        Umean2d = Umean1d      #constant value

    # mean plane
    if T_mean == 'plane':
        # apply the previous plane adjustment to the interpolation grid points
        # interpolation grid 2d, so Umean2d is 2d and does not need a reshape
        Umean2d = a1[0]*XXin + a1[1]*YYin + a1[2]
    
    ''' Add the mean value or mean plane to the interpolated anomaly '''
    # Xint=reshape(uint,nxg,nyg)+Umean2d --> I don't know what happens with Umean2d
    Xint = np.reshape(uint, XXin.shape)+ Umean2d
    
    
    ''' Infer the normalized error '''
    # Computing normalized squared percent error relative to the "true"
    numErr = np.matmul(C, np.matmul(Ainv, np.transpose(C))) #[total num grid points, total num grid points]
    
    # For each grid point i: error(i) = Diag(1 - C*Ainv*C.T)(i) = 1 - C*Ainv*C.T(i,i)
    # So: error(5) = 1 - C*Ainv*C.T(5,5), and so on.
    # Total num grid points = N = numErr.shape[0]
    Err = np.zeros(numErr.shape[0])
    for i in np.arange(numErr.shape[0]):
      Err[i] = 1 - numErr[i,i]
    
    # scaled to the observation variance(proves comparaciÛ ananda)
    # Err = std(U) * (Err.^0.5);    
    
    epsilon = np.reshape(Err, XXin.shape)
    
    return XXin, YYin, TTin, Xint, epsilon


def  Fcov1_time(x1,y1,t1,x2,y2,t2,a):
    
    '''

    [f]=Fcov1_time(x1,y1,t1,x2,y2,t2,a) computes anisotropic tilted 
                            gaussian covariance 
                            between points (x1,y1) and (x2,y2) 
                            including also in the correlation function 
                            the time variable.
 
    INPUT:  (x1,y1,t1),         coordinates of point 1
            (x2,y2,t2),         coordinates of point 2
           a=[lx,ly,lt,theta], parameters of gaussian function 
             
             -(lx,ly,lt) are the decorralation scales
             -theta is the angle of the gaussian function vs y-axes
   
    Author: Enric Pallàs Sanz 
           Scripps Institution of Oceanography 
           CASPO 2009

    Adapted to python by: Bàrbara Barceló Llull, IMEDEA, May 2018
    
    15 November 2021: include the time variable and correlation temporal scale
    
    See Escudier et al., 2013 for details.
    
    '''

    x1 =  x1*np.cos(a[3]) + y1*np.sin(a[3])
    y1 = -x1*np.sin(a[3]) + y1*np.cos(a[3])
    x2 =  x2*np.cos(a[3]) + y2*np.sin(a[3])
    y2 = -x2*np.sin(a[3]) + y2*np.cos(a[3])

    Tx2 =(x1-x2)**2 / (2*(a[0]**2))
    Ty2 =(y1-y2)**2 / (2*(a[1]**2))
    Tt2 =(t1-t2)**2 / (a[2]**2) # This is the exponent of the temporal Gaussian
    f   = np.exp(-Tx2-Ty2-Tt2)
    
    return f

def fitplane(x,y,z):
    
    '''
    From function:
        
     Fits a plane of the form a(1)+a(2)*x+a(3)*y to the spatially 
     distributed vector data using least-squares fitting. 

           

       z:       vector data at each (x,y) coordinates
       x:       x-axis
       y:       y-axis
       a:       parameters of the mean plane fitted


          adapted from detrend2.m on 30 Oct 2008
                  by 

                     PhD. Pallas-Sanz, E.
                     Scripps Institution of Oceanography
                     La Jolla, San Diego, CA

    Adapted to python by: Bàrbara Barceló Llull, IMEDEA, May 2018
    '''

    znonan = z[~np.isnan(z)]
    xnonan = x[~np.isnan(z)]
    ynonan = y[~np.isnan(z)]
    
    '''
    Original code: 
        
    M2=[ones(length(xnonan(:)),1) xnonan(:) ynonan(:)];
    a=(M2'*M2)\M2'*znonan
    '''
    
    
    '''
    From: https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
    
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    '''    
        
    data = np.c_[xnonan,ynonan,znonan]
    
    # best-fit linear plane
    M2      = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    a,_,_,_ = scipy.linalg.lstsq(M2, data[:,2])    # coefficients Z = a[0]*X + a[1]*Y + a[2]

    return a
