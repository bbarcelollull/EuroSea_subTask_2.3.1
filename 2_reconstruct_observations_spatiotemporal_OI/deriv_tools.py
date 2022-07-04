# -*- coding: utf-8 -*-
# deriv_tools.py

import numpy                 as np

''' 
Code to compute different derivatives. 

B. Barceló-LLull ULPGC 13 July 2015
from Enric Pallàs matlab functions
'''


def deriv1d_3d(data, axis, delta):
    '''
    deriv1d_3d computes the spatial derivates (in one direction) of a 
    3d array. Forward/backward derivates (1st order) are used for 
    border region and central derivates (2d order) for the interior 
    region. 

    INPUTS: 
    data:  3d grided input array
    axis:  string containing the direction of derivation:

           'I': derivates along x-direction
           'J': derivates along y-direction
           'K': derivates along z-direction
   
    delta: grid spacing along the direction selected (units: m).

    USAGE: deriv1d_3d (data,'K',0.2)

    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    '''

    print ('  ')
    print ('deriv1d_3d')
    [nz, ny, nx] = data.shape

    deri   = np.zeros(data.shape)
    delta2 = 2*delta

    if axis == 'K':
        print ('1d derivative in K direction')
        deri[0,:,:]    = (data[1,:,:]-data[0,:,:])/delta
        deri[1:-1,:,:] = (data[2:,:,:]-data[:-2,:,:])/delta2
        deri[-1,:,:]   = (data[-1,:,:]-data[-2,:,:])/delta
        return deri
    elif axis == 'J':
        print ('1d derivative in J direction')
        deri[:,0,:]    = (data[:,1,:]-data[:,0,:])/delta
        deri[:,1:-1,:] = (data[:,2:,:]-data[:,:-2,:])/delta2
        deri[:,-1,:]   = (data[:,-1,:]-data[:,-2,:])/delta  
        return deri
    elif axis == 'I':
        print ('1d derivative in I direction')
        deri[:,:,0]    = (data[:,:,1]-data[:,:,0])/delta
        deri[:,:,1:-1] = (data[:,:,2:]-data[:,:,:-2])/delta2
        deri[:,:,-1]   = (data[:,:,-1]-data[:,:,-2])/delta   
        return deri
    else:
        print ('Which direction?'  )          


def deriv1d_2d(data, axis, delta):
    '''
    deriv1d_2d computes the spatial derivates (in one direction) of a 
    2d array. Forward/backward derivates (1st order) are used for 
    border region and central derivates (2d order) for the interior 
    region. 

    INPUTS: 
    data:  2d grided input array
    axis:  string containing the direction of derivation:

           'I': derivates along x-direction
           'K': derivates along z-direction
   
    delta: grid spacing along the direction selected (units: m).

    USAGE: deriv1d_3d (data,'K',0.2)

    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    '''

    print ('  ')
    print ('deriv1d_2d')
    [nx, nz] = data.shape

    deri   = np.zeros(data.shape)
    delta2 = 2*delta

    if axis == 'K':
        print ('1d derivative in K direction')
        deri[:, 0]    = (data[:, 1]-data[:, 0])/delta
        deri[:, 1:-1] = (data[:, 2:]-data[:, :-2])/delta2
        deri[:, -1]   = (data[:, -1]-data[:, -2])/delta
        return deri

    elif axis == 'I':
        print ('1d derivative in I direction')
        deri[0, :]    = (data[1, :]-data[0, :])/delta
        deri[1:-1, :] = (data[2:, :]-data[:-2, :])/delta2
        deri[-1, :]   = (data[-1, :]-data[-2, :])/delta   
        return deri
    else:
        print ('Which direction?'  )    

def deriv1d_1d(data, delta):
    '''
    deriv1d_1d computes the spatial derivates in one direction of a 
    1d array. Forward/backward derivates (1st order) are used for 
    border region and central derivates (2d order) for the interior 
    region. 

    INPUTS: 
    data:  1d grided input array
   
    delta: grid spacing  (units: m).

    USAGE: deriv1d_1d (data,0.2)

    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, November 2015
    '''

    print ('  ')
    print ('deriv1d_1d')
    
    deri   = np.zeros(data.shape)
    delta2 = 2*delta

    print ('1d derivative in I direction')
    deri[0]    = (data[1]-data[0])/delta
    deri[1:-1] = (data[2:]-data[:-2])/delta2
    deri[-1]   = (data[-1]-data[-2])/delta   
    return deri
          
def grad2d_2d(data,dx,dy):
    '''
    grad2d_2d computes the 2d gradient of a 2d array. Forward/backward 
    derivates (1st order) are used for border region and 
    central derivates (2d order) for the interior region. 
                 
              d(data)/dx , d(data)/dy   

    INPUTS: 
        data:  2d grided input array
        dx: grid spacing along x-direction (units: m)
        dy: grid spacing along y-direction (units: m)
   
    OUTPUTS: a 2d complex array containing the horizontal gradient. 
             Real part corresponds to x-derivatives and Imaginary 
             part to y-derivatives.

    USAGE: grad = grad2d_2d(data,0.2,0.2)

    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    '''

    # print ('  ')
    # print ('grad2d_2d')
    [ny, nx] = data.shape
    graddata = np.zeros(data.shape) + 1j * np.zeros(data.shape)
    dx2 = 2*dx
    dy2 = 2*dy
        
    # inner region
    graddata[1:-1, 1:-1] = (data[1:-1, 2:] - 
                               data[1:-1, :-2])/dx2 + \
                                    1j * (data[2:,  1:-1] - 
                                          data[:-2, 1:-1])/dy2
                 
    # border region
    # ix=0, iy=1:-1
    graddata[1:-1, 0] = (data[1:-1, 1] - 
                            data[1:-1, 0])/dx + \
                                 1j * (data[2:, 0] - 
                                       data[:-2, 0])/dy2
    # ix=-1, iy=1:-1
    graddata[1:-1, -1] = (data[1:-1, -1] - 
                             data[1:-1, -2])/dx + \
                                 1j * (data[2:, -1] - 
                                       data[:-2, -1])/dy2
                                                                                                            
    # ix=1:-1, iy=0
    graddata[0, 1:-1] = (data[0, 2:] - 
                            data[ 0, :-2])/dx2 + \
                                 1j * (data[1, 1:-1] - 
                                       data[0, 1:-1])/dy
    # ix=1:-1, iy=-1
    graddata[-1, 1:-1] = (data[-1, 2:] - 
                             data[-1, :-2])/dx2 + \
                                 1j * (data[-1, 1:-1] - 
                                       data[-2, 1:-1])/dy
    # Four corners left
    # ix=0, iy=0
    graddata[0, 0] = (data[0, 1] - 
                         data[0, 0])/dx + \
                             1j * (data[1, 0] - 
                                   data[0, 0])/dy                
                                                     
    # ix=-1, iy=-1
    graddata[-1, -1] = (data[-1, -1] - 
                           data[-1, -2])/dx + \
                             1j * (data[-1, -1] - 
                                   data[-2, -1])/dy    
    # ix=0, iy=-1
    graddata[-1, 0] = (data[-1, 1] - 
                          data[-1, 0])/dx + \
                             1j * (data[-1, 0] - 
                                   data[-2, 0])/dy    
    # ix=-1, iy=0
    graddata[0, -1] = (data[0, -1] - 
                          data[0, -2])/dx + \
                             1j * (data[1, -1] - 
                                   data[0, -1])/dy      
                                                                    
    return graddata  

def grad2d_3d(data,dx,dy):
    '''
    grad2d_3d computes the 2d gradient of a 3d array. Forward/backward 
    derivates (1st order) are used for border region and 
    central derivates (2d order) for the interior region. 
                 
              d(data)/dx , d(data)/dy   

    INPUTS: 
        data:  3d grided input array
        dx: grid spacing along x-direction (units: m)
        dy: grid spacing along y-direction (units: m)
   
    OUTPUTS: a 3d complex array containing the horizontal gradient. 
             Real part corresponds to x-derivatives and Imaginary 
             part to y-derivatives.

    USAGE: grad = grad2d_3d(data,0.2,0.2)

    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    '''

    print ('  ')
    print ('grad2d_3d')
    [nz, ny, nx] = data.shape
    graddata = np.zeros(data.shape) + 1j * np.zeros(data.shape)
    dx2 = 2*dx
    dy2 = 2*dy
        
    # inner region
    graddata[:, 1:-1, 1:-1] = (data[:, 1:-1, 2:] - 
                               data[:, 1:-1, :-2])/dx2 + \
                                    1j * (data[:, 2:,  1:-1] - 
                                          data[:, :-2, 1:-1])/dy2
                 
    # border region
    # ix=0, iy=1:-1
    graddata[:, 1:-1, 0] = (data[:, 1:-1, 1] - 
                            data[:, 1:-1, 0])/dx + \
                                 1j * (data[:, 2:, 0] - 
                                       data[:, :-2, 0])/dy2
    # ix=-1, iy=1:-1
    graddata[:, 1:-1, -1] = (data[:, 1:-1, -1] - 
                             data[:, 1:-1, -2])/dx + \
                                 1j * (data[:, 2:, -1] - 
                                       data[:, :-2, -1])/dy2
                                                                                                            
    # ix=1:-1, iy=0
    graddata[:, 0, 1:-1] = (data[:, 0, 2:] - 
                            data[:, 0, :-2])/dx2 + \
                                 1j * (data[:, 1, 1:-1] - 
                                       data[:, 0, 1:-1])/dy
    # ix=1:-1, iy=-1
    graddata[:, -1, 1:-1] = (data[:, -1, 2:] - 
                             data[:, -1, :-2])/dx2 + \
                                 1j * (data[:, -1, 1:-1] - 
                                       data[:, -2, 1:-1])/dy
    # Four corners left
    # ix=0, iy=0
    graddata[:, 0, 0] = (data[:, 0, 1] - 
                         data[:, 0, 0])/dx + \
                             1j * (data[:, 1, 0] - 
                                   data[:, 0, 0])/dy                
                                                     
    # ix=-1, iy=-1
    graddata[:, -1, -1] = (data[:, -1, -1] - 
                           data[:, -1, -2])/dx + \
                             1j * (data[:, -1, -1] - 
                                   data[:, -2, -1])/dy    
    # ix=0, iy=-1
    graddata[:, -1, 0] = (data[:, -1, 1] - 
                          data[:, -1, 0])/dx + \
                             1j * (data[:, -1, 0] - 
                                   data[:, -2, 0])/dy    
    # ix=-1, iy=0
    graddata[:, 0, -1] = (data[:, 0, -1] - 
                          data[:, 0, -2])/dx + \
                             1j * (data[:, 1, -1] - 
                                   data[:, 0, -1])/dy      
                                                                    
    return graddata    
    
    
def lap2d_3d(data,dx,dy):
    '''
    lap2d_3d computes the 2d laplacian of a 3d array. Forward/backward 
    derivates (1st order) are used for border region and 
    central derivates (2d order) for the interior region. 
                 
              d²(data)/dx² , d²(data)/dy²   

    INPUTS: 
        data:  3d grided input array
        dx: grid spacing along x-direction (units: m)
        dy: grid spacing along y-direction (units: m)
   
    OUTPUTS: a 3d complex array containing the horizontal laplacian. Real part
            corresponds to x-derivatives and Imaginary part to y-derivatives.

    USAGE: grad = lap2d_3d(data,0.2,0.2)
    
    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    '''

    print ('  ')
    print ('lap2d_3d'  )
    [nz, ny, nx] = data.shape
    lapdata = np.zeros(data.shape) + 1j * np.zeros(data.shape)
    dx22 = dx**2
    dy22 = dy**2
    
    # inner region
    a = (data[:, 1:-1, 2:] - 2*data[:, 1:-1, 1:-1] + data[:, 1:-1, :-2])/dx22
    b = (data[:, 2:, 1:-1] - 2*data[:, 1:-1, 1:-1] + data[:, :-2, 1:-1])/dy22
    lapdata[:, 1:-1, 1:-1] = a + 1j*b
    
    # border lines
    # ix=0, iy=1:-1
    a = 2*lapdata[:, 1:-1, 1].real - lapdata[:, 1:-1, 2].real #
    b = (data[:, 2:, 0] - 2*data[:, 1:-1, 0] + data[:, :-2, 0])/dy22
    lapdata[:, 1:-1, 0] = a + 1j * b   
    
    # ix=-1, iy=1:-1
    a = 2*lapdata[:, 1:-1, -2].real - lapdata[:, 1:-1, -3].real #
    b = (data[:, 2:, -1] - 2*data[:, 1:-1, -1] + data[:, :-2, -1])/dy22
    lapdata[:, 1:-1, -1] = a + 1j * b     
        
    # ix=1:-1, iy=0
    a = (data[:, 0, 2:] - 2*data[:, 0, 1:-1] + data[:, 0, :-2])/dx22
    b = 2*lapdata[:, 1, 1:-1].imag - lapdata[:, 2, 1:-1].imag # comprovar
    #b2 = (data[:, 2, 1:-1] - 2*data[:, 1, 1:-1] + data[:, 0, 1:-1])/dy22
    lapdata[:, 0, 1:-1] = a + 1j * b  
                             
    # ix=1:-1, iy=-1
    a = (data[:, -1, 2:] - 2*data[:, -1, 1:-1] + data[:, -1, :-2])/dx22
    b = 2*lapdata[:, -2, 1:-1].imag - lapdata[:, -3, 1:-1].imag 
    lapdata[:, -1, 1:-1] = a + 1j*b
    
    # four corner left
    # ix=0, iy=0
    a = 2*lapdata[:, 0, 1].real - lapdata[:, 0, 2].real
    b = 2*lapdata[:, 1, 0].imag - lapdata[:, 2, 0].imag
    lapdata[:, 0, 0] = a + 1j * b
    
    # ix=-1, iy=0
    a = 2*lapdata[:, 0, -2].real - lapdata[:, 0, -3].real
    b = 2*lapdata[:, 1, -1].imag - lapdata[:, 2, -1].imag
    lapdata[:, 0, -1] = a + 1j * b    
    
    # ix=0, iy=-1
    a = 2*lapdata[:, -1, 1].real - lapdata[:, -1, 2].real
    b = 2*lapdata[:, -2, 0].imag - lapdata[:, -3, 0].imag
    lapdata[:, -1, 0] = a + 1j * b

    # ix=-1, iy=-1
    a = 2*lapdata[:, -1, -2].real - lapdata[:, -1, -3].real
    b = 2*lapdata[:, -2, -1].imag - lapdata[:, -3, -1].imag
    lapdata[:, -1, -1] = a + 1j * b    
    
    return lapdata
    
def Q_from_v_gradb(v,gradb,dx,dy):
    '''
    Q_from_v_gradb computes the Q-vector from a 3d array of velocity 
    and buoyancy horizontal gradient.
    Forward/backward derivates (1st order) are used for the border region 
    and central derivates (2d order) for the interior region. 
                 
    
                    Q = gradH v · gradH(b) 
                 
      Q = ( du/dx*db/dx + dv/dx*db/dy, du/dy*db/dx + dv/dy*db/dy)
      (Hosking and Pedder 1980 eq. 5)
      
                 UNITS= [s^-2]   

    INPUTS: 
        v:     3d grided velocity array (complex). 
               It can be: vgeos, vageos or vadcp.
        gradb: 3d grided buoyancy horizontal gradient array (complex)
        deltax: horizontal grid spacing x-direction (m)
        deltay: horizontal grid spacing y-direction (m)
   
    OUTPUTS: a 3d complex array containing the components of the Q-vector
            (complex array).

    USAGE: Q = Q_from_v_gradb(v,gradb,0.2,0.2)

    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    
    '''
    print ('  ')
    print ('Q_from_v_gradb'   )   
    [nz, ny, nx] = v.shape
    
    dx2 = 2*dx
    dy2 = 2*dy
    
    # Checking for dimensions
    [nzz, nyy, nxx] = gradb.shape
    
    if nx!=nxx or ny!=nyy or nz!=nzz:
        print ('error: check vegos and gradb dimensions')
        return
        
    Q = np.zeros(v.shape) + 1j * np.zeros(v.shape)
    
    # inner region
    
#    a = (((v[:, 1:-1, 2:].real - v[:, 1:-1, :-2].real)/dx2) \
#                         * gradb[:, 1:-1, 1:-1].real) 
#    b = (((v[:, 1:-1, 2:].imag - v[:, 1:-1, :-2].imag)/dx2) \
#                         * gradb[:, 1:-1, 1:-1].imag)  
#    c = (((v[:, 2:, 1:-1].real - v[:, :-2, 1:-1].real)/dy2) \
#                         * gradb[:, 1:-1, 1:-1].real)
#    d = (((v[:, 2:, 1:-1].imag - v[:, :-2, 1:-1].imag)/dy2) \
#                         * gradb[:, 1:-1, 1:-1].imag)
#                         
#    print  'inner region a min = ', a.min()
#    print  'inner region a max = ', a.max()   
#    print  '  '
#    print  'inner region b min = ', b.min()
#    print  'inner region b max = ', b.max()                              
#    print  '  '
#    print  'inner region c min = ', c.min()
#    print  'inner region c max = ', c.max() 
#    print  '  '
#    print  'inner region d min = ', d.min()
#    print  'inner region d max = ', d.max() 
    
                  
    Q[:, 1:-1, 1:-1] = ( (((v[:, 1:-1, 2:].real - v[:, 1:-1, :-2].real)/dx2) \
                         * gradb[:, 1:-1, 1:-1].real) \
                         + (((v[:, 1:-1, 2:].imag - v[:, 1:-1, :-2].imag)/dx2) \
                         * gradb[:, 1:-1, 1:-1].imag) ) \
                       + 1j * ( (((v[:, 2:, 1:-1].real - v[:, :-2, 1:-1].real)/dy2) \
                         * gradb[:, 1:-1, 1:-1].real) \
                         + (((v[:, 2:, 1:-1].imag - v[:, :-2, 1:-1].imag)/dy2) \
                         * gradb[:, 1:-1, 1:-1].imag) )
                         
#    print  '  '                     
#    print  'inner region Qx min = ', Q[:, 1:-1, 1:-1].real.min()
#    print  'inner region Qx max = ', Q[:, 1:-1, 1:-1].real.max()   
#    print  '  '
#    print  'inner region Qy min = ', Q[:, 1:-1, 1:-1].imag.min()
#    print  'inner region Qy max = ', Q[:, 1:-1, 1:-1].imag.max() 
                    
    # border region    
    
    # ix=0, iy=1:-1 
    Q[:, 1:-1, 0] = ( (((v[:, 1:-1, 1].real - v[:, 1:-1, 0].real)/dx) \
                         * gradb[:, 1:-1, 0].real) \
                         + (((v[:, 1:-1, 1].imag - v[:, 1:-1, 0].imag)/dx) \
                         * gradb[:, 1:-1, 0].imag) ) \
                       + 1j * ( (((v[:, 2:, 0].real - v[:, :-2, 0].real)/dy2) \
                         * gradb[:, 1:-1, 0].real) \
                         + (((v[:, 2:, 0].imag - v[:, :-2, 0].imag)/dy2) \
                         * gradb[:, 1:-1, 0].imag) )    
    
    # ix=-1, iy=1:-1
    Q[:, 1:-1, -1] = ( (((v[:, 1:-1, -1].real - v[:, 1:-1, -2].real)/dx) \
                         * gradb[:, 1:-1, -1].real) \
                         + (((v[:, 1:-1, -1].imag - v[:, 1:-1, -2].imag)/dx) \
                         * gradb[:, 1:-1, -1].imag) ) \
                       + 1j * ( (((v[:, 2:, -1].real - v[:, :-2, -1].real)/dy2) \
                         * gradb[:, 1:-1, -1].real) \
                         + (((v[:, 2:, -1].imag - v[:, :-2, -1].imag)/dy2) \
                         * gradb[:, 1:-1, -1].imag) )                        
                         
    # ix=1:-1 , iy=0
    Q[:, 0, 1:-1] = ( (((v[:, 0, 2:].real - v[:, 0, :-2].real)/dx2) \
                         * gradb[:, 0, 1:-1].real) \
                         + (((v[:, 0, 2:].imag - v[:, 0, :-2].imag)/dx2) \
                         * gradb[:, 0, 1:-1].imag) ) \
                       + 1j * ( (((v[:, 1, 1:-1].real - v[:, 0, 1:-1].real)/dy) \
                         * gradb[:, 0, 1:-1].real) \
                         + (((v[:, 1, 1:-1].imag - v[:, 0, 1:-1].imag)/dy) \
                         * gradb[:, 0, 1:-1].imag) )  
    
    # ix=1:-1 , iy=-1
    Q[:, -1, 1:-1] = ( (((v[:, -1, 2:].real - v[:, -1, :-2].real)/dx2) \
                         * gradb[:, -1, 1:-1].real) \
                         + (((v[:, -1, 2:].imag - v[:, -1, :-2].imag)/dx2) \
                         * gradb[:, -1, 1:-1].imag) ) \
                       + 1j * ( (((v[:, -1, 1:-1].real - v[:, -2, 1:-1].real)/dy) \
                         * gradb[:, -1, 1:-1].real) \
                         + (((v[:, -1, 1:-1].imag - v[:, -2, 1:-1].imag)/dy) \
                         * gradb[:, -1, 1:-1].imag) ) 
                         
    # four corner left                     
    # ix=0, iy=0 
    Q[:, 0, 0] = ( (((v[:, 0, 1].real - v[:, 0, 0].real)/dx) \
                         * gradb[:, 0, 0].real) \
                         + (((v[:, 0, 1].imag - v[:, 0, 0].imag)/dx) \
                         * gradb[:, 0, 0].imag) ) \
                       + 1j * ( (((v[:, 1, 0].real - v[:, 0, 0].real)/dy) \
                         * gradb[:, 0, 0].real) \
                         + (((v[:, 1, 0].imag - v[:, 0, 0].imag)/dy) \
                         * gradb[:, 0, 0].imag) )                          
    # ix=0, iy=-1 
    Q[:, -1, 0] = ( (((v[:, -1, 1].real - v[:, -1, 0].real)/dx) \
                         * gradb[:, -1, 0].real) \
                         + (((v[:, -1, 1].imag - v[:, -1, 0].imag)/dx) \
                         * gradb[:, -1, 0].imag) ) \
                       + 1j * ( (((v[:, -1, 0].real - v[:, -2, 0].real)/dy) \
                         * gradb[:, -1, 0].real) \
                         + (((v[:, -1, 0].imag - v[:, -2, 0].imag)/dy) \
                         * gradb[:, -1, 0].imag) )
    # ix=-1, iy=0                      
    Q[:, 0, -1] = ( (((v[:, 0, -1].real - v[:, 0, -2].real)/dx) \
                         * gradb[:, 0, -1].real) \
                         + (((v[:, 0, -1].imag - v[:, 0, -2].imag)/dx) \
                         * gradb[:, 0, -1].imag) ) \
                       + 1j * ( (((v[:, 1, -1].real - v[:, 0, -1].real)/dy) \
                         * gradb[:, 0, -1].real) \
                         + (((v[:, 1, -1].imag - v[:, 0, -1].imag)/dy) \
                         * gradb[:, 0, -1].imag) )                               
    # ix=-1, iy=-1                      
    Q[:, -1, -1] = ( (((v[:, -1, -1].real - v[:, -1, -2].real)/dx) \
                         * gradb[:, -1, -1].real) \
                         + (((v[:, -1, -1].imag - v[:, -1, -2].imag)/dx) \
                         * gradb[:, -1, -1].imag) ) \
                       + 1j * ( (((v[:, -1, -1].real - v[:, -2, -1].real)/dy) \
                         * gradb[:, -1, -1].real) \
                         + (((v[:, -1, -1].imag - v[:, -2, -1].imag)/dy) \
                         * gradb[:, -1, -1].imag) )   
                         
    return Q                      
         
def divQ_from_Q(data,dx,dy):
    '''
    Q_from_gradb C CALCULA LA DIVERGENCIA DEL VECTOR Q
                 
    
                    divQ = dQx/dx + dQy/dy

    INPUTS: 
        data:  complex 3d grided input array of Q
        dx: grid spacing along x-direction (units: m)
        dy: grid spacing along y-direction (units: m)
   
    OUTPUTS: a 3d complex array containing the divergence of the Q-vector

    USAGE: divQ=divQ_from_Q(Q,0.2,0.2)
    
    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    
    '''
    print ('  ')
    print ('divQ_from_Q'  )    
    [nz, ny, nx] = data.shape         
       
    divQ = np.zeros(data.shape)   
                              
    dx2 = 2*dx
    dy2 = 2*dy
    
    # inner region
    divQ[:, 1:-1, 1:-1] = ((data[:, 1:-1, 2:].real - 
                            data[:, 1:-1, :-2].real)/dx2) + \
                             ((data[:, 2:, 1:-1].imag - 
                               data[:, :-2, 1:-1].imag)/dy2)
                               
    #print  '  '                     
    #print  'inner region divQ min = ', divQ[:, 1:-1, 1:-1].min()
    #print  'inner region divQ max = ', divQ[:, 1:-1, 1:-1].max()       
                            
    # border lines                        
    # ix=0, iy=1:-1                        
    divQ[:, 1:-1, 0] = ((data[:, 1:-1, 1].real - 
                         data[:, 1:-1, 0].real)/dx) + \
                          ((data[:, 2:, 0].imag -
                            data[:, :-2, 0].imag)/dy2)    
  
    #print  '  '                     
    #print  'border lines ix=0, iy=1:-1 divQ min = ', divQ[:, 1:-1, 0].min()
    #print  'border lines ix=0, iy=1:-1 divQ max = ', divQ[:, 1:-1, 0].max()     
     
    # ix=-1, iy=1:-1                        
    divQ[:, 1:-1, -1] = ((data[:, 1:-1, -1].real - 
                          data[:, 1:-1, -2].real)/dx) + \
                          ((data[:, 2:, -1].imag -
                            data[:, :-2, -1].imag)/dy2)  
    #print  '  '                     
    #print  'border lines ix=-1, iy=1:-1 divQ min = ', divQ[:, 1:-1, -1].min()
    #print  'border lines ix=-1, iy=1:-1 divQ max = ', divQ[:, 1:-1, -1].max() 
                            
    # ix=1:-1, iy=0                        
    divQ[:, 0, 1:-1] = ((data[:, 0, 2:].real - 
                         data[:, 0, :-2].real)/dx2) + \
                          ((data[:, 1, 1:-1].imag -
                            data[:, 0, 1:-1].imag)/dy)                             
    #print  '  '                     
    #print  'border lines ix=1:-1, iy=0 divQ min = ', divQ[:, 0, 1:-1].min()
    #print  'border lines ix=1:-1, iy=0 divQ max = ', divQ[:, 0, 1:-1].max()
      
    # ix=1:-1, iy=-1                        
    divQ[:, -1, 1:-1] = ((data[:, -1, 2:].real - 
                          data[:, -1, :-2].real)/dx2) + \
                          ((data[:, -1, 1:-1].imag -
                            data[:, -2, 1:-1].imag)/dy)  
    #print  '  '                     
    #print  'border lines ix=1:-1, iy=-1 divQ min = ', divQ[:, -1, 1:-1].min()
    #print  'border lines ix=1:-1, iy=-1 divQ max = ', divQ[:, -1, 1:-1].max()
                                
    # corner points
    # ix=0, iy=0                        
    divQ[:, 0, 0] = ((data[:, 0, 1].real - 
                      data[:, 0, 0].real)/dx) + \
                          ((data[:, 1, 0].imag -
                            data[:, 0, 0].imag)/dy)        
    # ix=0, iy=-1                        
    divQ[:, -1, 0] = ((data[:, -1, 1].real - 
                       data[:, -1, 0].real)/dx) + \
                          ((data[:, -1, 0].imag -
                            data[:, -2, 0].imag)/dy)                              
    # ix=-1, iy=0                        
    divQ[:, 0, -1] = ((data[:, 0, -1].real - 
                       data[:, 0, -2].real)/dx) + \
                          ((data[:, 1, -1].imag -
                            data[:, 0, -1].imag)/dy)                                
    # ix=-1, iy=-1                        
    divQ[:, -1, -1] = ((data[:, -1, -1].real - 
                        data[:, -1, -2].real)/dx) + \
                          ((data[:, -1, -1].imag -
                            data[:, -2, -1].imag)/dy)  
    return divQ
    
def lapzz_3d(data,dz):     
    '''
    lapzz_3d computes the vertical laplacian component
                 
    
                    lapdata = d²data/dz²

    INPUTS: 
        data:  complex 3d grided input array
        dz: grid spacing along z-direction (units: m)

   
    OUTPUTS: d²data/dz²

    USAGE: lapdata=lapzz_3d(data,8)
    
    Author: Enric Pallàs Sanz 
    To python: Bàrbara Barceló Llull, ULPGC, July 2015
    
    '''
    print ('  ')
    print ('lapzz_3d'  )    
    [nz, ny, nx] = data.shape         
       
    lapdata = np.zeros(data.shape)   
                              
    dz22 = dz**2
    
    # inner region
    lapdata[1:-1, :, :] = (data[2:, :, :] - 2*data[1:-1, :, :] + 
                           data[:-2, :, :])/dz22
    # border region
    lapdata[0, :, :]  = 2*lapdata[1, :, :]- lapdata[2, :, :]
    lapdata[-1, :, :] = 2*lapdata[-2, :, :]- lapdata[-3, :, :]    
    
    return lapdata 
 
def R_from_b(data,dz):       


    # R_from_b computes the vertically integrated b.  
    #
    #                     dR/dz = b
    #
    #                     UNITS= [s^-1]
    #  
    #  INPUTS:
    #
    #    data:  3d grided b (units: m s^-1)
    #    dz:    vertical grid spacing (units: m)
    #   
    #  
    #  OUTPUTS: a 3d complex array containing the R array.
    #  
    #  USAGE: R=R_from_b(data,0.2) 
    #
    #  
    #  Author: Enric Pallàs Sanz 
    #  To python: Bàrbara Barceló Llull, CICESE, March 2016
    #  
                                    
    print ('  ')
    print ('R_FROM_B')                                             
    [nz, ny, nx] = data.shape 
    
    R = np.zeros(data.shape)   
    
    R[0,:,:] = data[0,:,:]
    for iz in np.arange(1, nz):
        R[iz,:,:] = R[iz-1,:,:] + data[iz,:,:]*dz
        
    return R
                                                                                                                                   
def vgeos_from_gradb(data,dz,vadcp=False):
    '''
    function a=vgeos_from_gradb(data,dz,vadcp)
    
     vgeos_from_gradb computes the geostrophic velocity from a 3d array of
                      buoyancy horizontal gradient. backward 
                      derivates from the reference level.
                      vgeos[0,:,:] = vadcp[0,:,:] is the reference level
                      Reference level in iz=0 (bottom!)
                     
    
                        -k X d(vgeos)/dz = - gradb
    
                          UNITS= [m s^-1]
      
    
      INPUTS:
    
        data:  3d grided buoyancy horizontal gradient array (units: s^-1)
        dz:    vertical grid spacing (units: m)
       
      
      OUTPUTS: a 3d complex array containing the components of the geostrophic
               velocity. Real part corresponds to the zonal (ug) component and 
               Imaginary part to the meridional (vg) component.
      
      USAGE: vgeos=vgeos_from_gradb(data,0.2) 
    
      
      Author: Enric Pall‡s Sanz 
              Scripps Institution of Oceanography 
              CASPO 2008
      
      To python by: Bàrbara Barceló Llull, IMEDEA, May 2018
    
    '''
    
    [nz, ny, nx] = data.shape

    if vadcp==False:
        print('Vgeo with reference level set to zero')
        vgeos = np.zeros(data.shape) + 1j * np.zeros(data.shape)
        
    else:
        print('Vgeo with reference level set from ADCP file')
        vgeos = np.zeros(data.shape) + 1j * np.zeros(data.shape)
        vgeos[0,:,:] = vadcp[0,:,:]
        
    for iz in  np.arange(1, nz):  
        vgeos[iz,:,:] = ( vgeos[iz-1,:,:].real + data[iz,:,:].imag * dz ) + \
                         1j * ( vgeos[iz-1,:,:].imag - data[iz,:,:].real * dz )

    return vgeos
