import numpy as np
import pylab as pl
import astropy as astro
import astropy.units as u
from astropy.time import Time
import astropy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import cPickle as pickle
from scipy.integrate import solve_bvp
from astropy.coordinates import SkyCoord
import scipy.optimize as optimize
import scipy.integrate as itg
from copy import deepcopy
import time
EPS = np.finfo(float).eps

import CR3BP_SolarSail_Ideal as opt
IS = opt.CR3BP_SolarSail_Ideal()


##################################################

## Final State Conditions
pklfile = open('VerticalLyapunovs_IC.p','rb')
D1 = pickle.load(pklfile)
pklfile.close()
        
VL  = D1['IC'][:,-1]
x,y,z,dx,dy,dz = VL
T   = D1['T'][-1]

sol = itg.solve_ivp(IS.EoM, [0,T], VL, method = 'Radau',rtol=4*EPS,atol=4*EPS)
sVL = sol.y

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X (0.001 AU)')
ax.set_ylabel('Y (0.001 AU)')
ax.set_zlabel('Z (0.001 AU)')
ax.plot(sVL[0,:],sVL[1,:],sVL[2,:],linewidth=3,label='Final Orbit')

## Initial Conditions

x0 = np.array([ 9.999969595766015e-1,
               -1.007824043999068e-3,
                0])
v0 = np.array([ 7.237761726886938e-2,
                0,
                0])

l0 = np.array([0.17062516, 1.42140934, 0.51488129, 0.4650598 , 0.56437841,
       0.49392316])*10
lf = np.array([0.51524846, 0.48574114, 0.48738081, 0.49665747, 0.4719813 ,
       0.52557947])*10

        
tF = 120 *(1/365.25)*2*np.pi #days
fs0 = np.hstack([x0,v0,l0,tF])               
fsF = np.hstack([VL,lf,tF])       
    
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X (0.001 AU)')
ax.set_ylabel('Y (0.001 AU)')
ax.set_zlabel('Z (0.001 AU)')
ax.plot(sVL[0,:],sVL[1,:],sVL[2,:],linewidth=3,label='Final Orbit')
    
    
s,t_s = IS.send_it(fs0,fsF)
print '--------------'
print 'Lambdas ',s[0,6:12]
print 'sA error ',IS.sA - s[ 0,0:6]
print 'sB error ',IS.sB - s[-1,0:6]
    


    
ax.plot(s[:,0],s[:,1],s[:,2],linewidth=3,label='Trajectory')
ax.plot([x0[0]],[x0[1]],[x0[2]],'d',linewidth=3,label='Start')
ax.plot([VL[0]],[VL[1]],[VL[2]],'s',linewidth=3,label='Finish')
ax.legend()

