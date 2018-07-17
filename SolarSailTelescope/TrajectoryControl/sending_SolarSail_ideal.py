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

Lr = 2e-2
alpha_r = np.pi/3
delta_r = np.pi/6

Lv = 3e-2
alpha_v = np.pi/3
delta_v = np.pi/6

x0 = np.array([ 9.999969595766015e-1,
               -1.007824043999068e-3,
                0])
v0 = np.array([ 7.237761726886938e-2,
                0,
                0])

L0 = np.array([Lr*np.cos(alpha_r),
               Lr*np.sin(alpha_r)*np.cos(delta_r),
               Lr*np.sin(alpha_r)*np.sin(delta_r),
               Lv*np.cos(alpha_v),
               Lv*np.sin(alpha_v)*np.cos(delta_v),
               Lv*np.sin(alpha_v)*np.sin(delta_v)])
              
#l0 = np.array([-3.42918206e-01,  9.36444073e-01, -1.96812520e-03, -2.13765223e-02,
#        8.85343229e-03, -5.96072730e-04])
#lf = np.array([ 0.00967418, -0.00770374,  0.00536951, -0.01269518, -0.01423319,
#        0.00039755])
#                
tF = 200 *(1/365.25)*2*np.pi #days
fs0 = np.hstack([x0,v0,L0,tF])               
fsF = np.hstack([VL,L0,tF])    



s,t_s = IS.send_it(fs0,fsF)
print s


ax.plot(s[:,0],s[:,1],s[:,2],linewidth=3,label='Trajectory')
ax.plot([x0[0]],[x0[1]],[x0[2]],'d',linewidth=3,label='Start')
ax.plot([VL[0]],[VL[1]],[VL[2]],'s',linewidth=3,label='Finish')
ax.legend()
