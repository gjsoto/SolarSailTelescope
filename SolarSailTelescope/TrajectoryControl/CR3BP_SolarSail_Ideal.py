from CR3BP_Thruster_Unconstrained import CR3BP_Thruster_Unconstrained
import numpy as np
import pylab as pl
import astropy as astro
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.interpolate as interpolate
import cPickle as pickle
from scipy.integrate import solve_bvp
from astropy.coordinates import SkyCoord
import astropy.constants as const
import scipy.optimize as optimize
import scipy.integrate as itg
from copy import deepcopy
import time

##################################################

class CR3BP_SolarSail_Ideal(CR3BP_Thruster_Unconstrained):
    
    def __init__(self,**specs) :
        CR3BP_Thruster_Unconstrained.__init__(self,**specs) 
        self.mu = const.M_earth/ (const.M_earth + const.M_sun)

        coeff = [1, 3-self.mu, 3-2*self.mu, -self.mu, -2*self.mu, -self.mu]
        roots = np.roots(coeff)
        self.g2 = np.real(roots[np.isreal(roots)])[0]
        self.L2 = self.g2 + (1-self.mu)
        self.gL = self.g2 / 1
        
        self.G = 1
        self.m1 = float(1-self.mu)
        self.m2 = self.mu
        
        S0 = 1368*u.W/u.m**2
        cs  = const.c
        r0 = 1*u.au
        
        P0 = S0*r0**2/cs
        P0 = P0.to('kg*au/yr**2')
        sigma = 10*u.g/u.m**2
        g0 = 2*P0/sigma
        self.g0 = g0.to('au**3/yr**2')/u.au**3*u.yr**2/(2*np.pi)**2
    
    def EoM(self,t,state):
        """Equations of motion of the CRTBP with Solar Radiation Pressure
        
        Equations of motion for the Circular Restricted Three Body 
        Problem (CRTBP). First order form of the equations for integration, 
        returns 3 velocities and 3 accelerations in (x,y,z) rotating frame.
        All parameters are normalized so that time = 2*pi sidereal year.
        Distances are normalized to 1AU. Coordinates are taken in a rotating 
        frame centered at the center of mass of the two primary bodies. Pitch
        angle of the starshade with respect to the Sun is assumed to be 60 
        degrees, meaning the 1/2 of the starshade cross sectional area is 
        always facing the Sun on average
        
        Args:
            t (float):
                Times in normalized units
            state (float nx6 array):
                State vector consisting of stacked position and velocity vectors
                in normalized units

        Returns:
            ds (integer Quantity nx6 array):
                First derivative of the state vector consisting of stacked 
                velocity and acceleration vectors in normalized units
        """
        
        mu = self.mu
        m1 = self.m1
        m2 = self.m2
        

        x,y,z,dx,dy,dz = state
        
        f = np.zeros(state.shape)

        #occulter distance from each of the two other bodies
        r1 = np.sqrt( (mu + x)**2 + y**2 + z**2 )
        r2 = np.sqrt( (1 - mu - x)**2 + y**2 + z**2 )
        
        #equations of motion
        f[0] = dx
        f[1] = dy
        f[2] = dz
        f[3] = x + 2*dy + m1*(-mu-x)/r1**3 + m2*(1-mu-x)/r2**3
        f[4] = y - 2*dx - m1*y/r1**3 - m2*y/r2**3
        f[5] = -m1*z/r1**3 - m2*z/r2**3
        
        return f
    
    def EoM_Adjoint_IS(self,t,state):
        
        mu = self.mu
        g0 = self.g0
        
        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = state
#        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6,tf = state

        f = np.zeros(state.shape)
        
        p1 = np.array([x+mu,y,z])
        p1 = p1/np.linalg.norm(p1,axis=0)
        
        Lv = np.array([L4,L5,L6])
        lv = Lv/np.linalg.norm(Lv,axis=0)
        
        crossP = np.cross( lv, p1,axis=0)
        BL = np.arctan2( np.linalg.norm( crossP , axis=0)*np.sign(crossP[2,:]), np.einsum('ij,ji->i', lv.T,p1))
        B  = np.arctan2( -(3*np.cos(BL) + np.sqrt(8*np.sin(BL)**2 + 9*np.cos(BL)**2)),4*np.sin(BL)) 
        
        f[0,:]   = dx
        f[1,:]   = dy
        f[2,:]   = dz
        f[3,:]   = 2*dy + g0*(L4*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + (mu + x)*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) + mu*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + x + (-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(3/2.)
        f[4,:]   = -2*dx + g0*(L5*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + y*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) - mu*y/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - y*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + y
        f[5,:]   = g0*(L6*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + z*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) - mu*z/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.)
        
        f[6,:]   = -L4*(g0*(-3*mu - 3*x)*(L4*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + (mu + x)*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) + g0*(L4*(mu + x)*np.sin(B)/np.sqrt(y**2 + z**2 + (mu + x)**2) + np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) + mu*(-3*mu - 3*x + 3)*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + (-3*mu - 3*x)*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + 1) - L5*(L5*g0*(mu + x)*np.sin(B)*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**2*np.sin(BL)) + g0*(-3*mu - 3*x)*(L5*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + y*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) - mu*y*(-3*mu - 3*x + 3)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - y*(-3*mu - 3*x)*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L6*(L6*g0*(mu + x)*np.sin(B)*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**2*np.sin(BL)) + g0*(-3*mu - 3*x)*(L6*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + z*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) - mu*z*(-3*mu - 3*x + 3)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - z*(-3*mu - 3*x)*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.))
        f[7,:]   = -L4*(L4*g0*y*np.sin(B)*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**2*np.sin(BL)) - 3*g0*y*(L4*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + (mu + x)*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) - 3*mu*y*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - 3*y*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L5*(-3*g0*y*(L5*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + y*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) + g0*(L5*y*np.sin(B)/np.sqrt(y**2 + z**2 + (mu + x)**2) + np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) + 3*mu*y**2/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + 3*y**2*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + 1) - L6*(L6*g0*y*np.sin(B)*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**2*np.sin(BL)) - 3*g0*y*(L6*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + z*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) + 3*mu*y*z/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) + 3*y*z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.))
        f[8,:]   = -L4*(L4*g0*z*np.sin(B)*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**2*np.sin(BL)) - 3*g0*z*(L4*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + (mu + x)*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) - 3*mu*z*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - 3*z*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L5*(L5*g0*z*np.sin(B)*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**2*np.sin(BL)) - 3*g0*z*(L5*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + y*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) + 3*mu*y*z/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) + 3*y*z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L6*(-3*g0*z*(L6*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + z*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(5/2.)*np.sin(BL)) + g0*(L6*z*np.sin(B)/np.sqrt(y**2 + z**2 + (mu + x)**2) + np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) + 3*mu*z**2/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + 3*z**2*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.))
        f[9,:]    = -L1 + 2*L5
        f[10,:]   = -L2 - 2*L4
        f[11,:]   = -L3
#        f[12,:]   = np.zeros(tf.shape)
        
#        return np.dot(f,np.diag(tf))
        return f
    
    def boundary_conditions_IS(self,sA,sB):
        """Creates boundary conditions for solving a boundary value problem
        """
        
        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = sB
#        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6,tf = sB
        mu = self.mu
        g0 = self.g0
        
        p1 = np.array([x+mu,y,z])
        p1 = p1/np.linalg.norm(p1,axis=0)
        
        Lv = np.array([L4,L5,L6])
        lv = Lv/np.linalg.norm(Lv,axis=0)
        
        crossP = np.cross( lv, p1,axis=0)
        BL = np.arctan2( np.linalg.norm( crossP , axis=0)*np.sign(crossP[2]), np.dot(lv,p1))
        B  = np.arctan2( -(3*np.cos(BL) + np.sqrt(8*np.sin(BL)**2 + 9*np.cos(BL)**2)),4*np.sin(BL)) 
    
        BCo1 = (sA[0] - self.sA[0])
        BCo2 = (sA[1] - self.sA[1])
        BCo3 = (sA[2] - self.sA[2])
        BCo4 = (sA[3] - self.sA[3])
        BCo5 = (sA[4] - self.sA[4])
        BCo6 = (sA[5] - self.sA[5])
        
        BCf1 = (sB[0] - self.sB[0])
        BCf2 = (sB[1] - self.sB[1])
        BCf3 = (sB[2] - self.sB[2])
        BCf4 = (sB[3] - self.sB[3])
        BCf5 = (sB[4] - self.sB[4])
        BCf6 = (sB[5] - self.sB[5])
        
#        HB = L1*dx + L2*dy + L3*dz + L4*(2*dy + g0*(L4*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + (mu + x)*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) + mu*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + x + (-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(3/2.)) + L5*(-2*dx + g0*(L5*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + y*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) - mu*y/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - y*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + y) + L6*(g0*(L6*np.sqrt(y**2 + z**2 + (mu + x)**2)*np.sin(B) + z*np.sqrt(L4**2 + L5**2 + L6**2)*np.sin(B - BL))*np.cos(B)**2/(np.sqrt(L4**2 + L5**2 + L6**2)*(y**2 + z**2 + (mu + x)**2)**(3/2.)*np.sin(BL)) - mu*z/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.))
#        HB += 1    
        
#        BC = np.array([BCo1,BCo2,BCo3,BCo4,BCo5,BCo6,BCf1,BCf2,BCf3,BCf4,BCf5,BCf6,HB])
        BC = np.array([BCo1,BCo2,BCo3,BCo4,BCo5,BCo6,BCf1,BCf2,BCf3,BCf4,BCf5,BCf6])
        
        return BC
    
    def send_it(self,fs0,fsF):
        
        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6,T = fs0
        self.sA = np.array([x,y,z,dx,dy,dz])

        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6,T = fsF
        self.sB = np.array([x,y,z,dx,dy,dz])

#        t = np.linspace(0,1,2)
        t = np.linspace(0,T,2)
#        sG = np.vstack([ fs0 , fsF ])
        sG = np.vstack([ fs0[0:12] , fsF[0:12] ])
        sol_TU = solve_bvp(self.EoM_Adjoint,self.boundary_conditions,t,sG.T,tol=1e-10)
        s_TU = sol_TU.y.T
        t_TU = sol_TU.x
        
        sol = solve_bvp(self.EoM_Adjoint_IS,self.boundary_conditions_IS,t_TU,s_TU.T,tol=1e-10)
        print(sol.message)
        
        s = sol.y.T
        t_s = sol.x
            
        return s,t_s
    
    