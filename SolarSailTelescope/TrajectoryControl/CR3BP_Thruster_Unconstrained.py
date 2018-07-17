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

class CR3BP_Thruster_Unconstrained(object):
    
    def __init__(self):
        self.mu = const.M_earth/ (const.M_earth + const.M_sun)

        coeff = [1, 3-self.mu, 3-2*self.mu, -self.mu, -2*self.mu, -self.mu]
        roots = np.roots(coeff)
        self.g2 = np.real(roots[np.isreal(roots)])[0]
        self.L2 = self.g2 + (1-self.mu)
        self.gL = self.g2 / 1
        
        self.G = 1
        self.m1 = float(1-self.mu)
        self.m2 = self.mu
    
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
    
    def EoM_Adjoint(self,t,state):
        
        mu = self.mu
        
        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = state

        f = np.zeros(state.shape)
        
        f[0,:]   = dx
        f[1,:]   = dy
        f[2,:]   = dz
        f[3,:]   = -L4 + 2*dy + mu*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + x + (-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(3/2.)
        f[4,:]   = -L5 - 2*dx - mu*y/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - y*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + y
        f[5,:]   = -L6 - mu*z/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.)
        
        f[6,:]    = -L4*(mu*(-3*mu - 3*x + 3)*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + (-3*mu - 3*x)*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + 1) - L5*(-mu*y*(-3*mu - 3*x + 3)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - y*(-3*mu - 3*x)*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L6*(-mu*z*(-3*mu - 3*x + 3)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - z*(-3*mu - 3*x)*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.))
        f[7,:]    = -L4*(-3*mu*y*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - 3*y*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L5*(3*mu*y**2/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + 3*y**2*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + 1) - L6*(3*mu*y*z/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) + 3*y*z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.))
        f[8,:]    = -L4*(-3*mu*z*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - 3*z*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L5*(3*mu*y*z/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) + 3*y*z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L6*(3*mu*z**2/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + 3*z**2*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.))
        f[9,:]    = -L1 + 2*L5
        f[10,:]   = -L2 - 2*L4
        f[11,:]   = -L3
        
        return f
    
    def boundary_conditions(self,sA,sB):
        """Creates boundary conditions for solving a boundary value problem
        """
    
        BCo1 = sA[0] - self.sA[0]
        BCo2 = sA[1] - self.sA[1]
        BCo3 = sA[2] - self.sA[2]
        BCo4 = sA[3] - self.sA[3]
        BCo5 = sA[4] - self.sA[4]
        BCo6 = sA[5] - self.sA[5]
        
        BCf1 = sB[0] - self.sB[0]
        BCf2 = sB[1] - self.sB[1]
        BCf3 = sB[2] - self.sB[2]
        BCf4 = sB[3] - self.sB[3]
        BCf5 = sB[4] - self.sB[4]
        BCf6 = sB[5] - self.sB[5]
        
        BC = np.array([BCo1,BCo2,BCo3,BCo4,BCo5,BCo6,BCf1,BCf2,BCf3,BCf4,BCf5,BCf6])
        
        return BC
    
    def send_it(self,fs0,fsF,tF):
        
        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = fs0
        self.sA = np.array([x,y,z,dx,dy,dz])

        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = fsF
        self.sB = np.array([x,y,z,dx,dy,dz])

        t = np.linspace(0,tF,2)
        t = t*(1/365.25)*2*np.pi
        
        sG = np.vstack([ fs0 , fsF ])
        sol = solve_bvp(self.EoM_Adjoint,self.boundary_conditions,t,sG.T,tol=1e-10)
        
        s = sol.y.T
        t_s = sol.x
            
        return s,t_s
        