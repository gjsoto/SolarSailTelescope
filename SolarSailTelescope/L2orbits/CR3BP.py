import numpy as np
import pylab as pl
import astropy as astro
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import astropy.constants as const
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.interpolate as interpolate
import cPickle as pickle
from scipy.integrate import solve_bvp
from astropy.coordinates import SkyCoord
import scipy.optimize as optimize
import scipy.integrate as itg
from copy import deepcopy
import time
EPS = np.finfo(float).eps

##################################################

class CR3BP(object):
    
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
        
     
    def equationsOfMotion_T(self,t,state):
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

        x,y,z,dx,dy,dz,tf = state
        
        f = np.zeros(state.shape)

        #occulter distance from each of the two other bodies
        r1 = np.sqrt( (mu + x)**2 + y**2 + z**2 )
        r2 = np.sqrt( (1 - mu - x)**2 + y**2 + z**2 )
        
        #equations of motion
        f[0,:] = dx
        f[1,:] = dy
        f[2,:] = dz
        f[3,:] = x + 2*dy + m1*(-mu-x)/r1**3 + m2*(1-mu-x)/r2**3
        f[4,:] = y - 2*dx - m1*y/r1**3 - m2*y/r2**3
        f[5,:] = -m1*z/r1**3 - m2*z/r2**3
        f[6,:]  = np.zeros(tf.shape)
        
        return np.dot(f,np.diag(tf))
    
    
    def equationsOfMotion(self,t,state,integrate_42=False):
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
        
        if integrate_42:
            x  = state[0]
            y  = state[1]
            z  = state[2]
            dx = state[3]
            dy = state[4]
            dz = state[5]
        else:
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
        
        if integrate_42:
            J = self.jacobian(state,False).flatten()
            f[6:] = J
            
        return f

    def c_fun(self,n):
        
        g  = self.g2
        cN = (1/g**3) * ( ((-1)**n)*self.mu + \
                ((-1)**n)*(1-self.mu)*g**(n+1)/(1+g)**(n+1))
        
        return cN
    
    def jacobian(self,s,get_Hessian=False):
        
        x  = s[0]
        y  = s[1]
        z  = s[2]
        
        Phi = s[6:].reshape([6,6])
        
        mu = self.mu
        
        r1 = np.sqrt( (mu + x)**2 + y**2 + z**2 )
        r2 = np.sqrt( (1 - mu - x)**2 + y**2 + z**2 )
    
        
        uxx= mu*(-3*mu - 3*x + 3)*(-mu - x + 1)/r2**5 - mu/r2**3 + (-3*mu - 3*x)*(-mu + 1)*(-mu - x)/r1**5 - (-mu + 1)/r1**3 + 1
        uyy= 3*mu*y**2/r2**5 - mu/r2**3 + 3*y**2*(-mu + 1)/r1**5 - (-mu + 1)/r1**3 + 1
        uzz= 3*mu*z**2/r2**5 - mu/r2**3 + 3*z**2*(-mu + 1)/r1**5 - (-mu + 1)/r1**3
        uxy= -3*mu*y*(-mu - x + 1)/r2**5 - 3*y*(-mu + 1)*(-mu - x)/r1**5
        uxz= -3*mu*z*(-mu - x + 1)/r2**5 - 3*z*(-mu + 1)*(-mu - x)/r1**5
        uyz= 3*mu*y*z/r2**5 + 3*y*z*(-mu + 1)/r1**5
        

        U  = np.array([[uxx, uxy, uxz], 
                       [uxy, uyy, uyz], 
                       [uxz, uyz, uzz]])
                       
        # dx,dy,dz wrt to x,y,z
        # this part of the jacobian has size 3 x 3 x m
        Z = np.zeros([3,3])
        
        # dx,dy,dz wrt to dx,dy,dz
        E = np.eye(3)

        # ddx,ddy,ddz wrt to dx,dy,dz
        W = np.array([[ 0 , 2 , 0],
                      [-2 , 0 , 0],
                      [ 0 , 0 , 0]])
        
        # stacking the different matrix blocks into a matrix 6 x 6 x m
        row1 = np.hstack( [ Z , E ])
        row2 = np.hstack( [ U , W ])

        J = np.matmul(np.vstack( [ row1, row2 ]),Phi)
        
#        if get_Hessian:
#            Hx,Hy,Hz = self.hessian(s) #3x3 arrays, each
#            
#            column1 = np.vstack( [ J , np.zeros([18,6]) ])
#            Hx = np.hstack( [ Hx , Z ])
#            Hy = np.hstack( [ Hy , Z ])
#            Hz = np.hstack( [ Hz , Z ])
#            
#            column1 = np.vstack( [ column1 , Hx , np.zeros([3,6]) , Hy , np.zeros([3,6]) , Hz , np.zeros([3,6]) ])
#            
#            J = np.hstack( [ column1 , np.zeros([42,36]) ])

        return J
        
    def hessian(self,s):
        
        x,y,z = s[0:3]
        mu = self.mu
        
        r1 = np.sqrt( (mu + x)**2 + y**2 + z**2 )
        r2 = np.sqrt( (1 - mu - x)**2 + y**2 + z**2 )
        
        #derivatives in X
        Uxxx = mu*(-5*mu - 5*x + 5)*(-3*mu - 3*x + 3)*(-mu - x + 1)/r2**7 - 2*mu*(-3*mu - 3*x + 3)/r2**5 - 3*mu*(-mu - x + 1)/r2**5 + (-5*mu - 5*x)*(-3*mu - 3*x)*(-mu + 1)*(-mu - x)/r1**7 - 2*(-3*mu - 3*x)*(-mu + 1)/r1**5 - 3*(-mu + 1)*(-mu - x)/r1**5
        Uxxy = -5*mu*y*(-3*mu - 3*x + 3)*(-mu - x + 1)/r2**7 + 3*mu*y/r2**5 - 5*y*(-3*mu - 3*x)*(-mu + 1)*(-mu - x)/r1**7 + 3*y*(-mu + 1)/r1**5
        Uxxz = -5*mu*z*(-3*mu - 3*x + 3)*(-mu - x + 1)/r2**7 + 3*mu*z/r2**5 - 5*z*(-3*mu - 3*x)*(-mu + 1)*(-mu - x)/r1**7 + 3*z*(-mu + 1)/r1**5
        
        Uxyx = -3*mu*y*(-5*mu - 5*x + 5)*(-mu - x + 1)/r2**7 + 3*mu*y/r2**5 - 3*y*(-5*mu - 5*x)*(-mu + 1)*(-mu - x)/r1**7 + 3*y*(-mu + 1)/r1**5
        Uxyy = 15*mu*y**2*(-mu - x + 1)/r2**7 - 3*mu*(-mu - x + 1)/r2**5 + 15*y**2*(-mu + 1)*(-mu - x)/r1**7 - 3*(-mu + 1)*(-mu - x)/r1**5
        Uxyz = 15*mu*y*z*(-mu - x + 1)/r2**7 + 15*y*z*(-mu + 1)*(-mu - x)/r1**7
        
        Uxzx = -3*mu*z*(-5*mu - 5*x + 5)*(-mu - x + 1)/r2**7 + 3*mu*z/r2**5 - 3*z*(-5*mu - 5*x)*(-mu + 1)*(-mu - x)/r1**7 + 3*z*(-mu + 1)/r1**5
        Uxzy = 15*mu*y*z*(-mu - x + 1)/r2**7 + 15*y*z*(-mu + 1)*(-mu - x)/r1**7
        Uxzz = 15*mu*z**2*(-mu - x + 1)/r2**7 - 3*mu*(-mu - x + 1)/r2**5 + 15*z**2*(-mu + 1)*(-mu - x)/r1**7 - 3*(-mu + 1)*(-mu - x)/r1**5
        
        #derivatives in Y
        Uyxx = -mu*y*(-5*mu - 5*x + 5)*(-3*mu - 3*x + 3)/r2**7 + 3*mu*y/r2**5 - y*(-5*mu - 5*x)*(-3*mu - 3*x)*(-mu + 1)/r1**7 + 3*y*(-mu + 1)/r1**5
        Uyxy = 5*mu*y**2*(-3*mu - 3*x + 3)/r2**7 - mu*(-3*mu - 3*x + 3)/r2**5 + 5*y**2*(-3*mu - 3*x)*(-mu + 1)/r1**7 - (-3*mu - 3*x)*(-mu + 1)/r1**5
        Uyxz = 5*mu*y*z*(-3*mu - 3*x + 3)/r2**7 + 5*y*z*(-3*mu - 3*x)*(-mu + 1)/r1**7
        
        Uyyx = 3*mu*y**2*(-5*mu - 5*x + 5)/r2**7 - mu*(-3*mu - 3*x + 3)/r2**5 + 3*y**2*(-5*mu - 5*x)*(-mu + 1)/r1**7 - (-3*mu - 3*x)*(-mu + 1)/r1**5
        Uyyy = -15*mu*y**3/r2**7 + 9*mu*y/r2**5 - 15*y**3*(-mu + 1)/r1**7 + 9*y*(-mu + 1)/r1**5
        Uyyz = -15*mu*y**2*z/r2**7 + 3*mu*z/r2**5 - 15*y**2*z*(-mu + 1)/r1**7 + 3*z*(-mu + 1)/r1**5
        
        Uyzx = 3*mu*y*z*(-5*mu - 5*x + 5)/r2**7 + 3*y*z*(-5*mu - 5*x)*(-mu + 1)/r1**7
        Uyzy = -15*mu*y**2*z/r2**7 + 3*mu*z/r2**5 - 15*y**2*z*(-mu + 1)/r1**7 + 3*z*(-mu + 1)/r1**5
        Uyzz = -15*mu*y*z**2/r2**7 + 3*mu*y/r2**5 - 15*y*z**2*(-mu + 1)/r1**7 + 3*y*(-mu + 1)/r1**5
        
        #derivatives in Z
        Uzxx = -mu*z*(-5*mu - 5*x + 5)*(-3*mu - 3*x + 3)/r2**7 + 3*mu*z/r2**5 - z*(-5*mu - 5*x)*(-3*mu - 3*x)*(-mu + 1)/r1**7 + 3*z*(-mu + 1)/r1**5
        Uzxy = 5*mu*y*z*(-3*mu - 3*x + 3)/r2**7 + 5*y*z*(-3*mu - 3*x)*(-mu + 1)/r1**7
        Uzxz = 5*mu*z**2*(-3*mu - 3*x + 3)/r2**7 - mu*(-3*mu - 3*x + 3)/r2**5 + 5*z**2*(-3*mu - 3*x)*(-mu + 1)/r1**7 - (-3*mu - 3*x)*(-mu + 1)/r1**5
        
        Uzyx = 3*mu*y*z*(-5*mu - 5*x + 5)/r2**7 + 3*y*z*(-5*mu - 5*x)*(-mu + 1)/r1**7
        Uzyy = -15*mu*y**2*z/r2**7 + 3*mu*z/r2**5 - 15*y**2*z*(-mu + 1)/r1**7 + 3*z*(-mu + 1)/r1**5
        Uzyz = -15*mu*y*z**2/r2**7 + 3*mu*y/r2**5 - 15*y*z**2*(-mu + 1)/r1**7 + 3*y*(-mu + 1)/r1**5
        
        Uzzx = 3*mu*z**2*(-5*mu - 5*x + 5)/r2**7 - mu*(-3*mu - 3*x + 3)/r2**5 + 3*z**2*(-5*mu - 5*x)*(-mu + 1)/r1**7 - (-3*mu - 3*x)*(-mu + 1)/r1**5
        Uzzy = -15*mu*y*z**2/r2**7 + 3*mu*y/r2**5 - 15*y*z**2*(-mu + 1)/r1**7 + 3*y*(-mu + 1)/r1**5
        Uzzz = -15*mu*z**3/r2**7 + 9*mu*z/r2**5 - 15*z**3*(-mu + 1)/r1**7 + 9*z*(-mu + 1)/r1**5
        
        Hx  = np.array([[Uxxx, Uxxy, Uxxz], 
                        [Uxyx, Uxyy, Uxyz], 
                        [Uxzx, Uxzy, Uxzz]])
        
        Hy  = np.array([[Uyxx, Uyxy, Uyxz], 
                        [Uyyx, Uyyy, Uyyz], 
                        [Uyzx, Uyzy, Uyzz]])
                        
        Hz  = np.array([[Uzxx, Uzxy, Uzxz], 
                        [Uzyx, Uzyy, Uzyz], 
                        [Uzzx, Uzzy, Uzzz]])
        
        return Hx,Hy,Hz
        
    def integrate(self,s0,t,events,t_eval=None,integrate_42=False):
        """Integrates motion in the CRTBP given initial conditions
        
        This method returns a star's position vector in the rotating frame of 
        the Circular Restricted Three Body Problem.  
        
        Args:
            s0 (integer 1x6 array):
                Initial state vector consisting of stacked position and velocity vectors
                in normalized units
            t (integer):
                Times in normalized units

        Returns:
            s (integer nx6 array):
                State vector consisting of stacked position and velocity vectors
                in normalized units
        """
        
        EoM = lambda t,s: self.equationsOfMotion(t,s,integrate_42)
#        JCB = lambda t,s: self.jacobian(s,integrate_42)

        sol = itg.solve_ivp(EoM, t, s0, method = 'Radau', t_eval = t_eval, \
           events=events, rtol=4*EPS,atol=4*EPS)
        s = sol.y
        
        if sol.status != 0:
                print("WARNING: IVP sol.status is %d" % sol.status)
        print(sol.message)
        
        if events == None:
            return s
        else:
            return s,sol.t_events
    
